"""
Canonical mission execution task.

Imported by:
  - app/api/main.py        (FastAPI background tasks)
  - app/core/scheduler.py  (recurring daily jobs)

This is the single source of truth for the Ingest → Analyze → Report pipeline.

Credit governance (enforced here, not only at the API layer):
  1. Credits are PENDED (held) before any S1/S2 data fetch begins.
  2. Credits are DEDUCTED only after the PDF Evidence Pack is generated.
  3. On any failure the hold is RELEASED — failed jobs cost nothing.
  4. Every job produces an immutable usage_log row regardless of outcome.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional

from sentinelhub import BBox

from app.ingestion.s2_client import S2Client
from app.ingestion.sentinel1_rtc import S1RTCClient
from app.analytics import processor
from app.analytics.chronos import AutoDifferencing, GifGenerator
from app.reporting.pdf_gen import PDFReportGenerator
from app.reporting.kml_gen import generate_kml
from app.reporting.summary_gen import generate_mobile_summary
from app.api.job_manager import JobManager
from app.api.usage_controller import UsageController, InsufficientCredits
from app.core.config import SCOUT_DB_PATH, RESULTS_DIR

logger = logging.getLogger(__name__)

DB_PATH = SCOUT_DB_PATH


def _parse_datetime(value) -> datetime:
    """Coerce a string or datetime to datetime."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def process_mission_task(
    job_id: str,
    bbox: BBox,
    time_interval: tuple,
    sensor: str,
) -> None:
    """
    Run the full Ingest → Analyze → Report pipeline for *job_id*.
    Blocking — intended to run in a thread (asyncio.to_thread / ThreadPoolExecutor).

    Credit flow:
      pend_credits()  → before any network I/O
      deduct_credits()→ after successful PDF generation
      release_credits()→ on any exception (including failed downloads)
    """
    jm = JobManager(db_path=DB_PATH)
    jm.update_job_status(job_id, "RUNNING")

    # ------------------------------------------------------------------
    # Resolve execution context
    # ------------------------------------------------------------------
    job = jm.get_job(job_id) or {}
    username: str = job.get("launched_by", "unknown")

    # Geometry metrics needed for cost calculation
    lon_min, lat_min = bbox.lower_left
    lon_max, lat_max = bbox.upper_right

    uc = UsageController(db_path=DB_PATH)
    area_km2 = uc.calc_bbox_area_km2(lon_min, lat_min, lon_max, lat_max)

    start_dt = _parse_datetime(time_interval[0])
    end_dt   = _parse_datetime(time_interval[1])
    temporal_depth = max(1, (end_dt - start_dt).days + 1)

    cost = uc.calculate_mission_cost(area_km2, sensor, temporal_depth)

    # ------------------------------------------------------------------
    # Credit Lock — pend before touching Copernicus/CDSE endpoints
    # ------------------------------------------------------------------
    ledger_id: Optional[int] = None
    try:
        ledger_id = uc.pend_credits(username, job_id, cost)
    except InsufficientCredits as exc:
        logger.warning(f"Job {job_id} aborted: credit lock failed — {exc}")
        jm.update_job_status(job_id, "FAILED", error=f"Credit lock failed: {exc}")
        uc.log_usage(
            job_id, username, 0.0, area_km2, sensor, temporal_depth, "FAILED"
        )
        return

    try:
        # ----------------------------------------------------------------
        # 1. Ingestion
        # ----------------------------------------------------------------
        raw_files: List[str] = []
        requested_indices: Optional[List[str]] = None

        if sensor in ["OPTICAL", "NDVI", "NDWI", "NBR"]:
            raw_files = S2Client().download_data(bbox=bbox, time_interval=time_interval)
            requested_indices = (
                ["NDVI", "NDWI", "NBR"] if sensor == "OPTICAL" else [sensor]
            )
        elif sensor in ["RADAR", "S1_RTC"]:
            raw_files = S1RTCClient().download_data(bbox=bbox, time_interval=time_interval)
            requested_indices = []
        else:
            raw_files = S2Client().download_data(bbox=bbox, time_interval=time_interval)
            requested_indices = ["NDVI"]

        if not raw_files:
            raise RuntimeError("No data downloaded from Copernicus/CDSE.")

        # ----------------------------------------------------------------
        # 2. Analytics
        # ----------------------------------------------------------------
        if sensor in ["RADAR", "S1_RTC"]:
            # S1 files contain VV/VH backscatter — no spectral indices apply.
            processed_files = raw_files
        else:
            processed_files = processor.run(
                input_files=raw_files, requested_indices=requested_indices
            )

        if not processed_files:
            raise RuntimeError("Analytics produced no output.")

        # ----------------------------------------------------------------
        # 3. Chronos (temporal differencing + timelapse)
        # ----------------------------------------------------------------
        gif_path: Optional[str] = None
        try:
            bbox_list = [lon_min, lat_min, lon_max, lat_max]
            diff_engine = AutoDifferencing()
            previous_job = diff_engine.find_previous_job(job_id, bbox_list, sensor)

            if previous_job:
                prev_files = previous_job.get("results", {}).get("processed_files", [])
                prev_ndvi = next((f for f in prev_files if "NDVI" in f), None)
                curr_ndvi = next((f for f in processed_files if "NDVI" in f), None)
                if prev_ndvi and curr_ndvi and os.path.exists(prev_ndvi):
                    date_str = os.path.basename(curr_ndvi).split("_")[0]
                    diff_map = diff_engine.compute_difference(curr_ndvi, prev_ndvi, date_str)
                    if diff_map:
                        processed_files.append(diff_map)

            gif_path = GifGenerator().generate_timelapse(job_id, bbox_list, processed_files)

        except Exception as exc:
            logger.warning(f"Chronos engine warning for job {job_id}: {exc}")

        # ----------------------------------------------------------------
        # 4. Reporting
        # ----------------------------------------------------------------
        os.makedirs(RESULTS_DIR, exist_ok=True)

        kml_path    = generate_kml(job_id, bbox, output_dir=RESULTS_DIR)
        mobile_path = generate_mobile_summary(
            processed_files, output_dir=RESULTS_DIR, job_id=job_id
        )

        report_name = f"Evidence_Pack_{job_id}.pdf"
        PDFReportGenerator().generate_pdf(
            filename=report_name,
            specific_files=processed_files,
            bbox=bbox,
            job_id=job_id,
        )
        report_path = os.path.join(RESULTS_DIR, report_name)

        # ----------------------------------------------------------------
        # Credit Deduction — only after successful PDF generation
        # ----------------------------------------------------------------
        uc.deduct_credits(username, job_id, ledger_id, area_km2, cost)
        uc.log_usage(job_id, username, cost, area_km2, sensor, temporal_depth, "COMPLETED")

        # ----------------------------------------------------------------
        # 5. Complete
        # ----------------------------------------------------------------
        results = {
            "raw_files":      raw_files,
            "processed_files": processed_files,
            "pdf_report":     report_path,
            "kml_export":     kml_path,
            "mobile_summary": mobile_path,
        }
        if gif_path:
            results["timelapse_gif"] = gif_path

        jm.update_job_status(job_id, "COMPLETED", results=results)
        logger.info(
            f"Job {job_id} completed | credits_deducted={cost:.2f} "
            f"area={area_km2:.2f} km² sensor={sensor}"
        )

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)

        # Release the credit hold — failed jobs cost nothing
        if ledger_id is not None:
            uc.release_credits(username, job_id, ledger_id, cost)

        uc.log_usage(job_id, username, 0.0, area_km2, sensor, temporal_depth, "FAILED")
        jm.update_job_status(job_id, "FAILED", error=str(exc))
