import asyncio
import json
import logging
import os
import sys
import rasterio
import numpy as np
from datetime import datetime, timedelta, timezone
from sentinelhub import BBox, CRS
from typing import List

from app.ingestion.s2_client import S2Client
from app.analytics import processor
from app.reporting.pdf_gen import PDFReportGenerator

logger = logging.getLogger("Supervisor")

TASK_FILE = "tasks.json"

def load_tasks():
    if not os.path.exists(TASK_FILE):
        logger.error(f"Task file {TASK_FILE} not found.")
        return []
    with open(TASK_FILE, 'r') as f:
        data = json.load(f)
    return data.get("tasks", [])

_COREG_TOLERANCE_PCT = 1.0   # max valid-pixel-count variance across acquisitions


def _count_valid_pixels(filepath: str) -> int:
    """
    Return the number of finite, non-NaN pixels in the first band of a GeoTIFF.
    Returns 0 on any read error.
    """
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(float)
        return int(np.sum(np.isfinite(data)))
    except Exception as exc:
        logger.warning(f"Could not count valid pixels in {filepath}: {exc}")
        return 0


def quality_gate_coregistration(file_paths: List[str]) -> bool:
    """
    Inter-scene co-registration check.

    Reads the valid (finite) pixel count from every GeoTIFF in *file_paths* and
    verifies that counts are consistent across acquisitions.  A swing of more
    than _COREG_TOLERANCE_PCT between the minimum and maximum count indicates
    misaligned grids, making pixel-level delta comparisons unreliable.

    Args:
        file_paths: Ordered list of downloaded scene paths for one task.

    Returns:
        True  — all scenes share a consistent pixel grid; Delta Report may proceed.
        False — co-registration FAILED; Delta Report must be suppressed.
    """
    if len(file_paths) < 2:
        return True  # nothing to compare

    counts = {fp: _count_valid_pixels(fp) for fp in file_paths}
    values = list(counts.values())
    max_px = max(values)

    if max_px == 0:
        logger.error("Co-registration gate: all scenes returned 0 valid pixels.")
        return False

    min_px  = min(values)
    var_pct = (max_px - min_px) / max_px * 100.0

    if var_pct > _COREG_TOLERANCE_PCT:
        logger.error(
            "Co-registration gate FAILED: valid-pixel count varies %.2f%% "
            "across %d acquisition(s) — exceeds %.1f%% tolerance. "
            "Scene counts: %s. Delta Report suppressed.",
            var_pct, len(file_paths), _COREG_TOLERANCE_PCT,
            {os.path.basename(fp): c for fp, c in counts.items()},
        )
        return False

    logger.info(
        "Co-registration gate PASSED: pixel-count variance %.2f%% across %d scene(s).",
        var_pct, len(file_paths),
    )
    return True


def check_alpha_health(raw_file_path, expected_bbox_coords):
    logger.debug(f"Health check: {raw_file_path}")

    # Checklist Item 1: File size > 0
    if not os.path.exists(raw_file_path):
        logger.error("Health Check Failed: File does not exist.")
        return False
    if os.path.getsize(raw_file_path) == 0:
        logger.error("Health Check Failed: File size is 0.")
        return False

    # Checklist Item 2: Image is not all-black (Mean > 100)
    try:
        with rasterio.open(raw_file_path) as src:
            # Check first band (B04) or all?
            # "Image is not all-black".
            # We assume at least one band should be meaningful.
            data = src.read(1)
            mean_val = np.mean(data)
            if mean_val <= 100:
                logger.error(f"Health Check Failed: Image mean {mean_val:.2f} <= 100 (All black?).")
                return False
    except Exception as e:
        logger.error(f"Health Check Failed: Error reading image: {e}")
        return False

    # Checklist Item 3: Metadata matches the BBox
    # Metadata is in _provenance.json
    meta_path = raw_file_path.replace(".tif", "_provenance.json")
    if not os.path.exists(meta_path):
        logger.error(f"Health Check Failed: Metadata file {meta_path} missing.")
        return False

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Provenance bbox: [minx, miny, maxx, maxy]
        meta_bbox = meta.get("bbox")
        if not meta_bbox:
            logger.error("Health Check Failed: No bbox in metadata.")
            return False

        # Compare approx
        # expected_bbox_coords is [16.2, 48.1, 16.5, 48.3]
        # Allow small tolerance
        if not np.allclose(meta_bbox, expected_bbox_coords, atol=0.01):
             logger.error(f"Health Check Failed: BBox mismatch. Expected {expected_bbox_coords}, got {meta_bbox}")
             return False

    except Exception as e:
        logger.error(f"Health Check Failed: Error checking metadata: {e}")
        return False

    logger.debug("Health check passed.")
    return True

async def supervisor_main():
    tasks = load_tasks()
    if not tasks:
        logger.warning("No tasks found.")
        return

    client = S2Client()
    pdf_gen = PDFReportGenerator()

    # Time interval: Last 30 days
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)
    time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    for task in tasks:
        task_name = task["name"]
        bbox_coords = task["bbox"]
        logger.info(f"Task: {task_name}")

        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)

        # 1. Alpha (Ingestion)
        try:
            downloaded_files = await asyncio.to_thread(
                client.download_data,
                bbox=bbox,
                time_interval=time_interval
            )
        except Exception as e:
            logger.error(f"Ingestion failed for {task_name}: {e}")
            continue

        if not downloaded_files:
            logger.warning(f"No files downloaded for {task_name}.")
            continue

        logger.info(f"{task_name}: {len(downloaded_files)} file(s) downloaded.")

        # ── Inter-scene co-registration gate ──────────────────────────────────
        # Must pass before any per-scene processing begins.  A pixel-count
        # variance > 1% means acquisitions are not on the same grid; delta
        # comparisons would produce false anomaly detections.
        if not quality_gate_coregistration(downloaded_files):
            logger.error(
                f"{task_name}: co-registration gate FAILED — "
                "valid-pixel-count variance exceeds 1%% tolerance. "
                "Mission series is geometrically inconsistent. Skipping task."
            )
            continue

        for raw_file in downloaded_files:
            # 2. Health Check
            if not check_alpha_health(raw_file, bbox_coords):
                logger.warning(f"Health check failed — skipping {os.path.basename(raw_file)}.")
                continue

            # 3. Beta (Analytics)
            try:
                processed_files = await asyncio.to_thread(
                    processor.process_scene,
                    raw_file
                )
            except Exception as e:
                logger.error(f"Analytics failed for {raw_file}: {e}")
                continue

            if not processed_files:
                logger.warning(f"Analytics produced no output for {os.path.basename(raw_file)}.")
                continue

            # 4. Gamma (Reporting)
            try:
                # Generate report for this specific tile's processed files
                # Filename: Evidence_Pack_{TileID}.pdf
                # We can extract Tile ID from filename
                # raw_file: 20231025_S2_T33UUE.tif
                parts = os.path.basename(raw_file).replace(".tif", "").split("_")
                tile_id = parts[2] if len(parts) > 2 else "unknown"
                date_str = parts[0]
                report_name = f"Evidence_Pack_{date_str}_{tile_id}.pdf"

                await asyncio.to_thread(
                    pdf_gen.generate_pdf,
                    filename=report_name,
                    specific_files=processed_files,
                    bbox=bbox
                )
            except Exception as e:
                logger.error(f"Gamma failed: {e}")

if __name__ == "__main__":
    asyncio.run(supervisor_main())
