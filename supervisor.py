import asyncio
import logging
import os
import random
import rasterio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
from sentinelhub import BBox, CRS

import main  # Import the main mission script
from app.analytics.alerting import check_coregistration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SUPERVISOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supervisor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MissionSupervisor:
    def __init__(self):
        self.failed_missions = []
        self.completed_missions = 0
        self.total_missions = 0

    def generate_grid(self, center_lat: float, center_lon: float, count: int = 100) -> List[BBox]:
        """
        Generates a grid of BBoxes around a center point.
        """
        grid = []
        # Calculate grid dimension (sqrt of count)
        dim = int(np.ceil(np.sqrt(count)))
        step = 0.01 # Approx 1km

        start_lon = center_lon - (dim / 2) * step
        start_lat = center_lat - (dim / 2) * step

        for i in range(dim):
            for j in range(dim):
                if len(grid) >= count:
                    break

                lon_min = start_lon + (i * step)
                lat_min = start_lat + (j * step)
                lon_max = lon_min + step
                lat_max = lat_min + step

                bbox = BBox(bbox=[lon_min, lat_min, lon_max, lat_max], crs=CRS.WGS84)
                grid.append(bbox)

        return grid

    def quality_gate(self, file_paths: List[str]) -> bool:
        """
        Inspects NDVI TIFs. Returns True if pass, False if fail.

        Fail conditions:
          - File missing or unreadable.
          - All-NaN or all-zero valid pixels (black image / no data).
          - Valid-pixel counts vary by more than 1 % across acquisitions
            (geometric co-registration failure).
        """
        if not file_paths:
            logger.warning("Quality Gate: No files provided.")
            return False

        valid_pixel_counts: List[int] = []

        for fp in file_paths:
            if not os.path.exists(fp):
                logger.error(f"Quality Gate: File not found {fp}")
                return False

            if "NDVI" not in fp:
                continue

            try:
                with rasterio.open(fp) as src:
                    data = src.read(1)

                    # Check for all NaN
                    if np.all(np.isnan(data)):
                        logger.warning(f"Quality Gate Failed: {fp} is all NaN.")
                        return False

                    valid_data = data[~np.isnan(data)]
                    if valid_data.size == 0:
                        logger.warning(f"Quality Gate Failed: {fp} has no valid pixels.")
                        return False

                    if np.all(valid_data == 0):
                        logger.warning(f"Quality Gate Failed: {fp} is all zeros.")
                        return False

                    valid_pixel_counts.append(int(valid_data.size))
                    logger.info(f"Quality Gate Passed: {fp} ({valid_data.size} valid pixels)")

            except Exception as e:
                logger.error(f"Quality Gate Error reading {fp}: {e}")
                return False

        # Co-registration check: pixel counts must be stable across all acquisitions.
        if not check_coregistration(valid_pixel_counts):
            logger.error(
                "Quality Gate Failed: geometric co-registration check failed. "
                "Valid pixel counts across acquisitions: %s. "
                "Delta analysis results are unreliable.",
                valid_pixel_counts,
            )
            return False

        return True

    def analyze_error(self, error: Exception, reason: str) -> str:
        """
        Analyzes the error to determine the correction strategy.
        """
        if reason == "Quality Gate Failure":
            return "EXPAND_TIME_WINDOW"
        if error:
            error_str = str(error).lower()
            if "timeout" in error_str:
                return "REDUCE_CONCURRENCY" # Not implemented in this loop, but logical
            if "auth" in error_str or "client_id" in error_str:
                return "CHECK_AUTH"

        return "GENERIC_RETRY"

    def apply_code_rewrite_simulation(self, strategy: str) -> Tuple[dict, str]:
        """
        Simulates the 'Code Rewrite' by returning modified parameters (runtime patching).
        In a full agentic system, this would use an LLM to generate a patch.

        Returns:
            Tuple[dict, str]: (New Parameters, Log Message)
        """
        params = {}
        log_msg = ""

        if strategy == "EXPAND_TIME_WINDOW":
            # "Rewriting" the default time interval
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=60)
            new_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            params["time_interval"] = new_interval
            log_msg = "Rewriting mission parameters -> Extending time search window to 60 days."

        elif strategy == "CHECK_AUTH":
            log_msg = "Critical Error: Authentication missing. 'Rewrite' would involve checking environment secrets."
            # Cannot fix auth via params usually, but could try a backup key.

        else:
            log_msg = "Applying generic retry patch."

        return params, log_msg

    async def self_correct_and_retry(self, bbox: BBox, error: Exception = None, reason: str = ""):
        """
        Attempts to fix the issue by 'rewriting' parameters or applying patches.
        """
        logger.info(f"Initiating Self-Correction Protocol. Reason: {reason or error}")

        strategy = self.analyze_error(error, reason)
        new_params, log_msg = self.apply_code_rewrite_simulation(strategy)

        logger.info(f"Correction Strategy: {strategy}")
        logger.info(f"Action: {log_msg}")

        # Placeholder for Actual Code Rewriting via LLM
        # ---------------------------------------------------------
        # if self.enable_llm_rewriting:
        #     source_code = read_file("main.py")
        #     prompt = f"Fix this error: {error}\nSource:\n{source_code}"
        #     new_code = llm.generate(prompt)
        #     write_file("main.py", new_code)
        #     reload_module(main)
        # ---------------------------------------------------------

        if strategy == "CHECK_AUTH":
            logger.error("Cannot auto-correct missing credentials without access to secure vault.")
            return False

        try:
            logger.info("Retrying mission with patched parameters...")
            # Unpack new params into the call
            files = await main.main(bbox=bbox, **new_params)

            if self.quality_gate(files):
                logger.info("Self-Correction Successful: Mission Passed.")
                return True
            else:
                logger.error("Self-Correction Failed: Quality Gate still failing.")
                return False

        except Exception as e:
            logger.error(f"Self-Correction Retry Failed: {e}")
            return False

    async def execute_mission(self, bbox: BBox, mission_id: int):
        """
        Runs a single mission for a bbox.
        """
        logger.info(f"=== Starting Mission {mission_id} ===")

        try:
            # Initial Run
            files = await main.main(bbox=bbox)

            # Quality Gate
            if self.quality_gate(files):
                self.completed_missions += 1
                logger.info(f"Mission {mission_id} Success.")
            else:
                logger.warning(f"Mission {mission_id} failed Quality Gate.")
                success = await self.self_correct_and_retry(bbox, reason="Quality Gate Failure")
                if success:
                    self.completed_missions += 1
                else:
                    self.failed_missions.append(mission_id)

        except Exception as e:
            logger.error(f"Mission {mission_id} crashed: {e}")
            # Try to recover
            success = await self.self_correct_and_retry(bbox, error=e)
            if success:
                self.completed_missions += 1
            else:
                self.failed_missions.append(mission_id)

    async def run_grid_mission(self):
        """
        Orchestrates the entire grid mission.
        """
        logger.info("Initializing Grid Mission...")

        # Vienna Center
        center_lat, center_lon = 48.2082, 16.3738

        # For demonstration/testing, let's limit to 5 instead of 100 if we want to finish quickly?
        # The prompt says "list of 100 coordinates". I should probably generate 100 but maybe
        # run them in parallel batches or just a few for the proof of concept if 100 takes too long.
        # I'll generate 100 but I'll add a flag to run only a subset for dev.
        # Let's generate 4 for testing purposes now, but the code supports 100.
        # I'll default to 100 as requested, but maybe add a LIMIT constant.

        grid = self.generate_grid(center_lat, center_lon, count=100)
        self.total_missions = len(grid)

        # Concurrency Control
        # We don't want to spawn 100 tasks at once (rate limits).
        # SentinelHub might rate limit us.
        semaphore = asyncio.Semaphore(5) # Run 5 at a time

        async def sem_task(task_id, bbox):
            async with semaphore:
                await self.execute_mission(bbox, task_id)

        # For safety in this environment, we use a semaphore.
        logger.info(f"Generated {len(grid)} grid points. Queuing for execution...")

        tasks = [sem_task(i, bbox) for i, bbox in enumerate(grid)]

        await asyncio.gather(*tasks)

        logger.info("Grid Mission Finished.")
        logger.info(f"Completed: {self.completed_missions}/{len(tasks)}")
        logger.info(f"Failed: {len(self.failed_missions)}")

if __name__ == "__main__":
    supervisor = MissionSupervisor()
    asyncio.run(supervisor.run_grid_mission())
