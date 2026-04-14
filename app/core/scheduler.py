import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List

from sentinelhub import BBox, CRS
from app.api.job_manager import JobManager
from app.core.mission_executor import process_mission_task

logger = logging.getLogger("Scheduler")

class MissionScheduler:
    def __init__(self):
        self.jm = JobManager()
        self.running = False

    async def start(self):
        self.running = True
        logger.info("MissionScheduler started.")
        while self.running:
            try:
                await self.check_jobs()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            # Poll every 5 minutes. Due-time logic is in check_jobs (24h window),
            # so frequent polling is cheap and ensures daily jobs fire promptly.
            await asyncio.sleep(300)

    def stop(self):
        self.running = False

    async def check_jobs(self):
        jobs = self.jm.list_recurring_jobs()
        for job in jobs:
            if job.get("recurrence") == "DAILY":
                await self.process_recurring_job(job)

    async def process_recurring_job(self, job):
        job_id = job.get("id")
        last_run_str = job.get("last_run")

        should_run = False
        if not last_run_str:
            # If never run, run now? Or wait for next scheduled time?
            # Let's run now to bootstrap.
            should_run = True
        else:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                # Simple logic: if > 24 hours since last run
                if datetime.now(timezone.utc) - last_run > timedelta(hours=24):
                    should_run = True
            except ValueError:
                should_run = True

        if should_run:
            logger.info(f"Triggering daily run for job {job_id}")

            # Create a NEW job for this execution
            # Copy relevant metadata
            bbox_list = job.get("bbox")
            sensor = job.get("sensor")

            if not bbox_list or not sensor:
                logger.warning(
                    f"Skipping recurring job {job_id}: missing bbox or sensor."
                )
                return

            # Define new time interval: Last 24 hours
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=1)

            # Create new job metadata
            new_metadata = {
                "sensor": sensor,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "bbox": bbox_list,
                "parent_job_id": job_id,
                "recurrence": None, # Execution instance is not recurring itself
                "tag": "DAILY_RUN",
                "launched_by": job.get("launched_by", "scheduler"),
            }

            new_job_id = self.jm.create_job(new_metadata)

            # Reconstruct BBox object
            # Assuming WGS84 as standard for stored bbox lists
            bbox = BBox(bbox=bbox_list, crs=CRS.WGS84)

            # Run task
            # We use asyncio.to_thread to run blocking code in a separate thread
            # This ensures the scheduler loop isn't blocked
            asyncio.create_task(
                asyncio.to_thread(
                    process_mission_task,
                    new_job_id,
                    bbox,
                    (start_date, end_date),
                    sensor
                )
            )

            # Update parent job last_run
            self.jm.update_last_run(job_id, datetime.now(timezone.utc))
