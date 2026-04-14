import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from datetime import datetime, timedelta
import asyncio
from app.analytics.alerting import calculate_alert_level
from app.analytics.masking import get_cloud_mask_from_qa60
from app.core.scheduler import MissionScheduler

class TestWatchdog(unittest.IsolatedAsyncioTestCase):
    def test_alert_level_high(self):
        # NDVI < 0.2 -> HIGH
        ndvi = np.array([0.1, 0.15, 0.0])
        cloud = 0.0
        level = calculate_alert_level(ndvi, cloud)
        self.assertEqual(level, "HIGH")

    def test_alert_level_low_cloudy(self):
        # Cloud > 50 -> LOW
        ndvi = np.array([0.1, 0.15, 0.0])
        cloud = 60.0
        level = calculate_alert_level(ndvi, cloud)
        self.assertEqual(level, "LOW")

    def test_alert_level_medium(self):
        # 0.2 < NDVI < 0.4 -> MEDIUM
        ndvi = np.array([0.3, 0.35])
        cloud = 10.0
        level = calculate_alert_level(ndvi, cloud)
        self.assertEqual(level, "MEDIUM")

    def test_qa60_masking(self):
        # Bit 10 set (1024)
        qa60 = np.array([1024, 0, 2048, 3072])
        # 1024 -> Opaque
        # 0 -> Clear
        # 2048 -> Cirrus
        # 3072 -> Both
        mask = get_cloud_mask_from_qa60(qa60)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])
        self.assertTrue(mask[3])

    @patch("app.core.scheduler.JobManager")
    @patch("app.core.scheduler.process_mission_task")
    @patch("app.core.scheduler.asyncio.create_task")
    async def test_scheduler(self, mock_create_task, mock_process, mock_jm_cls):
        mock_jm = mock_jm_cls.return_value

        # Setup Job
        job_id = "JOB-TEST"
        last_run = (datetime.utcnow() - timedelta(days=2)).isoformat()
        job = {
            "id": job_id,
            "recurrence": "DAILY",
            "last_run": last_run,
            "bbox": [0, 0, 1, 1],
            "sensor": "OPTICAL"
        }
        mock_jm.list_jobs.return_value = [job]
        mock_jm.create_job.return_value = "JOB-NEW"

        scheduler = MissionScheduler()
        await scheduler.check_jobs()

        # Should trigger
        mock_jm.create_job.assert_called()
        mock_jm.update_last_run.assert_called()
