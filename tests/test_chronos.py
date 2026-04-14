import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import numpy as np
import rasterio
from app.analytics.chronos import AutoDifferencing, GifGenerator

class TestChronos(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/test_data_chronos"
        os.makedirs(self.test_dir, exist_ok=True)
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create dummy TIFFs
        self.tif1 = os.path.join(self.test_dir, "20230101_NDVI.tif")
        self.tif2 = os.path.join(self.test_dir, "20230201_NDVI.tif")

        self.create_dummy_tif(self.tif1, val=0.2)
        self.create_dummy_tif(self.tif2, val=0.5) # increased NDVI

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_dummy_tif(self, path, val):
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': 100,
            'height': 100,
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': rasterio.transform.from_origin(0, 0, 1, 1)
        }
        data = np.full((1, 100, 100), val, dtype=np.float32)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data)

    @patch('app.analytics.chronos.JobManager')
    def test_auto_differencing(self, MockJobManager):
        # Mock Job Manager
        jm = MockJobManager.return_value
        jm.list_jobs.return_value = [
            {
                "id": "JOB-OLD",
                "status": "COMPLETED",
                "sensor": "OPTICAL",
                "bbox": [10, 10, 20, 20],
                "results": {"processed_files": [self.tif1]},
                "created_at": "2023-01-01T00:00:00"
            }
        ]

        diff = AutoDifferencing(output_dir=self.output_dir)

        # Test find_previous_job
        prev_job = diff.find_previous_job("JOB-NEW", [10, 10, 20, 20], "OPTICAL")
        self.assertIsNotNone(prev_job)
        self.assertEqual(prev_job["id"], "JOB-OLD")

        # Test compute_difference
        diff_path = diff.compute_difference(self.tif2, self.tif1, "20230201")
        self.assertIsNotNone(diff_path)
        self.assertTrue(os.path.exists(diff_path))

        # Verify difference calculation (0.5 - 0.2 = 0.3)
        with rasterio.open(diff_path) as src:
            data = src.read(1)
            self.assertTrue(np.allclose(data, 0.3, atol=0.01))

    @patch('app.analytics.chronos.JobManager')
    def test_gif_generator(self, MockJobManager):
        jm = MockJobManager.return_value
        jm.list_jobs.return_value = [
            {
                "id": "JOB-OLD",
                "status": "COMPLETED",
                "bbox": [10, 10, 20, 20],
                "start_date": "2023-01-01",
                "results": {"processed_files": [self.tif1]}
            }
        ]

        gif_gen = GifGenerator(output_dir=self.output_dir)

        # Current files (simulating the job currently running)
        current_files = [self.tif2]

        gif_path = gif_gen.generate_timelapse("JOB-NEW", [10, 10, 20, 20], current_files)

        self.assertIsNotNone(gif_path)
        self.assertTrue(os.path.exists(gif_path))
        self.assertTrue(gif_path.endswith(".gif"))

if __name__ == '__main__':
    unittest.main()
