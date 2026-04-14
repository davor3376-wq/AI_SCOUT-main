import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import numpy as np
import rasterio
from app.analytics.chronos import calculate_change
from app.ingestion.stac_catalog import StacCatalogManager
from app.ingestion.era5_climate import Era5Client # Ensure module is loaded
from datetime import datetime
from sentinelhub import BBox, CRS

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.t1 = os.path.join(self.test_dir, "20230101_NDVI.tif")
        self.t2 = os.path.join(self.test_dir, "20230102_NDVI.tif")

        # Create dummy TIFs
        profile = {
            'driver': 'GTiff', 'dtype': 'float32', 'width': 10, 'height': 10, 'count': 1,
            'crs': 'EPSG:4326', 'transform': rasterio.transform.from_bounds(0, 0, 1, 1, 10, 10)
        }
        with rasterio.open(self.t1, 'w', **profile) as dst:
            dst.write(np.ones((1, 10, 10), dtype='float32') * 0.5)

        with rasterio.open(self.t2, 'w', **profile) as dst:
            dst.write(np.ones((1, 10, 10), dtype='float32') * 0.2) # Decrease

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("data/stac_catalog"):
             shutil.rmtree("data/stac_catalog")

    def test_chronos_change(self):
        # Override OUTPUT_DIR for test safety, but logic in chronos is hardcoded.
        # We'll just rely on file existence check.
        diff_path = calculate_change(self.t2, self.t1) # t2 - t1 = 0.2 - 0.5 = -0.3
        self.assertIsNotNone(diff_path)
        self.assertTrue(os.path.exists(diff_path))

        with rasterio.open(diff_path) as src:
            data = src.read(1)
            self.assertAlmostEqual(data[0,0], -0.3, places=2)

        # Cleanup processed file
        if diff_path and os.path.exists(diff_path):
            os.remove(diff_path)

    def test_stac_manager(self):
        mgr = StacCatalogManager(catalog_root=os.path.join(self.test_dir, "stac"))
        bbox = BBox(bbox=[0, 0, 1, 1], crs=CRS.WGS84)
        mgr.add_item("item-1", self.t1, bbox, datetime.now())

        # Verify catalog exists
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "stac", "catalog.json")))

        # We can check via loading
        catalog = mgr.get_catalog()
        self.assertIsNotNone(catalog)
        items = list(catalog.get_items())
        self.assertEqual(len(items), 1)

    @patch("app.ingestion.era5_climate.SentinelHubRequest")
    def test_era5_client_mock(self, mock_req):
        client = Era5Client()
        # Just check it initializes without error
        self.assertIsNotNone(client)
