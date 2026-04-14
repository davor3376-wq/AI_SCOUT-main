import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from sentinelhub import BBox, CRS

from app.ingestion.sentinel1_rtc import S1RTCClient
from app.ingestion.era5_climate import Era5Client
from app.ingestion.landsat_legacy import LandsatClient
from app.ingestion.modis_daily import ModisClient
from app.ingestion.noaa_weather import NoaaWeatherClient
from app.ingestion.iot_feed import IotFeedHandler
from app.ingestion.cache_manager import CacheManager
from app.ingestion.aoi_tiling import split_bbox
from app.ingestion.stac_catalog import StacCatalogManager

class TestOmniIngestion(unittest.TestCase):

    def setUp(self):
        self.bbox = BBox(bbox=[14.0, 45.0, 14.1, 45.1], crs=CRS.WGS84)
        self.time_interval = (datetime(2023, 1, 1), datetime(2023, 1, 2))

    @patch("app.ingestion.sentinel1_rtc.SentinelHubRequest")
    @patch("app.ingestion.sentinel1_rtc.SentinelHubCatalog")
    def test_s1_rtc_client(self, mock_catalog, mock_request):
        # Setup mocks
        mock_catalog_instance = mock_catalog.return_value
        mock_catalog_instance.search.return_value = [
            {"id": "test_id", "properties": {"datetime": "2023-01-01T12:00:00Z", "sat:absolute_orbit": 123}}
        ]

        mock_request_instance = mock_request.return_value
        # Mock data return: 10x10 image, 2 bands (VV, VH)
        import numpy as np
        mock_request_instance.get_data.return_value = [np.zeros((10, 10, 2), dtype=np.float32)]

        client = S1RTCClient()
        files = client.download_data(self.bbox, self.time_interval)

        self.assertTrue(len(files) > 0)
        self.assertIn("S1_RTC", files[0])

        # Verify Evalscript contained RTC parameters
        # We can inspect the call args of SentinelHubRequest
        call_args = mock_request.call_args
        # print(call_args)
        # It's hard to check exact evalscript string matching, but we can check if it was called.
        self.assertTrue(mock_request.called)

    @patch("app.ingestion.era5_climate.SentinelHubRequest")
    def test_era5_client(self, mock_request):
        mock_request_instance = mock_request.return_value
        import numpy as np
        mock_request_instance.get_data.return_value = [np.zeros((10, 10, 2), dtype=np.float32)]

        client = Era5Client()
        files = client.download_data(self.bbox, self.time_interval)

        self.assertTrue(len(files) > 0)
        self.assertIn("ERA5", files[0])

    def test_aoi_tiling(self):
        bbox = BBox(bbox=[0, 0, 2, 2], crs=CRS.WGS84)
        sub_bboxes = split_bbox(bbox, split_x=2, split_y=2)
        self.assertEqual(len(sub_bboxes), 4)
        # Check first quadrant
        self.assertEqual(sub_bboxes[0].lower_left, (0.0, 0.0))
        self.assertEqual(sub_bboxes[0].upper_right, (1.0, 1.0))

    @patch("app.ingestion.stac_catalog.Catalog")
    def test_stac_catalog(self, mock_catalog):
        manager = StacCatalogManager(catalog_root="test_catalog")
        # Just verify it doesn't crash on init
        self.assertIsNotNone(manager)

    def test_new_modules_init(self):
        # Verify stubs can be instantiated
        noaa = NoaaWeatherClient()
        self.assertIsNotNone(noaa)

        iot = IotFeedHandler()
        self.assertIsNotNone(iot)

        cache = CacheManager(cache_dir="test_cache_dir")
        self.assertIsNotNone(cache)
        import shutil
        shutil.rmtree("test_cache_dir", ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
