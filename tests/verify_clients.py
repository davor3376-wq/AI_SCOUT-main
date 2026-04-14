import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import os
import shutil
import numpy as np
from sentinelhub import BBox, CRS

class TestIngestionClients(unittest.TestCase):

    def setUp(self):
        self.bbox = BBox(bbox=[16.2, 48.1, 16.5, 48.3], crs=CRS.WGS84)
        self.time_interval = ('2023-01-01', '2023-01-02')
        self.test_dir = "tests/temp_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

        self.data_raw = "data/raw"
        if not os.path.exists(self.data_raw):
            os.makedirs(self.data_raw)

    def tearDown(self):
        if os.path.exists(self.data_raw):
             # Ideally we should clean up specific files
             pass
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('app.ingestion.s1_client.SentinelHubAuth')
    @patch('app.ingestion.s1_client.SentinelHubRequest')
    @patch('app.ingestion.s1_client.SentinelHubCatalog')
    def test_s1_client(self, mock_catalog, mock_request, mock_auth):
        from app.ingestion.s1_client import S1Client

        # Mock Auth Config
        mock_auth_instance = mock_auth.return_value
        mock_auth_instance.config = MagicMock()

        # Mock Catalog Search
        mock_item = {
            "id": "test_s1_id",
            "properties": {
                "datetime": "2023-01-01T12:00:00Z",
                "sat:absolute_orbit": 12345,
                "sat:orbit_state": "descending"
            }
        }
        mock_catalog.return_value.search.return_value = [mock_item]

        # Mock Request Get Data
        mock_data = np.random.rand(10, 10, 2).astype(np.float32)
        mock_request.return_value.get_data.return_value = [mock_data]

        client = S1Client()
        output_files = client.download_data(self.bbox, self.time_interval)

        self.assertEqual(len(output_files), 1)
        self.assertTrue(os.path.exists(output_files[0]))
        self.assertTrue(output_files[0].endswith("20230101_S1_test_s1_id.tif"))

        # Check metadata exists
        meta_file = output_files[0].replace(".tif", "_provenance.json")
        self.assertTrue(os.path.exists(meta_file))

        # Clean up
        os.remove(output_files[0])
        os.remove(meta_file)

    @patch('app.ingestion.s2_client.SentinelHubAuth')
    @patch('app.ingestion.s2_client.SentinelHubRequest')
    @patch('app.ingestion.s2_client.SentinelHubCatalog')
    def test_s2_client(self, mock_catalog, mock_request, mock_auth):
        from app.ingestion.s2_client import S2Client

        # Mock Auth Config
        mock_auth_instance = mock_auth.return_value
        mock_auth_instance.config = MagicMock()

        # Mock Catalog Search
        mock_item = {
            "id": "test_s2_id",
            "properties": {
                "datetime": "2023-01-01T10:00:00Z",
                "sat:absolute_orbit": 67890,
                "eo:cloud_cover": 5.0
            }
        }
        mock_catalog.return_value.search.return_value = [mock_item]

        # Mock Request Get Data
        mock_data = np.random.randint(0, 10000, (10, 10, 3), dtype=np.uint16)
        mock_request.return_value.get_data.return_value = [mock_data]

        client = S2Client()
        output_files = client.download_data(self.bbox, self.time_interval)

        self.assertEqual(len(output_files), 1)
        self.assertTrue(os.path.exists(output_files[0]))
        self.assertTrue(output_files[0].endswith("20230101_S2_test_s2_id.tif"))

        # Check metadata exists
        meta_file = output_files[0].replace(".tif", "_provenance.json")
        self.assertTrue(os.path.exists(meta_file))

        # Clean up
        os.remove(output_files[0])
        os.remove(meta_file)

if __name__ == '__main__':
    unittest.main()
