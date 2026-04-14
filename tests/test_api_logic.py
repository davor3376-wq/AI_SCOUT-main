import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
from fastapi import BackgroundTasks
from app.api.main import app, launch_mission, MissionRequest

class TestMissionLogic(unittest.IsolatedAsyncioTestCase):

    @patch('app.api.main.SentinelHubAuth')
    @patch('app.api.main.SentinelHubCatalog')
    async def test_optical_good_cloud_cover(self, MockCatalog, MockAuth):
        # Setup Mock
        mock_catalog_instance = MockCatalog.return_value

        # Mock Search Results for Optical
        optical_item = {
            "id": "S2_TILE_1",
            "datetime": "2023-10-01T12:00:00Z",
            "properties": {"eo:cloud_cover": 10},
            "assets": {"thumbnail": {"href": "http://preview/s2"}}
        }
        mock_catalog_instance.search.return_value = iter([optical_item])

        # Request
        request = MissionRequest(
            geometry={"type": "Polygon", "coordinates": [[[16.2, 48.1], [16.5, 48.1], [16.5, 48.3], [16.2, 48.3], [16.2, 48.1]]]},
            start_date="2023-09-01",
            end_date="2023-10-01",
            sensor="OPTICAL"
        )

        # Mock BackgroundTasks
        mock_bg_tasks = MagicMock(spec=BackgroundTasks)

        response = await launch_mission(request, mock_bg_tasks)

        self.assertEqual(response["tile_id"], "S2_TILE_1")
        self.assertIn("job_id", response) # New assertion
        self.assertIsNone(response.get("tag"))
        self.assertEqual(response["preview_url"], "http://preview/s2")

    @patch('app.api.main.SentinelHubAuth')
    @patch('app.api.main.SentinelHubCatalog')
    async def test_optical_cloud_fallback(self, MockCatalog, MockAuth):
        # Setup Mock
        mock_catalog_instance = MockCatalog.return_value

        # Mock Search Results:
        # First call (Optical) returns cloudy item
        # Second call (Radar) returns radar item

        optical_item = {
            "id": "S2_CLOUDY",
            "datetime": "2023-10-01T12:00:00Z",
            "properties": {"eo:cloud_cover": 90},
            "assets": {"thumbnail": {"href": "http://preview/s2_cloudy"}}
        }

        radar_item = {
            "id": "S1_RADAR",
            "datetime": "2023-10-01T12:00:00Z",
            "properties": {},
            "assets": {"thumbnail": {"href": "http://preview/s1"}}
        }

        # Side effect for search: depending on collection arg or call count
        # Since we can't easily inspect arguments in side_effect list for iterators,
        # we'll assume the code calls search twice.
        # But wait, create a generator for each call.

        def search_side_effect(*args, **kwargs):
            # Check collection to distinguish
            # DataCollection objects are passed.
            # We can check if "SENTINEL1" is in string repr or look at kwargs
            collection = kwargs.get('collection')
            # Assuming collection object.
            # SentinelHubCatalog.search takes collection.

            # Simplified: The code calls Optical search first.
            if "SENTINEL2" in str(collection) or "CDSE_S2_L2A" in str(collection):
                 return iter([optical_item])
            else:
                 return iter([radar_item])

        mock_catalog_instance.search.side_effect = search_side_effect

        # Request
        request = MissionRequest(
            geometry={"type": "Polygon", "coordinates": [[[16.2, 48.1], [16.5, 48.1], [16.5, 48.3], [16.2, 48.3], [16.2, 48.1]]]},
            start_date="2023-09-01",
            end_date="2023-10-01",
            sensor="OPTICAL"
        )

        # Mock BackgroundTasks
        mock_bg_tasks = MagicMock(spec=BackgroundTasks)

        response = await launch_mission(request, mock_bg_tasks)

        self.assertEqual(response["tile_id"], "S1_RADAR")
        self.assertEqual(response["tag"], "CLOUD_PIERCED")
        self.assertEqual(response["preview_url"], "http://preview/s1")

    @patch('app.api.main.SentinelHubAuth')
    @patch('app.api.main.SentinelHubCatalog')
    async def test_point_geometry(self, MockCatalog, MockAuth):
        # Setup Mock
        mock_catalog_instance = MockCatalog.return_value

        item = {
            "id": "S1_POINT",
            "datetime": "2023-10-01T12:00:00Z",
            "properties": {},
            "assets": {"thumbnail": {"href": "http://preview/s1_point"}},
            "bbox": [16.2, 48.1, 16.5, 48.3]
        }
        mock_catalog_instance.search.return_value = iter([item])

        # Request with Point
        request = MissionRequest(
            geometry={"type": "Point", "coordinates": [16.35, 48.2]},
            start_date="2023-09-01",
            end_date="2023-10-01",
            sensor="RADAR"
        )

        # Mock BackgroundTasks
        mock_bg_tasks = MagicMock(spec=BackgroundTasks)

        response = await launch_mission(request, mock_bg_tasks)

        self.assertEqual(response["tile_id"], "S1_POINT")
        self.assertEqual(response.get("bbox"), [16.2, 48.1, 16.5, 48.3])

if __name__ == '__main__':
    unittest.main()
