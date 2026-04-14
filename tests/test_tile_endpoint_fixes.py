import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
from app.api.main import app
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

client = TestClient(app)

class TestTileEndpointFixes(unittest.TestCase):

    @patch('app.api.main.os.path.exists')
    @patch('app.api.main.rasterio.open')
    def test_path_sanitization(self, mock_open, mock_exists):
        # We simulate that 'results/secret.tif' exists.
        def exists_side_effect(path):
            if path == os.path.join("results", "secret.tif"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        # Mock rasterio
        mock_src = MagicMock()
        mock_src.count = 1
        mock_src.crs = 'EPSG:4326'
        mock_src.transform = from_origin(0, 0, 1, 1)
        mock_open.return_value.__enter__.return_value = mock_src

        # Patch reproject to avoid logic
        with patch('app.api.main.reproject'):
             response = client.get("/tiles/0/0/0?file=../../secret.tif")

        # rasterio.open should be called with the sanitized path "results/secret.tif"
        mock_open.assert_called_with(os.path.join("results", "secret.tif"))

    @patch('app.api.main.rasterio.band')
    @patch('app.api.main.os.path.exists')
    @patch('app.api.main.rasterio.open')
    def test_rgb_float_normalization(self, mock_open, mock_exists, mock_band):
        # Test that float data 0.0-1.0 is scaled to 0-255

        mock_exists.return_value = True
        mock_band.return_value = "mock_band"

        def reproject_side_effect(source, destination, **kwargs):
            # Fill destination with 0.5 (which is float)
            destination.fill(0.5)

        with patch('app.api.main.reproject', side_effect=reproject_side_effect):
             # Mock src as RGB
            mock_src = MagicMock()
            mock_src.count = 3 # RGB
            mock_src.crs = 'EPSG:4326'
            mock_src.transform = from_origin(0, 0, 1, 1)
            mock_open.return_value.__enter__.return_value = mock_src

            response = client.get("/tiles/0/0/0?file=test.tif")

            assert response.status_code == 200, f"Status code {response.status_code}"

            # Verify the image content
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(response.content))
            data = np.array(img)

            # Check center pixel. 0.5 * 255 = 127.5 -> 127
            pixel = data[128, 128]
            # Expecting ~127 for RGB channels
            self.assertTrue(126 <= pixel[0] <= 129, f"Pixel value {pixel[0]} not in range [126, 129]")
            self.assertTrue(126 <= pixel[1] <= 129)
            self.assertTrue(126 <= pixel[2] <= 129)
            self.assertEqual(pixel[3], 255) # Opaque
