"""
Tests for Analytics Layer (Roles 6 & 8).
"""
import unittest
import numpy as np
from app.analytics.masking import get_cloud_mask, calculate_cloud_coverage
from app.analytics.indices import calculate_ndvi, calculate_ndwi

class TestAnalytics(unittest.TestCase):

    def test_cloud_masking(self):
        # SCL values: 3 (Shadow), 4 (Veg), 8 (Cloud Med), 9 (Cloud High), 11 (Snow)
        scl = np.array([
            [3, 4, 8],
            [9, 11, 5]
        ])

        expected_mask = np.array([
            [True, False, True],
            [True, True, False]
        ])

        mask = get_cloud_mask(scl)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_cloud_coverage(self):
        mask = np.array([True, False, True, False]) # 50%
        coverage = calculate_cloud_coverage(mask)
        self.assertEqual(coverage, 50.0)

        mask_empty = np.array([])
        self.assertEqual(calculate_cloud_coverage(mask_empty), 0.0)

    def test_ndvi(self):
        # NDVI = (NIR - Red) / (NIR + Red)
        # Case 1: Vegetation (Low Red, High NIR) -> High Positive
        red = np.array([10.0])
        nir = np.array([100.0])
        # (100 - 10) / (100 + 10) = 90 / 110 = 0.8181...

        ndvi = calculate_ndvi(red, nir)
        self.assertAlmostEqual(ndvi[0], 0.8181818, places=5)

        # Case 2: Water (Red > NIR usually? Or low both. )
        # Let's just test math.
        # Case 3: Division by Zero
        red_zero = np.array([0])
        nir_zero = np.array([0])
        ndvi_zero = calculate_ndvi(red_zero, nir_zero)
        self.assertTrue(np.isnan(ndvi_zero[0]))

    def test_ndwi(self):
        # NDWI = (Green - NIR) / (Green + NIR)
        green = np.array([100.0])
        nir = np.array([10.0])
        # (100 - 10) / (100 + 10) = 90 / 110 = 0.8181...

        ndwi = calculate_ndwi(green, nir)
        self.assertAlmostEqual(ndwi[0], 0.8181818, places=5)

        # Zero check
        green_zero = np.array([0])
        nir_zero = np.array([0])
        ndwi_zero = calculate_ndwi(green_zero, nir_zero)
        self.assertTrue(np.isnan(ndwi_zero[0]))

import os
import shutil
import tempfile
import rasterio
from rasterio.transform import from_origin
from app.analytics.processor import run, INPUT_DIR, OUTPUT_DIR

class TestAnalyticsIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "data", "raw")
        self.processed_dir = os.path.join(self.test_dir, "data", "processed")
        os.makedirs(self.raw_dir)

        # Patch the processor's directories
        self.original_input = 'app.analytics.processor.INPUT_DIR'
        self.original_output = 'app.analytics.processor.OUTPUT_DIR'

        # We need to patch the global variables in the module
        import app.analytics.processor as processor
        processor.INPUT_DIR = self.raw_dir
        processor.OUTPUT_DIR = self.processed_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Restore globals (though this might be flaky if run in parallel,
        # but standard unittest is sequential)
        import app.analytics.processor as processor
        processor.INPUT_DIR = "data/raw"
        processor.OUTPUT_DIR = "data/processed"

    def test_end_to_end_processing(self):
        # Create dummy S2 data
        width, height = 10, 10
        red = np.ones((height, width), dtype=np.uint16) * 1000
        nir = np.ones((height, width), dtype=np.uint16) * 5000
        scl = np.ones((height, width), dtype=np.uint16) * 4 # Vegetation

        # Mask one pixel
        scl[0, 0] = 9

        transform = from_origin(16.3, 48.2, 0.0001, 0.0001)
        filename = "20230101_S2_TESTTILE.tif"
        filepath = os.path.join(self.raw_dir, filename)

        with rasterio.open(
            filepath,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=rasterio.uint16,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(red, 1)
            dst.write(nir, 2)
            dst.write(scl, 3)

        # Run processor
        import app.analytics.processor as processor
        processor.run()

        expected_output = os.path.join(self.processed_dir, "20230101_NDVI_analysis.tif")
        self.assertTrue(os.path.exists(expected_output))

        with rasterio.open(expected_output) as src:
            ndvi = src.read(1)
            # Check valid pixel
            self.assertAlmostEqual(ndvi[0, 1], 0.6666667, places=5)
            # Check masked pixel
            self.assertTrue(np.isnan(ndvi[0, 0]))

if __name__ == '__main__':
    unittest.main()
