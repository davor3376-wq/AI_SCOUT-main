"""
Integration tests for the Analyze → Report pipeline.

Strategy: create real (tiny, synthetic) GeoTIFFs that mimic S2 output,
then run the real processor and reporting code against them.
No Sentinel Hub credentials required — ingestion is bypassed entirely.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from sentinelhub import BBox

# Patch data dirs to a temp location BEFORE importing project modules
_TMP = tempfile.mkdtemp(prefix="scout_integration_")
os.environ.setdefault("SCOUT_DB_PATH", os.path.join(_TMP, "test_jobs.db"))


def _make_synthetic_s2(path: str, rows: int = 64, cols: int = 64) -> str:
    """
    Write a 5-band (B03 Green, B04 Red, B08 NIR, SCL, QA60) UINT16 GeoTIFF
    that looks like a typical S2 L2A tile.
    Band values simulate moderate vegetation (NDVI ~0.4).
    """
    rng = np.random.default_rng(42)
    green = (rng.integers(500, 1500, (rows, cols))).astype(np.uint16)
    red   = (rng.integers(400,  900, (rows, cols))).astype(np.uint16)
    nir   = (rng.integers(2000, 4000, (rows, cols))).astype(np.uint16)
    scl   = np.full((rows, cols), 4, dtype=np.uint16)   # 4 = vegetation
    qa60  = np.zeros((rows, cols), dtype=np.uint16)      # no clouds

    transform = from_bounds(16.2, 48.1, 16.5, 48.4, cols, rows)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=rows, width=cols, count=5,
        dtype=np.uint16, crs=crs, transform=transform,
    ) as dst:
        dst.write(green, 1)
        dst.write(red,   2)
        dst.write(nir,   3)
        dst.write(scl,   4)
        dst.write(qa60,  5)
    return path


class TestAnalyticsPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_dir  = os.path.join(_TMP, "data", "raw")
        self.proc_dir = os.path.join(_TMP, "data", "processed")
        self.res_dir  = os.path.join(_TMP, "results")
        for d in (self.raw_dir, self.proc_dir, self.res_dir):
            os.makedirs(d, exist_ok=True)

        # Create a synthetic S2 tile (name format expected by processor)
        self.tif_path = os.path.join(self.raw_dir, "20240101_S2_SYNTHETIC.tif")
        _make_synthetic_s2(self.tif_path)

    def tearDown(self):
        # Remove only files created in setUp, leave _TMP for inspection on failure
        for d in (self.raw_dir, self.proc_dir, self.res_dir):
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------
    # Patch OUTPUT_DIR before importing so processor writes to _TMP
    # ------------------------------------------------------------------
    def _run_processor(self):
        import app.analytics.processor as proc
        orig_out = proc.OUTPUT_DIR
        proc.OUTPUT_DIR = self.proc_dir
        try:
            return proc.run(input_files=[self.tif_path])
        finally:
            proc.OUTPUT_DIR = orig_out

    def test_processor_generates_ndvi(self):
        outputs = self._run_processor()
        ndvi_files = [f for f in outputs if "NDVI" in f]
        self.assertTrue(
            len(ndvi_files) >= 1,
            f"Expected at least one NDVI output, got: {outputs}"
        )
        # File should exist and be non-empty
        self.assertTrue(os.path.exists(ndvi_files[0]))
        self.assertGreater(os.path.getsize(ndvi_files[0]), 0)

    def test_processor_generates_ndwi(self):
        outputs = self._run_processor()
        ndwi_files = [f for f in outputs if "NDWI" in f]
        self.assertTrue(
            len(ndwi_files) >= 1,
            f"Expected at least one NDWI output, got: {outputs}"
        )

    def test_ndvi_values_in_valid_range(self):
        outputs = self._run_processor()
        ndvi_path = next((f for f in outputs if "NDVI" in f), None)
        self.assertIsNotNone(ndvi_path)
        with rasterio.open(ndvi_path) as src:
            data = src.read(1)
        valid = data[~np.isnan(data)]
        self.assertTrue(np.all(valid >= -1.0), "NDVI below -1")
        self.assertTrue(np.all(valid <= 1.0),  "NDVI above +1")

    def test_requested_indices_filtering(self):
        """Only NDVI should be produced when requested_indices=["NDVI"]."""
        import app.analytics.processor as proc
        orig_out = proc.OUTPUT_DIR
        proc.OUTPUT_DIR = self.proc_dir
        try:
            outputs = proc.run(
                input_files=[self.tif_path],
                requested_indices=["NDVI"],
            )
        finally:
            proc.OUTPUT_DIR = orig_out

        ndwi = [f for f in outputs if "NDWI" in f]
        self.assertEqual(ndwi, [], "NDWI should not be produced when not requested")


class TestReportingPipeline(unittest.TestCase):
    """Tests KML and PDF generation given pre-processed NDVI TIFs."""

    def setUp(self):
        self.proc_dir = os.path.join(_TMP, "data", "processed2")
        self.res_dir  = os.path.join(_TMP, "results2")
        os.makedirs(self.proc_dir, exist_ok=True)
        os.makedirs(self.res_dir,  exist_ok=True)

        # Create a tiny NDVI TIF directly
        self.ndvi_path = os.path.join(self.proc_dir, "20240101_NDVI_analysis.tif")
        rng = np.random.default_rng(7)
        ndvi_data = rng.uniform(-0.2, 0.8, (32, 32)).astype(np.float32)
        transform = from_bounds(16.2, 48.1, 16.5, 48.4, 32, 32)
        with rasterio.open(
            self.ndvi_path, "w", driver="GTiff",
            height=32, width=32, count=1,
            dtype=np.float32, crs=CRS.from_epsg(4326), transform=transform,
        ) as dst:
            dst.write(ndvi_data, 1)

        self.bbox = BBox(bbox=[16.2, 48.1, 16.5, 48.4], crs=None)

    def tearDown(self):
        shutil.rmtree(self.proc_dir, ignore_errors=True)
        shutil.rmtree(self.res_dir,  ignore_errors=True)

    def test_kml_generation(self):
        from app.reporting.kml_gen import generate_kml
        kml_path = generate_kml("TEST-JOB", self.bbox, output_dir=self.res_dir)
        self.assertIsNotNone(kml_path)
        self.assertTrue(os.path.exists(kml_path), f"KML not found at {kml_path}")
        self.assertGreater(os.path.getsize(kml_path), 0)

    def test_pdf_generation(self):
        from app.reporting.pdf_gen import PDFReportGenerator
        gen = PDFReportGenerator()
        report_name = "test_report.pdf"
        gen.generate_pdf(
            filename=report_name,
            specific_files=[self.ndvi_path],
            bbox=self.bbox,
        )
        pdf_path = os.path.join("results", report_name)
        # PDF gen writes to ./results by default; accept either location
        alt_path = os.path.join(self.res_dir, report_name)
        found = os.path.exists(pdf_path) or os.path.exists(alt_path)
        self.assertTrue(found, "PDF report was not created")


if __name__ == "__main__":
    unittest.main()
