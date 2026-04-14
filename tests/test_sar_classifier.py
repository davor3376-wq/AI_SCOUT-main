"""
Tests for app/analytics/sar_classifier.py

Covers:
  - Backward-compatible base classes (WATER, BARE_SOIL, VEGETATION)
  - New 3-tier dual-pol CR discriminator (URBAN / WET_FOREST / VEGETATION)
  - apply_persistence_filter()  — temporal consistency
  - apply_forest_mask()         — S2 NDVI spatial masking
  - classify_temporal_stack()   — integration pipeline
  - generate_backscatter_trend_report() — ASCII table renderer
"""
import unittest
import numpy as np

from app.analytics.sar_classifier import (
    WATER,
    BARE_SOIL,
    VEGETATION,
    URBAN,
    ANOMALOUS_VEG_MOISTURE,
    WET_FOREST,
    NO_DATA,
    CLASS_LABELS,
    CLASS_PLAIN_LANGUAGE,
    _CR_DOUBLE_BOUNCE_MAX,
    _CR_VEG_MIN,
    _VV_VEG_MAX,
    _VV_URBAN_MIN_NOVH,
    classify_sar,
    classification_stats,
    apply_persistence_filter,
    apply_forest_mask,
    classify_temporal_stack,
    generate_backscatter_trend_report,
    explain_pixel,
    suggest_cr_thresholds,
    generate_validation_protocol,
    compute_accuracy_metrics,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _scalar_vv(vv: float, vh: float = None) -> tuple:
    """Return 1-pixel (1,1) arrays for vv and optional vh."""
    arr_vv = np.array([[vv]], dtype=np.float32)
    arr_vh = np.array([[vh]], dtype=np.float32) if vh is not None else None
    return arr_vv, arr_vh


def _classify_single(vv: float, vh: float = None) -> int:
    arr_vv, arr_vh = _scalar_vv(vv, vh)
    return int(classify_sar(arr_vv, arr_vh)[0, 0])


# ── Base class backward-compatibility ──────────────────────────────────────────

class TestClassifySarBaseClasses(unittest.TestCase):

    def test_water(self):
        self.assertEqual(_classify_single(0.005), WATER)

    def test_water_with_low_vh(self):
        self.assertEqual(_classify_single(0.005, vh=0.001), WATER)

    def test_bare_soil(self):
        self.assertEqual(_classify_single(0.03), BARE_SOIL)

    def test_vegetation_intermediate_vv(self):
        self.assertEqual(_classify_single(0.08), VEGETATION)

    def test_no_data_negative(self):
        result = classify_sar(np.array([[-1.0]], dtype=np.float32))
        self.assertEqual(int(result[0, 0]), NO_DATA)

    def test_no_data_nan(self):
        result = classify_sar(np.array([[np.nan]], dtype=np.float32))
        self.assertEqual(int(result[0, 0]), NO_DATA)


# ── 3-tier Dual-Polarisation Discriminator ─────────────────────────────────────

class TestClassifySarDualPol(unittest.TestCase):
    """
    Verify the 3-tier CR decision for high-VV pixels.
    Thresholds: _CR_DOUBLE_BOUNCE_MAX=0.15, _CR_VEG_MIN=0.30
    """

    def _make_scene(self, vv_lin: float, cr: float):
        """Return (vv, vh) arrays for a given VV (linear) and CR = VH/VV."""
        vh_lin = vv_lin * cr
        return _classify_single(vv_lin, vh=vh_lin)

    def test_tier1_urban_cr_below_double_bounce_max(self):
        # CR = 0.10 < 0.15 → definitive double-bounce → URBAN
        label = self._make_scene(vv_lin=0.20, cr=0.10)
        self.assertEqual(label, URBAN)

    def test_tier1_urban_cr_at_boundary_exclusive(self):
        # CR just below threshold
        label = self._make_scene(vv_lin=0.20, cr=_CR_DOUBLE_BOUNCE_MAX - 0.001)
        self.assertEqual(label, URBAN)

    def test_tier2_wet_forest_cr_in_intermediate_zone(self):
        # CR = 0.22 in [0.15, 0.30) → WET_FOREST
        label = self._make_scene(vv_lin=0.20, cr=0.22)
        self.assertEqual(label, WET_FOREST, msg="JOB-C7FE2C61 scenario: moisture spike")

    def test_tier2_wet_forest_cr_just_above_lower_boundary(self):
        # CR = 0.151 > 0.15 → WET_FOREST (not Urban).
        # Using 0.151 rather than the exact 0.15 boundary to avoid float32
        # rounding where 0.20 * 0.15 / 0.20 ≈ 0.14999998 (lands in URBAN tier).
        label = self._make_scene(vv_lin=0.20, cr=_CR_DOUBLE_BOUNCE_MAX + 0.001)
        self.assertEqual(label, WET_FOREST)

    def test_tier2_wet_forest_cr_just_below_veg_min(self):
        # CR = 0.299 → still WET_FOREST
        label = self._make_scene(vv_lin=0.20, cr=_CR_VEG_MIN - 0.001)
        self.assertEqual(label, WET_FOREST)

    def test_tier3_vegetation_cr_above_veg_min(self):
        # CR = 0.35 ≥ 0.30 → dense vegetation (volume scattering)
        label = self._make_scene(vv_lin=0.20, cr=0.35)
        self.assertEqual(label, VEGETATION)

    def test_no_vh_conservative_floor_vegetation(self):
        # VV = 0.17 < 0.20, no VH → VEGETATION (conservative floor)
        label = _classify_single(0.17)
        self.assertEqual(label, VEGETATION)

    def test_no_vh_conservative_floor_urban(self):
        # VV = 0.22 ≥ 0.20, no VH → URBAN
        label = _classify_single(0.22)
        self.assertEqual(label, URBAN)

    def test_no_vh_no_wet_forest_class(self):
        # Without VH the WET_FOREST class must never appear for high-VV pixels.
        vv = np.full((4, 4), 0.20, dtype=np.float32)
        result = classify_sar(vv, vh=None)
        self.assertNotIn(WET_FOREST, result)

    def test_ndvi_veto_overrides_urban(self):
        # Even if SAR says URBAN, NDVI > 0.5 forces VEGETATION
        vv   = np.array([[0.20]], dtype=np.float32)
        vh   = np.array([[0.10 * 0.20]], dtype=np.float32)   # CR = 0.10 → URBAN
        ndvi = np.array([[0.75]], dtype=np.float32)
        result = classify_sar(vv, vh, ndvi)
        self.assertEqual(int(result[0, 0]), VEGETATION)


# ── apply_persistence_filter ───────────────────────────────────────────────────

class TestPersistenceFilter(unittest.TestCase):

    def _make_stack(self, labels: list) -> np.ndarray:
        """Build a (T, 1, 1) uint8 stack from a flat list of class IDs."""
        arr = np.array(labels, dtype=np.uint8).reshape(len(labels), 1, 1)
        return arr

    def test_transient_single_spike_downgraded(self):
        # T=5: [VEG, VEG, URBAN, VEG, VEG] → run=1 < 3 → ANOMALOUS
        stack = self._make_stack([VEGETATION, VEGETATION, URBAN, VEGETATION, VEGETATION])
        out = apply_persistence_filter(stack, min_persistence=3)
        self.assertEqual(int(out[2, 0, 0]), ANOMALOUS_VEG_MOISTURE)

    def test_transient_two_consecutive_downgraded(self):
        # run=2 < 3 → ANOMALOUS
        stack = self._make_stack([VEGETATION, URBAN, URBAN, VEGETATION, VEGETATION])
        out = apply_persistence_filter(stack, min_persistence=3)
        self.assertEqual(int(out[1, 0, 0]), ANOMALOUS_VEG_MOISTURE)
        self.assertEqual(int(out[2, 0, 0]), ANOMALOUS_VEG_MOISTURE)

    def test_sustained_three_consecutive_retained(self):
        # run=3 ≥ 3 → URBAN kept
        stack = self._make_stack([VEGETATION, URBAN, URBAN, URBAN, VEGETATION])
        out = apply_persistence_filter(stack, min_persistence=3)
        self.assertEqual(int(out[1, 0, 0]), URBAN)
        self.assertEqual(int(out[2, 0, 0]), URBAN)
        self.assertEqual(int(out[3, 0, 0]), URBAN)

    def test_non_vegetation_baseline_not_filtered(self):
        # Pixel was BARE_SOIL in baseline — persistence filter should not touch it
        stack = self._make_stack([BARE_SOIL, BARE_SOIL, URBAN, BARE_SOIL, BARE_SOIL])
        out = apply_persistence_filter(stack, min_persistence=3)
        # baseline_mask should be False for this pixel (was not VEGETATION)
        # URBAN should therefore be left untouched
        self.assertEqual(int(out[2, 0, 0]), URBAN)

    def test_no_urban_pixels_unchanged(self):
        # No URBAN in stack → output must be identical to input
        stack = self._make_stack([VEGETATION, VEGETATION, VEGETATION])
        out = apply_persistence_filter(stack, min_persistence=3)
        np.testing.assert_array_equal(out, stack)

    def test_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            apply_persistence_filter(np.zeros((3, 4), dtype=np.uint8))

    def test_baseline_majority_vote(self):
        # T=6: first 3 are VEGETATION (baseline window = 3).  Run of URBAN = 2 < 3.
        stack = self._make_stack([
            VEGETATION, VEGETATION, VEGETATION, URBAN, URBAN, VEGETATION
        ])
        out = apply_persistence_filter(stack, min_persistence=3)
        self.assertEqual(int(out[3, 0, 0]), ANOMALOUS_VEG_MOISTURE)
        self.assertEqual(int(out[4, 0, 0]), ANOMALOUS_VEG_MOISTURE)


# ── apply_forest_mask ──────────────────────────────────────────────────────────

class TestForestMask(unittest.TestCase):

    def test_urban_in_high_ndvi_forest_reclassified(self):
        cls  = np.array([[URBAN]], dtype=np.uint8)
        ndvi = np.array([[0.75]], dtype=np.float32)   # > 0.6 threshold
        out  = apply_forest_mask(cls, ndvi)
        self.assertEqual(int(out[0, 0]), ANOMALOUS_VEG_MOISTURE)

    def test_urban_below_ndvi_threshold_retained(self):
        cls  = np.array([[URBAN]], dtype=np.uint8)
        ndvi = np.array([[0.45]], dtype=np.float32)   # < 0.6 → not forest
        out  = apply_forest_mask(cls, ndvi)
        self.assertEqual(int(out[0, 0]), URBAN)

    def test_non_urban_class_not_affected(self):
        cls  = np.array([[VEGETATION]], dtype=np.uint8)
        ndvi = np.array([[0.80]], dtype=np.float32)
        out  = apply_forest_mask(cls, ndvi)
        self.assertEqual(int(out[0, 0]), VEGETATION)

    def test_nan_ndvi_does_not_mask(self):
        cls  = np.array([[URBAN]], dtype=np.uint8)
        ndvi = np.array([[np.nan]], dtype=np.float32)
        out  = apply_forest_mask(cls, ndvi)
        self.assertEqual(int(out[0, 0]), URBAN)

    def test_custom_threshold(self):
        cls  = np.array([[URBAN, URBAN]], dtype=np.uint8)
        ndvi = np.array([[0.55, 0.85]], dtype=np.float32)
        out  = apply_forest_mask(cls, ndvi, ndvi_forest_threshold=0.8)
        self.assertEqual(int(out[0, 0]), URBAN)                    # 0.55 < 0.80
        self.assertEqual(int(out[0, 1]), ANOMALOUS_VEG_MOISTURE)   # 0.85 > 0.80

    def test_custom_replace_class(self):
        cls  = np.array([[URBAN]], dtype=np.uint8)
        ndvi = np.array([[0.75]], dtype=np.float32)
        out  = apply_forest_mask(cls, ndvi, replace_class=WET_FOREST)
        self.assertEqual(int(out[0, 0]), WET_FOREST)


# ── classify_temporal_stack ────────────────────────────────────────────────────

class TestClassifyTemporalStack(unittest.TestCase):
    """
    Integration test: simulate the JOB-C7FE2C61 scenario
    (March 16 – April 11 2026, known forest AOI).

    Acquisitions:
      t0 2026-03-16  VV=-10.8dB (0.0832), VH=-18.3dB (0.0148), CR=0.178 → VEG baseline
      t1 2026-03-22  VV=-10.3dB (0.0933), VH=-17.8dB (0.0166), CR=0.178 → VEG baseline
      t2 2026-03-28  VV= -7.3dB (0.1862), VH=-14.0dB (0.0398), CR=0.214 → WET_FOREST (dual-pol)
      t3 2026-04-03  VV= -7.1dB (0.1950), VH=-15.9dB (0.0257), CR=0.132 → URBAN (single)
                     → ANOMALOUS after forest mask / persistence (run=1)
      t4 2026-04-09  VV= -7.5dB (0.1778), VH=-15.2dB (0.0302), CR=0.170 → WET_FOREST (dual-pol)
      t5 2026-04-11  VV=-10.5dB (0.0891), VH=-18.1dB (0.0155), CR=0.174 → VEG recovery
    """

    DATES = [
        "2026-03-16", "2026-03-22", "2026-03-28",
        "2026-04-03", "2026-04-09", "2026-04-11",
    ]
    # VV and VH linear gamma0 values per acquisition (scalar scene, 1 pixel)
    VV_DB = [-10.8, -10.3, -7.3, -7.1, -7.5, -10.5]
    VH_DB = [-18.3, -17.8, -14.0, -15.9, -15.2, -18.1]

    @staticmethod
    def _db_to_lin(db_val: float) -> float:
        return float(10 ** (db_val / 10.0))

    def setUp(self):
        T = len(self.DATES)
        self.vv = np.array(
            [[[self._db_to_lin(v)]] for v in self.VV_DB], dtype=np.float32
        )   # (6, 1, 1)
        self.vh = np.array(
            [[[self._db_to_lin(v)]] for v in self.VH_DB], dtype=np.float32
        )
        self.ndvi_baseline = np.array([[0.72]], dtype=np.float32)   # dense forest

    def test_baseline_acquisitions_are_vegetation(self):
        class_stack, _ = classify_temporal_stack(
            self.vv, self.DATES, self.vh, self.ndvi_baseline
        )
        self.assertEqual(int(class_stack[0, 0, 0]), VEGETATION)
        self.assertEqual(int(class_stack[1, 0, 0]), VEGETATION)

    def test_moisture_spike_t2_is_wet_forest(self):
        class_stack, _ = classify_temporal_stack(
            self.vv, self.DATES, self.vh, self.ndvi_baseline
        )
        # t2: CR=0.214 in [0.15, 0.30) → WET_FOREST from single-frame, unchanged
        self.assertEqual(int(class_stack[2, 0, 0]), WET_FOREST)

    def test_urban_t3_demoted_by_forest_mask_or_persistence(self):
        # t3 single-frame: CR=0.132 < 0.15 → URBAN
        # forest mask (NDVI=0.72 > 0.6) should catch it first
        class_stack, _ = classify_temporal_stack(
            self.vv, self.DATES, self.vh, self.ndvi_baseline
        )
        self.assertEqual(int(class_stack[3, 0, 0]), ANOMALOUS_VEG_MOISTURE)

    def test_moisture_spike_t4_is_wet_forest(self):
        class_stack, _ = classify_temporal_stack(
            self.vv, self.DATES, self.vh, self.ndvi_baseline
        )
        self.assertEqual(int(class_stack[4, 0, 0]), WET_FOREST)

    def test_recovery_t5_is_vegetation(self):
        class_stack, _ = classify_temporal_stack(
            self.vv, self.DATES, self.vh, self.ndvi_baseline
        )
        self.assertEqual(int(class_stack[5, 0, 0]), VEGETATION)

    def test_trend_length_matches_dates(self):
        _, trend = classify_temporal_stack(self.vv, self.DATES, self.vh)
        self.assertEqual(len(trend), len(self.DATES))

    def test_trend_vv_db_values_reasonable(self):
        _, trend = classify_temporal_stack(self.vv, self.DATES, self.vh)
        for entry in trend:
            self.assertIsNotNone(entry["vv_db"])
            # All values must be negative dB (sub-unity linear power)
            self.assertLess(entry["vv_db"], 0.0)

    def test_mismatched_dates_raises(self):
        with self.assertRaises(ValueError):
            classify_temporal_stack(self.vv, self.DATES[:3], self.vh)

    def test_wrong_ndim_raises(self):
        with self.assertRaises(ValueError):
            classify_temporal_stack(self.vv[0], self.DATES[:1], self.vh[0])


# ── generate_backscatter_trend_report ──────────────────────────────────────────

class TestBackscatterTrendReport(unittest.TestCase):

    def _sample_trend(self) -> list:
        return [
            {"date": "2026-03-16", "vv_db": -10.8, "vh_db": -18.3,
             "dominant_class_id": VEGETATION, "dominant_cover": "Vegetation / Forest"},
            {"date": "2026-03-28", "vv_db":  -7.3, "vh_db": -14.0,
             "dominant_class_id": WET_FOREST, "dominant_cover": "Wet Forest / Dielectric Anomaly"},
            {"date": "2026-04-03", "vv_db":  -7.1, "vh_db": -15.9,
             "dominant_class_id": ANOMALOUS_VEG_MOISTURE,
             "dominant_cover": "Anomalous Vegetation / Moisture"},
        ]

    def test_returns_string(self):
        report = generate_backscatter_trend_report(self._sample_trend())
        self.assertIsInstance(report, str)

    def test_contains_all_dates(self):
        report = generate_backscatter_trend_report(self._sample_trend())
        for entry in self._sample_trend():
            self.assertIn(entry["date"], report)

    def test_contains_cover_labels(self):
        report = generate_backscatter_trend_report(self._sample_trend())
        self.assertIn("Wet Forest", report)
        self.assertIn("Anomalous Vegetation", report)

    def test_vv_vh_ratio_computed(self):
        # VV/VH (dB) for row 0 should be -10.8 - (-18.3) = 7.5
        report = generate_backscatter_trend_report(self._sample_trend())
        self.assertIn("7.5", report)

    def test_none_values_render_as_na(self):
        trend = [{"date": "2026-03-16", "vv_db": None, "vh_db": None,
                  "dominant_class_id": NO_DATA, "dominant_cover": "No Data"}]
        report = generate_backscatter_trend_report(trend)
        self.assertIn("N/A", report)

    def test_empty_trend_returns_separator_only(self):
        report = generate_backscatter_trend_report([])
        self.assertTrue(len(report) > 0)


# ── classification_stats backward compat ──────────────────────────────────────

class TestClassificationStats(unittest.TestCase):

    def test_all_classes_present_in_output(self):
        cls_map = np.array([[WATER, BARE_SOIL], [VEGETATION, URBAN]], dtype=np.uint8)
        stats   = classification_stats(cls_map)
        for cls_id in (WATER, BARE_SOIL, VEGETATION, URBAN):
            self.assertIn(cls_id, stats)

    def test_new_classes_counted(self):
        cls_map = np.array(
            [[ANOMALOUS_VEG_MOISTURE, WET_FOREST]], dtype=np.uint8
        )
        stats = classification_stats(cls_map)
        self.assertEqual(stats[ANOMALOUS_VEG_MOISTURE]["count"], 1)
        self.assertEqual(stats[WET_FOREST]["count"], 1)

    def test_no_data_excluded_from_pct(self):
        cls_map = np.array([[VEGETATION, NO_DATA]], dtype=np.uint8)
        stats   = classification_stats(cls_map)
        self.assertAlmostEqual(stats[VEGETATION]["pct"], 100.0)


# ── CLASS_PLAIN_LANGUAGE ───────────────────────────────────────────────────────

class TestClassPlainLanguage(unittest.TestCase):

    def test_all_classes_have_plain_language_entry(self):
        for cls_id in (WATER, BARE_SOIL, VEGETATION, URBAN,
                       ANOMALOUS_VEG_MOISTURE, WET_FOREST):
            self.assertIn(cls_id, CLASS_PLAIN_LANGUAGE)
            self.assertGreater(len(CLASS_PLAIN_LANGUAGE[cls_id]), 20)

    def test_anomalous_is_marked_as_confidence_filter(self):
        text = CLASS_PLAIN_LANGUAGE[ANOMALOUS_VEG_MOISTURE].upper()
        self.assertIn("FILTER", text)

    def test_urban_mentions_double_bounce(self):
        text = CLASS_PLAIN_LANGUAGE[URBAN].lower()
        self.assertIn("double-bounce", text)

    def test_wet_forest_mentions_precipitation_guard(self):
        text = CLASS_PLAIN_LANGUAGE[WET_FOREST].lower()
        self.assertIn("false-positive", text)


# ── explain_pixel ──────────────────────────────────────────────────────────────

class TestExplainPixel(unittest.TestCase):

    def test_invalid_vv_returns_no_data(self):
        result = explain_pixel(float("nan"))
        self.assertEqual(result["final_class"], NO_DATA)
        self.assertEqual(result["confidence"], "N/A")

    def test_water_pixel_trace(self):
        result = explain_pixel(vv=0.005, vh=0.001)
        self.assertEqual(result["final_class"], WATER)
        self.assertEqual(result["confidence"], "HIGH")
        self.assertTrue(any("WATER" in s for s in result["trace"]))

    def test_bare_soil_pixel_trace(self):
        result = explain_pixel(vv=0.03)
        self.assertEqual(result["final_class"], BARE_SOIL)
        self.assertEqual(result["confidence"], "HIGH")

    def test_vegetation_intermediate_vv(self):
        result = explain_pixel(vv=0.08)
        self.assertEqual(result["final_class"], VEGETATION)
        self.assertEqual(result["confidence"], "HIGH")

    def test_urban_tier1_full_trace(self):
        # CR = 0.10 < 0.15 → URBAN, persistence_run=3 ≥ 3 → confirmed
        result = explain_pixel(vv=0.20, vh=0.02, persistence_run=3)
        self.assertEqual(result["final_class"], URBAN)
        self.assertEqual(result["confidence"], "HIGH")
        trace_text = " ".join(result["trace"])
        self.assertIn("Tier\u202f1", trace_text)
        self.assertIn("CONFIRMED", trace_text)

    def test_wet_forest_tier2(self):
        # CR = 0.22 in [0.15, 0.30)
        result = explain_pixel(vv=0.20, vh=0.044)
        self.assertEqual(result["final_class"], WET_FOREST)
        self.assertEqual(result["confidence"], "MEDIUM")
        self.assertIn("Tier\u202f2", " ".join(result["trace"]))

    def test_ndvi_veto_overrides_urban(self):
        # CR → URBAN, but NDVI=0.75 overrides
        result = explain_pixel(vv=0.20, vh=0.02, ndvi=0.75)
        self.assertEqual(result["final_class"], VEGETATION)
        trace_text = " ".join(result["trace"])
        self.assertIn("NDVI Veto", trace_text)
        self.assertIn("override applied", trace_text)

    def test_forest_mask_demotes_urban(self):
        # CR → URBAN, high NDVI baseline → ANOMALOUS
        result = explain_pixel(vv=0.20, vh=0.02, ndvi_baseline=0.75)
        self.assertEqual(result["final_class"], ANOMALOUS_VEG_MOISTURE)
        trace_text = " ".join(result["trace"])
        self.assertIn("Forest Mask", trace_text)

    def test_persistence_demotes_insufficient_urban(self):
        # CR → URBAN, persistence_run=1 < 3 → ANOMALOUS
        result = explain_pixel(vv=0.20, vh=0.02, persistence_run=1)
        self.assertEqual(result["final_class"], ANOMALOUS_VEG_MOISTURE)
        self.assertEqual(result["confidence"], "LOW")

    def test_plain_language_present(self):
        result = explain_pixel(vv=0.08)
        self.assertIn("plain_language", result)
        self.assertGreater(len(result["plain_language"]), 0)

    def test_trace_is_ordered_list(self):
        result = explain_pixel(vv=0.20, vh=0.02)
        self.assertIsInstance(result["trace"], list)
        self.assertGreater(len(result["trace"]), 2)
        # Last step always contains final class
        self.assertIn("Final Class", result["trace"][-1])


# ── suggest_cr_thresholds ─────────────────────────────────────────────────────

class TestSuggestCRThresholds(unittest.TestCase):

    def test_alpine_temperate_returns_current_values(self):
        profile = suggest_cr_thresholds("alpine_temperate")
        self.assertEqual(profile["CR_DOUBLE_BOUNCE_MAX"], _CR_DOUBLE_BOUNCE_MAX)
        self.assertEqual(profile["CR_VEG_MIN"], _CR_VEG_MIN)

    def test_tropical_raises_cr_thresholds(self):
        profile = suggest_cr_thresholds("tropical")
        self.assertGreater(profile["CR_DOUBLE_BOUNCE_MAX"], _CR_DOUBLE_BOUNCE_MAX)
        self.assertGreater(profile["CR_VEG_MIN"], _CR_VEG_MIN)
        self.assertGreaterEqual(profile["min_persistence"], 4)

    def test_arid_lowers_vv_veg_max(self):
        profile = suggest_cr_thresholds("arid")
        self.assertLess(profile["VV_VEG_MAX"], _VV_VEG_MAX)

    def test_boreal_lowers_cr_veg_min_slightly(self):
        alpine = suggest_cr_thresholds("alpine_temperate")
        boreal  = suggest_cr_thresholds("boreal")
        self.assertLessEqual(boreal["CR_VEG_MIN"], alpine["CR_VEG_MIN"])

    def test_case_insensitive_lookup(self):
        a = suggest_cr_thresholds("Tropical")
        b = suggest_cr_thresholds("tropical")
        self.assertEqual(a["CR_DOUBLE_BOUNCE_MAX"], b["CR_DOUBLE_BOUNCE_MAX"])

    def test_unknown_climate_raises_value_error(self):
        with self.assertRaises(ValueError):
            suggest_cr_thresholds("arctic_tundra")

    def test_threshold_diff_populated_for_tropical(self):
        profile = suggest_cr_thresholds("tropical")
        self.assertIn("threshold_diff", profile)
        self.assertGreater(len(profile["threshold_diff"]), 0)

    def test_rationale_is_non_empty_string(self):
        for climate in ("alpine_temperate", "tropical", "arid", "boreal"):
            profile = suggest_cr_thresholds(climate)
            self.assertIsInstance(profile["rationale"], str)
            self.assertGreater(len(profile["rationale"]), 50)


# ── generate_validation_protocol ─────────────────────────────────────────────

class TestGenerateValidationProtocol(unittest.TestCase):

    def test_total_points_equals_n_points(self):
        protocol = generate_validation_protocol(area_km2=14.49, n_points=50)
        total = sum(protocol["per_class_samples"].values())
        self.assertEqual(total, 50)

    def test_minimum_3_points_per_class(self):
        protocol = generate_validation_protocol(n_points=50)
        for cls_label, count in protocol["per_class_samples"].items():
            self.assertGreaterEqual(count, 3, msg=f"{cls_label} has < 3 points")

    def test_min_spacing_positive(self):
        protocol = generate_validation_protocol(area_km2=14.49, n_points=50)
        self.assertGreater(protocol["min_spacing_km"], 0)

    def test_custom_area_changes_spacing(self):
        p1 = generate_validation_protocol(area_km2=14.49, n_points=50)
        p2 = generate_validation_protocol(area_km2=100.0, n_points=50)
        self.assertGreater(p2["min_spacing_km"], p1["min_spacing_km"])

    def test_field_columns_include_required_keys(self):
        protocol = generate_validation_protocol()
        required = {"point_id", "latitude", "longitude",
                    "automated_class_label", "reference_class_label"}
        self.assertTrue(required.issubset(set(protocol["field_columns"])))

    def test_standard_is_iso_19157(self):
        protocol = generate_validation_protocol()
        self.assertIn("ISO 19157", protocol["standard"])


# ── compute_accuracy_metrics ──────────────────────────────────────────────────

class TestComputeAccuracyMetrics(unittest.TestCase):

    def _perfect_cm(self, n=3):
        """n×n identity confusion matrix (perfect classification)."""
        return np.eye(n, dtype=int) * 10

    def test_perfect_classification_oa_100(self):
        result = compute_accuracy_metrics(self._perfect_cm())
        self.assertAlmostEqual(result["overall_accuracy_pct"], 100.0)

    def test_perfect_classification_kappa_1(self):
        result = compute_accuracy_metrics(self._perfect_cm())
        self.assertAlmostEqual(result["kappa_coefficient"], 1.0, places=4)

    def test_kappa_almost_perfect_label(self):
        result = compute_accuracy_metrics(self._perfect_cm())
        self.assertIn("Almost perfect", result["kappa_interpretation"])

    def test_producers_accuracy_100_on_diagonal(self):
        result = compute_accuracy_metrics(
            self._perfect_cm(), class_names=["A", "B", "C"]
        )
        for name in ("A", "B", "C"):
            self.assertAlmostEqual(result["producers_accuracy_pct"][name], 100.0)

    def test_users_accuracy_100_on_diagonal(self):
        result = compute_accuracy_metrics(
            self._perfect_cm(), class_names=["A", "B", "C"]
        )
        for name in ("A", "B", "C"):
            self.assertAlmostEqual(result["users_accuracy_pct"][name], 100.0)

    def test_off_diagonal_reduces_oa(self):
        cm = np.array([[9, 1], [2, 8]], dtype=int)
        result = compute_accuracy_metrics(cm, class_names=["X", "Y"])
        self.assertLess(result["overall_accuracy_pct"], 100.0)
        self.assertGreater(result["overall_accuracy_pct"], 0.0)

    def test_kappa_below_1_for_imperfect(self):
        cm = np.array([[9, 1], [2, 8]], dtype=int)
        result = compute_accuracy_metrics(cm)
        self.assertLess(result["kappa_coefficient"], 1.0)

    def test_standard_is_iso_19157(self):
        result = compute_accuracy_metrics(self._perfect_cm())
        self.assertIn("ISO 19157", result["standard"])

    def test_non_square_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_accuracy_metrics(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_all_zeros_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_accuracy_metrics(np.zeros((3, 3)))

    def test_class_names_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_accuracy_metrics(self._perfect_cm(3), class_names=["A", "B"])

    def test_n_control_points_matches_sum(self):
        cm = np.array([[9, 1], [2, 8]], dtype=int)
        result = compute_accuracy_metrics(cm)
        self.assertEqual(result["n_control_points"], 20)


if __name__ == "__main__":
    unittest.main()
