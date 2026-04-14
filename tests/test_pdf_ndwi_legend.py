"""
Regression tests — NDWI hydraulic-status interpretation & legend consistency.

Verifies that _ndwi_interpretation() returns water-centric labels and that
_ndvi_interpretation() is never called for NDWI values (index isolation).
"""
import pytest
from app.reporting.pdf_gen import _ndwi_interpretation, _ndvi_interpretation


# ── Hydraulic Status breakpoint tests ──────────────────────────────────────────

@pytest.mark.parametrize("mean, expected", [
    (0.50,  "Surface Water / Flooding"),
    (0.31,  "Surface Water / Flooding"),
    (0.30,  "Elevated Moisture / Wet Canopy"),   # boundary: NOT > 0.3
    (0.244, "Elevated Moisture / Wet Canopy"),    # regression: was "Sparse vegetation"
    (0.10,  "Elevated Moisture / Wet Canopy"),
    (0.09,  "Balanced Moisture / Transitional"),
    (0.00,  "Balanced Moisture / Transitional"),
    (-0.10, "Balanced Moisture / Transitional"),
    (-0.11, "Arid / Dry Surface"),
    (-0.50, "Arid / Dry Surface"),
])
def test_ndwi_interpretation_hydraulic_scale(mean, expected):
    assert _ndwi_interpretation(mean) == expected


# ── Cross-index isolation: NDVI must NOT return water labels ──────────────────

def test_ndvi_does_not_return_water_labels():
    """NDVI labels must never bleed into the NDWI interpretation path."""
    ndwi_specific = {
        "Surface Water / Flooding",
        "Elevated Moisture / Wet Canopy",
        "Balanced Moisture / Transitional",
        "Arid / Dry Surface",
    }
    for mean in [0.50, 0.30, 0.244, 0.10, 0.00, -0.10, -0.50]:
        ndvi_label, _ = _ndvi_interpretation(mean)
        assert ndvi_label not in ndwi_specific, (
            f"NDVI interpretation returned NDWI label '{ndvi_label}' for mean={mean}"
        )


# ── NDVI vegetation scale unaffected ──────────────────────────────────────────

@pytest.mark.parametrize("mean, expected", [
    (0.70, "Dense vegetation"),
    (0.45, "Moderate vegetation"),
    (0.20, "Sparse vegetation"),
    (0.00, "Bare soil / urban"),
    (-0.20, "Water / snow / cloud"),
])
def test_ndvi_interpretation_vegetation_scale(mean, expected):
    label, _ = _ndvi_interpretation(mean)
    assert label == expected
