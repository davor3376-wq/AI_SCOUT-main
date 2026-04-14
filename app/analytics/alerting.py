import numpy as np

# Anomalous-pixel thresholds for alert escalation.
# These are independent of NDVI so that change-detection signals in SAR delta
# reports (or any other source) can elevate the alert even when mean NDVI looks
# healthy (e.g. intact Amazon canopy that is still showing 6 %+ anomalous pixels).
_ANOMALY_HIGH_PCT = 10.0    # ≥10 % anomalous pixels → HIGH
_ANOMALY_MEDIUM_PCT = 5.0   # ≥5 % anomalous pixels → at least MEDIUM
COREG_TOLERANCE_PCT = 1.0   # max allowable valid-pixel-count variance between acquisitions


def check_coregistration(valid_pixel_counts: list[int]) -> bool:
    """
    Returns True when all acquisitions share a consistent pixel grid.

    A variance of more than COREG_TOLERANCE_PCT between the minimum and maximum
    valid-pixel count indicates that acquisitions are not on the same grid, making
    pixel-level delta analysis unreliable.

    Args:
        valid_pixel_counts: List of valid (non-masked) pixel counts, one per
            acquisition in the analysis stack.

    Returns:
        True  → co-registration OK (variance within tolerance).
        False → co-registration FAILED (variance exceeds tolerance).
    """
    if len(valid_pixel_counts) < 2:
        return True

    max_count = max(valid_pixel_counts)
    if max_count == 0:
        return False

    min_count = min(valid_pixel_counts)
    variance_pct = (max_count - min_count) / max_count * 100.0

    if variance_pct > COREG_TOLERANCE_PCT:
        import logging
        logging.getLogger(__name__).warning(
            "Co-registration WARNING: valid pixel count varies %.1f %% across "
            "acquisitions (min=%d, max=%d). Threshold is %.1f %%. "
            "Delta conclusions are unreliable.",
            variance_pct, min_count, max_count, COREG_TOLERANCE_PCT,
        )
        return False

    return True


def calculate_alert_level(
    ndvi: np.ndarray,
    cloud_cover_pct: float,
    anomalous_pixel_pct: float = 0.0,
    coreg_valid: bool = True,
    gate_status: str = "PASS",
) -> str:
    """
    Determines the Alert Level based on NDVI, cloud cover, anomalous-pixel
    percentage, co-registration validity, and quality gate status.

    Args:
        ndvi: NDVI raster array for the current period.
        cloud_cover_pct: Percentage of pixels masked as cloud/shadow (0–100).
        anomalous_pixel_pct: Percentage of pixels flagged as anomalous by the
            delta / change-detection step (0–100).  Defaults to 0.0 so that
            callers that have not yet been updated keep the old behaviour.
        coreg_valid: False when the geometric co-registration check has failed
            (e.g. valid-pixel-count variance between acquisitions exceeds the
            1 % tolerance).  An invalid co-registration forces at least MEDIUM
            because pixel-level change conclusions are unreliable.
        gate_status: String status returned by quality_gate_coregistration.
            Any value other than "PASS" immediately forces "ERROR: INVALID DATA",
            overriding all backscatter and NDVI metrics.

    Returns:
        "LOW", "MEDIUM", "HIGH", or "ERROR: INVALID DATA".
    """
    if gate_status != "PASS":
        return "ERROR: INVALID DATA"

    if cloud_cover_pct > 50.0:
        # Scene too obscured to draw conclusions, but still respect a
        # co-registration failure as a data-quality signal.
        return "MEDIUM" if not coreg_valid else "LOW"

    mean_ndvi = np.nanmean(ndvi)

    # --- NDVI-based vegetation health ---
    if mean_ndvi < 0.2:
        ndvi_level = "HIGH"
    elif mean_ndvi < 0.4:
        ndvi_level = "MEDIUM"
    else:
        ndvi_level = "LOW"

    # --- Anomalous-pixel escalation (change-detection signal) ---
    if anomalous_pixel_pct >= _ANOMALY_HIGH_PCT:
        anomaly_level = "HIGH"
    elif anomalous_pixel_pct >= _ANOMALY_MEDIUM_PCT:
        anomaly_level = "MEDIUM"
    else:
        anomaly_level = "LOW"

    # --- Co-registration gate ---
    # If grids are misaligned we cannot trust pixel-level delta conclusions.
    # Floor the result at MEDIUM so the report does not falsely conclude stability.
    coreg_level = "LOW" if coreg_valid else "MEDIUM"

    # Return the worst (highest) of the three signals.
    _rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    _unrank = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

    final_rank = max(_rank[ndvi_level], _rank[anomaly_level], _rank[coreg_level])
    return _unrank[final_rank]
