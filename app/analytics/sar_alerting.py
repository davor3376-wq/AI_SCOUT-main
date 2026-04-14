"""
SAR Backscatter Change Detection.
Computes pixel-level VV delta between two acquisitions and flags anomalies
using a combined sigma + absolute-floor threshold — the "Delta Report" layer.

Threshold design
----------------
A pixel is considered anomalous only when it exceeds BOTH:
  (a) sigma_threshold × std(delta)  — relative, scene-adaptive
  (b) abs_delta_min                 — absolute physical floor

Condition (a) alone is statistically self-defeating: using 1 σ means ~31.7 % of
pixels from ANY distribution fall outside by definition, guaranteeing a HIGH
alert even in a perfectly stable scene.  Raising to 2 σ brings that baseline
to ~4.5 %, well inside the LOW band.  The absolute floor (b) prevents sub-noise
fluctuations (< 0.02 linear ≈ 0.4 dB at typical forest/urban signal levels)
from inflating the anomaly count.

Alert levels (percentage of valid pixels that exceed BOTH thresholds)
----------------------------------------------------------------------
  LOW    — < 10 %   (sensor noise, phenological variation)
  MEDIUM — 10–20 %  (localised change, monitor for trend)
  HIGH   — > 20 %   (widespread structural or hydrological event)
"""
import numpy as np
from typing import Tuple, Dict, Any


def compute_sar_delta(
    current_vv: np.ndarray,
    previous_vv: np.ndarray,
    sigma_threshold: float = 2.0,
    abs_delta_min: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute VV backscatter delta and classify anomalous pixels.

    Args:
        current_vv:       VV array from the newer acquisition (linear gamma0).
        previous_vv:      VV array from the older acquisition (same geometry).
        sigma_threshold:  Sigma multiplier for the relative threshold.
                          Default 2.0 — at 2 σ only ~4.5 % of values from a
                          stable (normally distributed) delta field fall outside,
                          keeping benign scenes well within the LOW band.
        abs_delta_min:    Absolute floor (linear gamma0).  A pixel must also
                          change by at least this amount to be counted as
                          anomalous regardless of sigma.  Default 0.02 (≈ 0.4 dB
                          at signal levels around 0.20), which is approximately
                          2× the temporal precision of Sentinel-1 IW mode.

    Returns:
        delta:  float32 array of (current − previous).  NaN where either
                input was non-finite.
        stats:  dict with keys —
                  mean_delta, std_delta, anomaly_threshold, abs_delta_min,
                  anomaly_pct, decrease_pct, increase_pct,
                  alert_level, valid_pixels, sigma_threshold
    """
    valid = np.isfinite(current_vv) & np.isfinite(previous_vv)
    delta = np.where(valid, current_vv - previous_vv, np.nan).astype(np.float32)

    valid_delta = delta[np.isfinite(delta)]
    if valid_delta.size == 0:
        return delta, {
            "mean_delta":        0.0,
            "std_delta":         0.0,
            "anomaly_threshold": 0.0,
            "abs_delta_min":     abs_delta_min,
            "anomaly_pct":       0.0,
            "decrease_pct":      0.0,
            "increase_pct":      0.0,
            "alert_level":       "LOW",
            "valid_pixels":      0,
            "sigma_threshold":   sigma_threshold,
        }

    mean_d = float(np.mean(valid_delta))
    std_d  = float(np.std(valid_delta))
    # Relative threshold: sigma band around the distribution
    sigma_band = sigma_threshold * std_d
    # Effective threshold: pixel must exceed BOTH the sigma band and the physical floor
    effective_threshold = max(sigma_band, abs_delta_min)

    finite_delta = np.isfinite(delta)
    total = int(valid_delta.size)

    anomaly_mask  = finite_delta & (np.abs(delta) > effective_threshold)
    decrease_mask = finite_delta & (delta < -effective_threshold)
    increase_mask = finite_delta & (delta >  effective_threshold)

    anomaly_pct  = float(np.sum(anomaly_mask)  / total * 100.0)
    decrease_pct = float(np.sum(decrease_mask) / total * 100.0)
    increase_pct = float(np.sum(increase_mask) / total * 100.0)

    if anomaly_pct > 20.0:
        alert_level = "HIGH"
    elif anomaly_pct > 10.0:
        alert_level = "MEDIUM"
    else:
        alert_level = "LOW"

    stats: Dict[str, Any] = {
        "mean_delta":        mean_d,
        "std_delta":         std_d,
        "anomaly_threshold": effective_threshold,
        "abs_delta_min":     abs_delta_min,
        "anomaly_pct":       anomaly_pct,
        "decrease_pct":      decrease_pct,
        "increase_pct":      increase_pct,
        "alert_level":       alert_level,
        "valid_pixels":      total,
        "sigma_threshold":   sigma_threshold,
    }
    return delta, stats


def delta_interpretation(stats: Dict[str, Any]) -> str:
    """
    Land-cover-agnostic one-line summary of the delta report.

    Deliberately avoids causal inference (flooding, construction, clearance).
    All language describes the observed backscatter signal only; contextual
    interpretation is the responsibility of the analyst.
    """
    level   = stats.get("alert_level", "LOW")
    dec     = stats.get("decrease_pct", 0.0)
    inc     = stats.get("increase_pct", 0.0)
    sigma   = stats.get("sigma_threshold", 2.0)
    floor   = stats.get("abs_delta_min", 0.02)
    anom    = stats.get("anomaly_pct", 0.0)

    gate = f"({sigma:.1f}\u03c3 \u2227 \u2265{floor:.3f} linear)"

    if level == "HIGH":
        if dec > inc:
            return (
                f"HIGH — Significant Backscatter Deviation detected. "
                f"{dec:.1f}% of pixels exceed the dual threshold {gate} with a "
                f"decrease in VV intensity. Field verification required."
            )
        return (
            f"HIGH — Significant Backscatter Deviation detected. "
            f"{inc:.1f}% of pixels exceed the dual threshold {gate} with an "
            f"increase in VV intensity. Field verification required."
        )
    if level == "MEDIUM":
        return (
            f"MEDIUM — Backscatter Deviation detected. "
            f"{anom:.1f}% of pixels exceed the dual threshold {gate}. "
            f"Monitor for trend continuation."
        )
    return (
        f"LOW — {anom:.1f}% of pixels exceed the dual threshold {gate}. "
        f"Surface characteristics nominally stable."
    )
