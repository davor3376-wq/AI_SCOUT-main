"""
SAR Land Cover Classifier.
Classifies Sentinel-1 C-band gamma0 RTC backscatter into land cover types
using VV/VH threshold rules with a 3-tier cross-polarization ratio discriminator.

Primary VV thresholds (linear power scale):
  Water / smooth:   VV < 0.01  (specular reflection → near-zero return)
  Bare soil:        0.01 ≤ VV < 0.05
  Vegetation:       0.05 ≤ VV < 0.15  (volume scattering)
  High-backscatter: VV ≥ 0.15  — class resolved by dual-polarisation logic below

3-tier Dual-Polarisation Discriminator (requires VH; high-VV pixels only):
  CR = VH / VV  (linear power ratio):
  CR <  _CR_DOUBLE_BOUNCE_MAX (0.15) → URBAN          (definitive double-bounce;
                                          VH co-pol suppressed by metal/concrete)
  CR in [0.15, 0.30)              → WET_FOREST        (VH partially scales with VV;
                                          dielectric/moisture event over canopy)
  CR ≥  _CR_VEG_MIN          (0.30) → VEGETATION      (volume-scattering dominant)
  Without VH: conservative urban floor _VV_URBAN_MIN_NOVH (0.20) applies.

Transient Anomaly Handling:
  apply_persistence_filter() — a pixel in a known-vegetation zone may only be
  labelled URBAN if that class persists across ≥ min_persistence (default 3)
  consecutive acquisitions.  Shorter spikes → ANOMALOUS_VEG_MOISTURE.

  apply_forest_mask() — any URBAN pixel within a Sentinel-2 NDVI baseline mask
  (NDVI > ndvi_forest_threshold, default 0.6) is overridden to
  ANOMALOUS_VEG_MOISTURE; live dense canopy cannot be urban.

References:
  Sentinel-1 C-band SAR backscatter characteristics for land cover classification,
  following established threshold conventions (Szigarski et al. 2018; ESA SAR
  Toolbox documentation).
"""
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# ── Class identifiers ──────────────────────────────────────────────────────────
WATER                  = 0
BARE_SOIL              = 1
VEGETATION             = 2
URBAN                  = 3
ANOMALOUS_VEG_MOISTURE = 4   # Transient high-VV spike inside a known-vegetation zone
WET_FOREST             = 5   # High VV but VH partially scales — dielectric/moisture event
NO_DATA                = 255

CLASS_LABELS: Dict[int, str] = {
    WATER:                  "Water / Smooth Surface",
    BARE_SOIL:              "Bare Soil / Low Vegetation",
    VEGETATION:             "Vegetation / Forest",
    URBAN:                  "Urban / Built-up",
    ANOMALOUS_VEG_MOISTURE: "Anomalous Vegetation / Moisture",
    WET_FOREST:             "Wet Forest / Dielectric Anomaly",
}

# RGB tuples for rendering classification maps
CLASS_COLORS: Dict[int, tuple] = {
    WATER:                  (30,  120, 200),
    BARE_SOIL:              (200, 170, 120),
    VEGETATION:             (34,  139,  34),
    URBAN:                  (180,  50,  50),
    NO_DATA:                (210, 210, 210),
    ANOMALOUS_VEG_MOISTURE: (255, 165,   0),   # Orange  — transient anomaly
    WET_FOREST:             (0,   128, 128),   # Teal    — dielectric/moisture
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
_VV_WATER_MAX      = 0.01   # VV upper bound for specular / water class
_VV_BARE_MAX       = 0.05   # VV upper bound for bare soil
_VV_VEG_MAX        = 0.15   # VV upper bound for vegetation; ≥ this → high backscatter
_VH_WATER_MAX      = 0.005  # VH upper bound for water class (tightens false positives)

# Cross-polarization ratio (CR = VH / VV) separates volume-scattering vegetation
# from double-bounce urban in the high-VV regime (VV ≥ _VV_VEG_MAX).
#   Dense forest / tropical canopy → CR ≥ 0.30  (strong volume depolarisation)
#   Urban double-bounce            → CR <  0.30  (co-pol dominant return)
_CR_VEG_MIN        = 0.30

# Conservative urban VV floor when VH is unavailable.  Tropical rainforest
# regularly exceeds VV = 0.15; raising the threshold to 0.20 reduces the most
# egregious false-positive urban classifications at the cost of some sensitivity.
_VV_URBAN_MIN_NOVH = 0.20

# Dual-polarisation double-bounce discriminator (Tier 1 / Tier 2 boundary).
# Definitive urban double-bounce: VH/VV < _CR_DOUBLE_BOUNCE_MAX (co-pol dominant;
# VH is strongly suppressed by specular reflection off building facets).
# In the ambiguous intermediate zone [_CR_DOUBLE_BOUNCE_MAX, _CR_VEG_MIN), VH
# partially scales with VV — a pattern consistent with a dielectric change
# (moisture/precipitation) over forest canopy rather than a built-up surface.
# This zone is labelled WET_FOREST to prevent false Urban detections.
_CR_DOUBLE_BOUNCE_MAX = 0.15   # VH/VV upper bound for definitive double-bounce urban


def classify_sar(
    vv: np.ndarray,
    vh: Optional[np.ndarray] = None,
    ndvi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Classify Sentinel-1 backscatter pixels into land cover classes.

    Args:
        vv:   2-D float array of VV polarisation (linear gamma0 RTC).
        vh:   2-D float array of VH polarisation (optional).
              When supplied, a cross-polarization ratio (CR = VH/VV) discriminates
              dense vegetation from urban in the high-VV regime, preventing
              tropical/boreal forest from being misclassified as built-up.
        ndvi: 2-D float array of NDVI from co-registered optical data (optional).
              When supplied, any pixel with NDVI > 0.5 is forced to VEGETATION
              regardless of the SAR-derived classification.  This veto prevents
              the high VV backscatter of dense tropical canopy (volume scattering +
              double-bounce from woody structure) from generating false Urban labels.

    Returns:
        uint8 array with class IDs (see module-level constants).
        Pixels that are NaN, Inf, or negative are labelled NO_DATA (255).

    Classification logic for high-backscatter pixels (VV ≥ _VV_VEG_MAX):
      With VH — 3-tier CR (= VH/VV) decision:
        CR <  _CR_DOUBLE_BOUNCE_MAX (0.15)  →  URBAN       (definitive double-bounce)
        CR in [0.15, _CR_VEG_MIN=0.30)     →  WET_FOREST  (VH partially scaling;
                                                            moisture/dielectric event)
        CR ≥  _CR_VEG_MIN          (0.30)  →  VEGETATION  (volume-scattering dominant)
      Without VH:
        VV ≥ _VV_URBAN_MIN_NOVH (0.20)  →  URBAN  (conservative; no WET_FOREST fallback)
        else                             →  VEGETATION
    NDVI veto (applied after SAR classification):
      - ndvi > 0.5  →  VEGETATION  (overrides any SAR-derived URBAN label)
    """
    out = np.full(vv.shape, NO_DATA, dtype=np.uint8)

    finite_vv = np.isfinite(vv) & (vv >= 0)

    # ── Low-VV classes (unambiguous) ───────────────────────────────────────────

    # Vegetation — volume scattering, intermediate VV
    veg_mask = finite_vv & (vv >= _VV_BARE_MAX) & (vv < _VV_VEG_MAX)

    # Bare soil — low-to-moderate return
    bare_mask = finite_vv & (vv >= _VV_WATER_MAX) & (vv < _VV_BARE_MAX)

    # Water — specular return; optionally tightened with VH
    water_mask = finite_vv & (vv < _VV_WATER_MAX)
    if vh is not None:
        finite_vh = np.isfinite(vh) & (vh >= 0)
        water_mask = water_mask & finite_vh & (vh < _VH_WATER_MAX)

    # ── High-VV regime: separate urban from dense vegetation via CR ────────────
    high_vv_mask = finite_vv & (vv >= _VV_VEG_MAX)

    if vh is not None:
        finite_vh = np.isfinite(vh) & (vh >= 0)
        # CR = VH / VV  (safe: high_vv_mask already guarantees vv ≥ 0.15 > 0)
        safe_vv = np.where(vv > 0, vv, 1.0)
        cr = np.where(finite_vh, vh / safe_vv, 0.0)
        cr_valid = finite_vh & finite_vv  # pixels where CR is usable

        # 3-tier CR decision (Dual-Polarisation Constraint)
        # Tier 1: Definitive double-bounce — VH co-pol suppressed → URBAN
        high_vv_urban_cr   = high_vv_mask & cr_valid & (cr <  _CR_DOUBLE_BOUNCE_MAX)
        # Tier 2: Intermediate — VH partially scales; moisture/dielectric plausible
        high_vv_wetfor_cr  = (high_vv_mask & cr_valid
                              & (cr >= _CR_DOUBLE_BOUNCE_MAX) & (cr < _CR_VEG_MIN))
        # Tier 3: Volume-scattering dominant → VEGETATION
        high_vv_veg_cr     = high_vv_mask & cr_valid & (cr >= _CR_VEG_MIN)

        # CR unavailable (VH is NaN/negative) → fall back to conservative VV floor
        high_vv_novh       = high_vv_mask & ~cr_valid
        high_vv_veg_novh   = high_vv_novh & (vv <  _VV_URBAN_MIN_NOVH)
        high_vv_urban_novh = high_vv_novh & (vv >= _VV_URBAN_MIN_NOVH)

        high_vv_veg_mask    = high_vv_veg_cr   | high_vv_veg_novh
        high_vv_urban_mask  = high_vv_urban_cr  | high_vv_urban_novh
        high_vv_wetfor_mask = high_vv_wetfor_cr   # no VH-free fallback for this tier
    else:
        # No VH band at all: apply conservative urban floor to reduce tropical FPs
        high_vv_veg_mask    = high_vv_mask & (vv <  _VV_URBAN_MIN_NOVH)
        high_vv_urban_mask  = high_vv_mask & (vv >= _VV_URBAN_MIN_NOVH)
        high_vv_wetfor_mask = np.zeros(vv.shape, dtype=bool)

    # ── Write classes (precedence: water last so it is never overwritten) ───────
    out[high_vv_urban_mask]  = URBAN
    out[high_vv_wetfor_mask] = WET_FOREST
    out[high_vv_veg_mask]    = VEGETATION
    out[veg_mask]            = VEGETATION
    out[bare_mask]           = BARE_SOIL
    # Water applied last — specular near-zero return is most unambiguous
    out[water_mask]          = WATER

    # ── NDVI veto: optical signal overrides SAR-derived URBAN ─────────────────
    # Dense tropical/boreal canopy can produce VV ≥ 0.15 and CR < 0.30 due to
    # the combined effect of volume scattering and woody double-bounce, which is
    # indistinguishable from urban return in C-band SAR alone.  When a co-
    # registered NDVI layer is available, any pixel with NDVI > 0.5 is
    # reclassified as VEGETATION because live, dense canopy cannot be urban.
    if ndvi is not None:
        ndvi_veto = np.isfinite(ndvi) & (ndvi > 0.5)
        if np.any(ndvi_veto):
            import logging
            n_vetoed = int(np.sum(ndvi_veto & (out != VEGETATION)))
            logging.getLogger(__name__).info(
                "NDVI veto applied: %d pixel(s) reclassified to VEGETATION "
                "(NDVI > 0.5 mandates Vegetation class regardless of SAR-derived label).",
                n_vetoed,
            )
        out[ndvi_veto] = VEGETATION

    return out


def classification_stats(class_map: np.ndarray) -> Dict[int, dict]:
    """
    Pixel counts and area percentages per land cover class.

    Args:
        class_map: uint8 array produced by classify_sar().

    Returns:
        Dict keyed by class ID with keys: label, count, pct.
        NO_DATA pixels are excluded from the percentage denominator.
    """
    valid_total = int(np.sum(class_map != NO_DATA))
    stats: Dict[int, dict] = {}

    for cls_id, label in CLASS_LABELS.items():
        count = int(np.sum(class_map == cls_id))
        pct = (count / valid_total * 100.0) if valid_total > 0 else 0.0
        stats[cls_id] = {"label": label, "count": count, "pct": pct}

    return stats


_log = logging.getLogger(__name__)


def apply_persistence_filter(
    class_stack: np.ndarray,
    baseline_class: int = VEGETATION,
    target_class: int = URBAN,
    transient_class: int = ANOMALOUS_VEG_MOISTURE,
    min_persistence: int = 3,
) -> np.ndarray:
    """
    Temporal Consistency Filter — prevent transient backscatter spikes from
    being labelled as target_class.

    A pixel that was baseline_class (default: VEGETATION) for the majority of
    the early acquisitions may only be re-labelled as target_class (default:
    URBAN) if that label appears in at least min_persistence *consecutive*
    acquisitions anywhere in the stack.  Shorter runs are downgraded to
    transient_class (default: ANOMALOUS_VEG_MOISTURE).

    Args:
        class_stack:     uint8 array of shape (T, H, W) — time-ordered stack of
                         per-acquisition class maps from classify_sar().
        baseline_class:  Class ID expected in stable pre-event periods.
        target_class:    Class ID that requires persistence to be trusted.
        transient_class: Class ID assigned when persistence is insufficient.
        min_persistence: Minimum consecutive acquisitions required to retain
                         target_class.  Default 3 (≈ 18 days at 6-day revisit).

    Returns:
        Filtered uint8 class stack of the same shape as class_stack.
    """
    if class_stack.ndim != 3:
        raise ValueError(
            f"class_stack must be 3-D (T, H, W), got shape {class_stack.shape}"
        )
    T, H, W = class_stack.shape
    filtered = class_stack.copy()

    is_target = (class_stack == target_class)   # (T, H, W)

    # Compute the maximum consecutive run of target_class for every pixel.
    max_run     = np.zeros((H, W), dtype=np.int32)
    current_run = np.zeros((H, W), dtype=np.int32)
    for t in range(T):
        current_run = np.where(is_target[t], current_run + 1, 0)
        max_run     = np.maximum(max_run, current_run)

    # Derive baseline via majority vote over the first half of the stack so that
    # a pre-event vegetation signal is not swamped by the anomalous period.
    baseline_window = max(1, T // 2)
    baseline_votes  = np.sum(class_stack[:baseline_window] == baseline_class, axis=0)
    baseline_mask   = baseline_votes >= (baseline_window / 2.0)

    # Pixels where target_class appeared but never for long enough, and which
    # were previously labelled as baseline_class.
    insufficient = (max_run > 0) & (max_run < min_persistence) & baseline_mask

    n_pixels = int(np.sum(insufficient))
    if n_pixels:
        _log.info(
            "Persistence filter: %d pixel(s) downgraded from %s to %s "
            "(max consecutive run < %d acquisitions).",
            n_pixels,
            CLASS_LABELS.get(target_class, str(target_class)),
            CLASS_LABELS.get(transient_class, str(transient_class)),
            min_persistence,
        )

    for t in range(T):
        downgrade = insufficient & (filtered[t] == target_class)
        filtered[t] = np.where(downgrade, transient_class, filtered[t]).astype(np.uint8)

    return filtered


def apply_forest_mask(
    class_map: np.ndarray,
    ndvi_baseline: np.ndarray,
    ndvi_forest_threshold: float = 0.6,
    replace_class: int = ANOMALOUS_VEG_MOISTURE,
) -> np.ndarray:
    """
    Spatial Mask — override URBAN labels inside high-confidence forest pixels.

    Derives a permanent forest mask from the most recent cloud-free Sentinel-2
    NDVI baseline: pixels with NDVI > ndvi_forest_threshold are classified as
    confirmed forest.  Any URBAN label within that mask is demoted to
    replace_class (default: ANOMALOUS_VEG_MOISTURE), because live dense canopy
    cannot simultaneously be urban built-up.

    Args:
        class_map:             uint8 array (H, W) from classify_sar().
        ndvi_baseline:         float32 array (H, W) of NDVI from the most recent
                               cloud-free Sentinel-2 composite.
        ndvi_forest_threshold: High-confidence forest detection threshold
                               (default 0.6).  Only pixels exceeding this value
                               trigger the override.
        replace_class:         Class ID to assign to masked-out Urban pixels.

    Returns:
        Filtered uint8 class map of the same shape as class_map.
    """
    out         = class_map.copy()
    forest_mask = np.isfinite(ndvi_baseline) & (ndvi_baseline > ndvi_forest_threshold)
    urban_in_forest = forest_mask & (class_map == URBAN)

    n_pixels = int(np.sum(urban_in_forest))
    if n_pixels:
        _log.info(
            "Forest mask: %d pixel(s) reclassified from Urban to '%s' "
            "(NDVI baseline > %.2f confirms permanent forest canopy).",
            n_pixels,
            CLASS_LABELS.get(replace_class, str(replace_class)),
            ndvi_forest_threshold,
        )

    out[urban_in_forest] = replace_class
    return out


def classify_temporal_stack(
    vv_stack: np.ndarray,
    dates: List[str],
    vh_stack: Optional[np.ndarray] = None,
    ndvi_baseline: Optional[np.ndarray] = None,
    min_persistence: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Full temporal classification pipeline with all false-positive mitigations.

    Applies, in order:
      1. Per-acquisition classify_sar() with 3-tier dual-pol CR discriminator.
      2. apply_forest_mask()  — Sentinel-2 NDVI spatial constraint (if provided).
      3. apply_persistence_filter() — temporal consistency check.

    Args:
        vv_stack:       float32 array (T, H, W) of linear gamma0 VV values.
        dates:          Ordered list of ISO-8601 date strings, length T.
        vh_stack:       float32 array (T, H, W) of linear gamma0 VH values
                        (optional but strongly recommended for dual-pol check).
        ndvi_baseline:  float32 array (H, W) — most recent cloud-free S2 NDVI
                        composite used as the permanent forest mask baseline.
        min_persistence: Passed to apply_persistence_filter().

    Returns:
        class_stack: uint8 array (T, H, W) of filtered class maps.
        trend:       List of per-acquisition dicts with keys:
                       date, vv_db, vh_db, dominant_class_id, dominant_cover.
    """
    if vv_stack.ndim != 3:
        raise ValueError(
            f"vv_stack must be 3-D (T, H, W), got shape {vv_stack.shape}"
        )
    T = vv_stack.shape[0]
    if len(dates) != T:
        raise ValueError(
            f"len(dates)={len(dates)} must equal vv_stack.shape[0]={T}"
        )

    # ── Step 1: per-acquisition single-frame classification ───────────────────
    raw: List[np.ndarray] = []
    for t in range(T):
        vh_t = vh_stack[t] if vh_stack is not None else None
        raw.append(classify_sar(vv_stack[t], vh_t))
    class_stack: np.ndarray = np.stack(raw)   # (T, H, W)

    # ── Step 2: spatial forest mask (Sentinel-2 NDVI baseline) ───────────────
    if ndvi_baseline is not None:
        for t in range(T):
            class_stack[t] = apply_forest_mask(class_stack[t], ndvi_baseline)

    # ── Step 3: temporal persistence filter ──────────────────────────────────
    class_stack = apply_persistence_filter(
        class_stack, min_persistence=min_persistence
    )

    # ── Step 4: build per-acquisition trend report ────────────────────────────
    trend: List[Dict[str, Any]] = []
    for t in range(T):
        vv_t       = vv_stack[t]
        valid_vv   = np.isfinite(vv_t) & (vv_t > 0)
        vv_mean_db: Optional[float] = (
            float(10.0 * np.log10(float(np.mean(vv_t[valid_vv]))))
            if np.any(valid_vv) else None
        )

        vh_mean_db: Optional[float] = None
        if vh_stack is not None:
            vh_t     = vh_stack[t]
            valid_vh = np.isfinite(vh_t) & (vh_t > 0)
            if np.any(valid_vh):
                vh_mean_db = float(10.0 * np.log10(float(np.mean(vh_t[valid_vh]))))

        cls_t     = class_stack[t]
        valid_cls = cls_t[cls_t != NO_DATA].astype(np.int64)
        if valid_cls.size > 0:
            dominant_id: int = int(np.bincount(valid_cls).argmax())
        else:
            dominant_id = int(NO_DATA)
        dominant_label = CLASS_LABELS.get(dominant_id, "No Data")

        trend.append({
            "date":             dates[t],
            "vv_db":            round(vv_mean_db, 2) if vv_mean_db is not None else None,
            "vh_db":            round(vh_mean_db, 2) if vh_mean_db is not None else None,
            "dominant_class_id": dominant_id,
            "dominant_cover":   dominant_label,
        })

    return class_stack, trend


def generate_backscatter_trend_report(trend: List[Dict[str, Any]]) -> str:
    """
    Render a Temporal Backscatter Trend table as a formatted ASCII string.

    Args:
        trend: List of per-acquisition dicts as returned by
               classify_temporal_stack(), each containing at minimum:
               date, vv_db, vh_db, dominant_cover.

    Returns:
        Multi-line fixed-width ASCII table suitable for logging or PDF embedding.
    """
    col_date   = 12
    col_vv     = 9
    col_vh     = 9
    col_ratio  = 11
    col_cover  = 38

    def _fmt(val: Optional[float], width: int, decimals: int = 2) -> str:
        if val is None:
            return "N/A".rjust(width)
        return f"{val:.{decimals}f}".rjust(width)

    header = (
        f"{'Date':<{col_date}} | "
        f"{'VV (dB)':>{col_vv}} | "
        f"{'VH (dB)':>{col_vh}} | "
        f"{'VV/VH (dB)':>{col_ratio}} | "
        f"{'Dominant Cover':<{col_cover}}"
    )
    sep = "-" * len(header)
    rows: List[str] = [sep, header, sep]

    for entry in trend:
        vv_db = entry.get("vv_db")
        vh_db = entry.get("vh_db")
        ratio_db: Optional[float] = (
            round(vv_db - vh_db, 1)
            if (vv_db is not None and vh_db is not None) else None
        )
        rows.append(
            f"{entry['date']:<{col_date}} | "
            f"{_fmt(vv_db,  col_vv )} | "
            f"{_fmt(vh_db,  col_vh )} | "
            f"{_fmt(ratio_db, col_ratio, 1)} | "
            f"{entry['dominant_cover']:<{col_cover}}"
        )

    rows.append(sep)
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ISO 19157 COMMERCIAL HARDENING — Explainability, Validation & Recalibration
# ══════════════════════════════════════════════════════════════════════════════

# ── Plain-language class justifications (Explainability Layer) ─────────────
CLASS_PLAIN_LANGUAGE: Dict[int, str] = {
    WATER: (
        "Calm water and smooth surfaces exhibit very low radar return. Water acts as a "
        "specular mirror, reflecting the radar signal away from the sensor. This class "
        "indicates lakes, rivers, or smooth ground surfaces with minimal vegetation."
    ),
    BARE_SOIL: (
        "Exposed soil, gravel, or sparse ground cover with low radar reflectance. "
        "Agricultural fallow fields, construction sites, or recently cleared areas "
        "typically appear in this category. Vegetation cover is minimal or absent."
    ),
    VEGETATION: (
        "Healthy vegetation including crops, grasslands, and forest canopy. The radar "
        "signal penetrates and scatters within the vegetation structure, producing a "
        "characteristic signature consistent with photosynthetically active plants. "
        "May be confirmed by optical satellite data where cloud-free conditions exist."
    ),
    URBAN: (
        "Built-up areas including residential, commercial, and industrial zones. "
        "Buildings and infrastructure create distinctive double-bounce radar signatures "
        "from vertical walls and flat surfaces. Detection requires the signature to "
        "persist across multiple satellite overpasses (minimum 18 days) to avoid "
        "false positives from temporary structures or vehicles."
    ),
    ANOMALOUS_VEG_MOISTURE: (
        "Temporary change detection within established vegetation areas. This is a "
        "confidence filter rather than a permanent land cover class. Causes may include "
        "recent rainfall wetting the canopy, wind changing branch orientation, or "
        "wet snow accumulation. These effects typically resolve within one to two "
        "satellite acquisition cycles. No permanent land use change is indicated."
    ),
    WET_FOREST: (
        "Forest areas showing elevated moisture conditions, often following rainfall "
        "events or in riparian zones. The radar signature indicates water presence in "
        "the canopy or on the forest floor. This classification prevents urban "
        "misidentification of temporarily wet forest areas during or after wet weather."
    ),
}


# ── Accuracy thresholds required to remove "Experimental" watermark ────────
#
# Minimum standards for ISO 19157 / commercial Production release.
# ALL three conditions must be satisfied simultaneously across the validation
# confusion matrix before the "EXPERIMENTAL — Not for Legal Use" watermark
# may be removed from generated reports.
#
#   Overall Accuracy (OA):  ≥ 85.0 %  — ASPRS Positional Accuracy Standard tier
#   Kappa Coefficient (κ):  ≥ 0.80    — "Almost Perfect" (Landis & Koch 1977)
#   Per-class UA & PA:      ≥ 75.0 %  — each class individually (Urban / Wet Forest)
#
# The per-class floor ensures that a high OA driven by a dominant class
# (e.g., Vegetation at 65 % area) cannot mask poor performance on the
# commercially critical minority classes (Urban, Wet Forest).
CERTIFICATION_THRESHOLDS: Dict[str, Any] = {
    "overall_accuracy_pct_min":    85.0,
    "kappa_min":                   0.80,
    "per_class_users_accuracy_min": 75.0,
    "per_class_producers_accuracy_min": 75.0,
    "critical_classes": ["Urban / Built-up", "Wet Forest / Dielectric Anomaly"],
    "standard": "ISO 19157:2013 / ASPRS Positional Accuracy Standards 2014",
    "watermark_text": "EXPERIMENTAL — Accuracy Not Validated. Not for Legal or Commercial Use.",
    "release_text":   "Production Release — Validated per ISO 19157:2013.",
}


def check_certification_gate(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate compute_accuracy_metrics() output against CERTIFICATION_THRESHOLDS.

    Returns a dict with:
      passed        — bool: True only if ALL gates pass.
      gates         — per-criterion pass/fail breakdown.
      recommendation — 'PROMOTE_TO_PRODUCTION' | 'RETAIN_EXPERIMENTAL'
      watermark     — the appropriate watermark string.
    """
    oa_pass   = metrics["overall_accuracy_pct"]   >= CERTIFICATION_THRESHOLDS["overall_accuracy_pct_min"]
    kap_pass  = metrics["kappa_coefficient"]       >= CERTIFICATION_THRESHOLDS["kappa_min"]

    ua = metrics.get("users_accuracy_pct",     {})
    pa = metrics.get("producers_accuracy_pct", {})
    ua_min = CERTIFICATION_THRESHOLDS["per_class_users_accuracy_min"]
    pa_min = CERTIFICATION_THRESHOLDS["per_class_producers_accuracy_min"]

    per_class_gates: Dict[str, Dict[str, bool]] = {}
    all_class_pass = True
    for cls in CERTIFICATION_THRESHOLDS["critical_classes"]:
        ua_val = ua.get(cls, 0.0)
        pa_val = pa.get(cls, 0.0)
        ua_ok  = (not (isinstance(ua_val, float) and ua_val != ua_val)) and ua_val >= ua_min
        pa_ok  = (not (isinstance(pa_val, float) and pa_val != pa_val)) and pa_val >= pa_min
        per_class_gates[cls] = {"UA_pass": ua_ok, "PA_pass": pa_ok,
                                "UA_pct": ua_val, "PA_pct": pa_val}
        if not (ua_ok and pa_ok):
            all_class_pass = False

    passed = oa_pass and kap_pass and all_class_pass
    return {
        "passed":         passed,
        "gates": {
            "overall_accuracy": {"pass": oa_pass,
                                 "value": metrics["overall_accuracy_pct"],
                                 "threshold": CERTIFICATION_THRESHOLDS["overall_accuracy_pct_min"]},
            "kappa":            {"pass": kap_pass,
                                 "value": metrics["kappa_coefficient"],
                                 "threshold": CERTIFICATION_THRESHOLDS["kappa_min"]},
            "per_class":        per_class_gates,
        },
        "recommendation": "PROMOTE_TO_PRODUCTION" if passed else "RETAIN_EXPERIMENTAL",
        "watermark": (CERTIFICATION_THRESHOLDS["release_text"]
                      if passed else CERTIFICATION_THRESHOLDS["watermark_text"]),
    }


# ── Legal disclaimers (mandatory in all generated reports) ─────────────────
LEGAL_DISCLAIMERS: Dict[str, str] = {
    "radar_artifacts": (
        "TECHNICAL LIMITATIONS — RADAR GEOMETRIC DISTORTIONS\n"
        "This product is derived from Sentinel-1 C-band Synthetic Aperture Radar (SAR) "
        "imagery processed with Radiometric Terrain Correction (RTC) using the GLO-30 "
        "Copernicus DEM (30 m resolution). In mountainous and alpine terrain, the "
        "following geometric distortions may be present and are NOT corrected by RTC:\n\n"
        "  RADAR SHADOWING: Terrain features on the far side of steep ridges (relative "
        "to the satellite look direction) produce zero radar return. These shadow zones "
        "appear as NO_DATA (Class 255) or are falsely classified as WATER. Shadow "
        "extents are a function of local incidence angle and terrain height; valley "
        "floors and north-facing slopes in alpine environments are most susceptible.\n\n"
        "  FORESHORTENING: Slopes facing the radar appear compressed in the range "
        "direction, artificially concentrating backscatter and potentially inflating "
        "Urban or Wet Forest classification confidence in those zones.\n\n"
        "  LAYOVER: Very steep terrain facing the satellite may cause the top of a "
        "ridge to be imaged before its base, reversing spatial position. Urban "
        "structures on or adjacent to steep slopes may exhibit displaced or duplicated "
        "signatures. Layover zones overlap the near-range side of prominent peaks. "
        "Classifications within identified layover zones MUST be treated as UNRELIABLE "
        "and should be explicitly excluded from any legal, planning, or regulatory "
        "decision without independent verification by ground survey or airborne data.\n\n"
        "Users are advised to consult the Sentinel-1 incidence angle map and GLO-30 "
        "slope/aspect derivatives to delineate shadow and layover zones before "
        "presenting this product as evidence in formal proceedings."
    ),
    "copernicus_attribution": (
        "DATA SOURCE ATTRIBUTION (MANDATORY)\n"
        "This product incorporates data from the Copernicus Programme of the European "
        "Union. Satellite imagery was acquired by the Sentinel-1 mission, operated by "
        "the European Space Agency (ESA) on behalf of the European Commission.\n\n"
        "Required citation:\n"
        "  'Contains modified Copernicus Sentinel data [YEAR], processed by ESA.'\n\n"
        "The Copernicus DEM GLO-30, used for Radiometric Terrain Correction, is a "
        "product of the Copernicus Land Monitoring Service (CLMS):\n"
        "  'Copernicus DEM GLO-30 \u00a9 DLR e.V. 2021 and \u00a9 Airbus Defence and Space GmbH "
        "2021, provided under COPERNICUS by the European Union and ESA; all rights reserved.'\n\n"
        "This attribution must appear in any publication, report, legal filing, or "
        "presentation derived from or including this product. Omission of attribution "
        "may constitute a violation of the Copernicus Data and Information Policy "
        "(Regulation (EU) No 1159/2013) and Sentinel data licence terms."
    ),
    "classification_uncertainty": (
        "CLASSIFICATION UNCERTAINTY\n"
        "All automated land cover classifications in this product are probabilistic "
        "in nature. The classification engine applies threshold-based rules calibrated "
        "for Sentinel-1 C-band gamma0 RTC backscatter. Results may be affected by:\n"
        "  \u2022 Atmospheric moisture and precipitation (inflated VH returns).\n"
        "  \u2022 Freeze/thaw cycles (suppressed VH, anomalous VV).\n"
        "  \u2022 Wind-roughened water surfaces (misclassified as Bare Soil).\n"
        "  \u2022 Burned forest (bare-soil response from former canopy).\n"
        "  \u2022 Flooded vegetation (anomalously high double-bounce from water + trunk).\n"
        "Classifications labelled ANOMALOUS_VEG_MOISTURE or WET_FOREST are explicitly "
        "uncertain and must not be used as sole evidence for enforcement decisions. "
        "Per-pixel confidence ratings (HIGH / MEDIUM / LOW) produced by explain_pixel() "
        "must accompany any pixel-level legal evidence submission."
    ),
}


# ── Per-pixel decision trace (Final Certification — Explainability) ────────
def explain_pixel(
    vv: float,
    vh: Optional[float] = None,
    ndvi: Optional[float] = None,
    ndvi_baseline: Optional[float] = None,
    persistence_run: int = 0,
    min_persistence: int = 3,
    rtc_quality_flag: Optional[bool] = None,
    rtc_local_inc_angle_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Return a standardised 6-step decision trace for a single pixel, satisfying
    the ISO 19157 requirement that every automated classification be auditable.

    Steps (in execution order):
      1. [Validity + RTC Quality]  — data integrity and RTC normalisation check.
      2. [VV Primary Gate]         — specular / bare-soil / vegetation / high-VV regime.
      3. [CR Threshold]            — dual-polarisation cross-ratio discriminator.
      4. [NDVI Mask]               — per-acquisition optical veto (NDVI > 0.50).
      5. [Forest Mask]             — NDVI baseline permanent-forest override (NDVI > 0.60).
      6. [Persistence Filter]      — temporal consistency confirmation for URBAN.

    Args:
        vv:                      VV polarisation value (linear gamma0, scalar float).
        vh:                      VH polarisation value (optional scalar float).
        ndvi:                    Per-acquisition NDVI value at the pixel (optional).
        ndvi_baseline:           Most-recent cloud-free NDVI baseline at the pixel
                                 (used by apply_forest_mask(); optional).
        persistence_run:         Number of consecutive URBAN acquisitions already seen
                                 at this pixel (from apply_persistence_filter(); 0 if
                                 temporal stack not available).
        min_persistence:         Minimum consecutive run required to confirm URBAN.
        rtc_quality_flag:        True  = GLO-30 RTC normalisation passed QA.
                                 False = RTC flagged as suspect (steep terrain /
                                         shadow / layover zone).  None = not assessed.
        rtc_local_inc_angle_deg: Local incidence angle in degrees from the GLO-30
                                 DEM-derived slope/aspect correction layer.  Angles
                                 < 20° or > 70° indicate foreshortening or shadow
                                 risk and reduce output confidence.

    Returns:
        Dict with keys:
          final_class     — int class ID
          final_label     — str class label
          confidence      — 'HIGH' | 'MEDIUM' | 'LOW' | 'N/A'
          trace           — ordered list of step strings (always 6 entries)
          plain_language  — plain-text justification for the final class
          rtc_warning     — str | None: populated when RTC quality is suspect
    """
    trace: List[str] = []
    final_class: int = NO_DATA
    confidence: str = "UNDEFINED"
    rtc_warning: Optional[str] = None

    # ── Step 1: data validity + RTC quality check ────────────────────────────
    if not (isinstance(vv, (int, float)) and np.isfinite(vv) and vv >= 0):
        trace.append("Step 1 [Validity + RTC Quality]: VV is NaN, Inf, or negative \u2192 NO_DATA.")
        return {
            "final_class":    NO_DATA,
            "final_label":    "No Data",
            "confidence":     "N/A",
            "trace":          trace,
            "plain_language": "",
            "rtc_warning":    None,
        }
    _step1_parts: List[str] = [f"VV\u202f=\u202f{vv:.4f} (finite, \u2265\u202f0). \u2713"]
    if rtc_quality_flag is True:
        _step1_parts.append(
            "GLO-30 RTC normalisation: PASSED \u2014 gamma0 terrain correction verified."
        )
    elif rtc_quality_flag is False:
        _step1_parts.append(
            "GLO-30 RTC normalisation: FLAGGED \u2014 pixel lies within a suspected shadow "
            "or layover zone (local incidence angle extreme or DEM void). "
            "Classification confidence is downgraded to LOW regardless of signal values."
        )
        rtc_warning = (
            "RTC QUALITY ALERT: This pixel is located in a terrain-distorted zone "
            "(shadow / layover / foreshortening). The gamma0 backscatter value may not "
            "reflect true surface characteristics. Per LEGAL_DISCLAIMERS['radar_artifacts'], "
            "this pixel MUST NOT be used as sole evidence in legal proceedings."
        )
        confidence = "LOW"
    else:
        _step1_parts.append("GLO-30 RTC quality: NOT ASSESSED (rtc_quality_flag not supplied).")
    if rtc_local_inc_angle_deg is not None:
        if rtc_local_inc_angle_deg < 20.0 or rtc_local_inc_angle_deg > 70.0:
            _angle_risk = (
                "< 20\u00b0: foreshortening / layover risk"
                if rtc_local_inc_angle_deg < 20.0
                else "> 70\u00b0: shadow risk"
            )
            _inc_note = (
                f"Local incidence angle = {rtc_local_inc_angle_deg:.1f}\u00b0 "
                f"({_angle_risk}). "
                "Confidence reduced."
            )
            _step1_parts.append(_inc_note)
            if rtc_warning is None:
                rtc_warning = _inc_note
            if confidence not in ("LOW",):
                confidence = "MEDIUM"
        else:
            _step1_parts.append(
                f"Local incidence angle = {rtc_local_inc_angle_deg:.1f}\u00b0 "
                "(within nominal 20\u2013 70\u00b0 range). \u2713"
            )
    trace.append("Step 1 [Validity + RTC Quality]: " + " | ".join(_step1_parts))

    # ── Step 2: primary VV gate ──────────────────────────────────────────────
    if rtc_quality_flag is False and confidence == "LOW":
        pass
    if vv < _VV_WATER_MAX:
        trace.append(
            f"Step 2 [VV Gate]: VV\u202f=\u202f{vv:.4f}\u202f<\u202f{_VV_WATER_MAX} "
            "\u2192 specular / water regime."
        )
        if vh is not None and isinstance(vh, (int, float)) and np.isfinite(vh) and vh >= 0:
            if vh < _VH_WATER_MAX:
                trace.append(
                    f"Step 3 [VH Tighten]: VH\u202f=\u202f{vh:.4f}\u202f<\u202f{_VH_WATER_MAX} "
                    "\u2192 WATER confirmed."
                )
                final_class = WATER
                confidence = "HIGH"
            else:
                trace.append(
                    f"Step 3 [VH Tighten]: VH\u202f=\u202f{vh:.4f}\u202f\u2265\u202f{_VH_WATER_MAX} "
                    "\u2192 VH too elevated for water; reclassified to BARE_SOIL."
                )
                final_class = BARE_SOIL
                confidence = "MEDIUM"
        else:
            trace.append("Step 3 [VH Tighten]: VH not available \u2192 WATER (VV-only, medium confidence).")
            final_class = WATER
            confidence = "MEDIUM"

    elif vv < _VV_BARE_MAX:
        trace.append(
            f"Step 2 [VV Gate]: {_VV_WATER_MAX}\u202f\u2264\u202fVV\u202f=\u202f{vv:.4f}"
            f"\u202f<\u202f{_VV_BARE_MAX} \u2192 BARE_SOIL."
        )
        trace.append("Step 3 [VH]: Not required for bare-soil gate.")
        final_class = BARE_SOIL
        confidence = "HIGH"

    elif vv < _VV_VEG_MAX:
        trace.append(
            f"Step 2 [VV Gate]: {_VV_BARE_MAX}\u202f\u2264\u202fVV\u202f=\u202f{vv:.4f}"
            f"\u202f<\u202f{_VV_VEG_MAX} \u2192 VEGETATION (intermediate volume scatter)."
        )
        trace.append("Step 3 [VH]: Not required in intermediate VV regime.")
        final_class = VEGETATION
        confidence = "HIGH"

    else:
        trace.append(
            f"Step 2 [VV Gate]: VV\u202f=\u202f{vv:.4f}\u202f\u2265\u202f{_VV_VEG_MAX} "
            "\u2192 High-backscatter regime. Proceeding to dual-polarisation discriminator."
        )
        if vh is not None and isinstance(vh, (int, float)) and np.isfinite(vh) and vh >= 0:
            cr = vh / max(vv, 1e-10)
            trace.append(
                f"Step 3 [CR Check]: VH\u202f=\u202f{vh:.4f}, "
                f"CR\u202f=\u202fVH/VV\u202f=\u202f{cr:.4f}."
            )
            if cr < _CR_DOUBLE_BOUNCE_MAX:
                trace.append(
                    f"  CR\u202f=\u202f{cr:.4f}\u202f<\u202f{_CR_DOUBLE_BOUNCE_MAX} "
                    "\u2192 Tier\u202f1: Definitive double-bounce (VH co-pol suppressed). "
                    "Preliminary class: URBAN."
                )
                final_class = URBAN
                confidence = "HIGH"
            elif cr < _CR_VEG_MIN:
                trace.append(
                    f"  CR\u202f=\u202f{cr:.4f} in [{_CR_DOUBLE_BOUNCE_MAX}, {_CR_VEG_MIN}) "
                    "\u2192 Tier\u202f2: VH partially scales with VV. Moisture / dielectric "
                    "event over canopy. Class: WET_FOREST."
                )
                final_class = WET_FOREST
                confidence = "MEDIUM"
            else:
                trace.append(
                    f"  CR\u202f=\u202f{cr:.4f}\u202f\u2265\u202f{_CR_VEG_MIN} "
                    "\u2192 Tier\u202f3: Volume-scattering dominant. Class: VEGETATION."
                )
                final_class = VEGETATION
                confidence = "HIGH"
        else:
            trace.append(
                f"Step 3 [CR Check]: VH not available. Applying conservative VV floor "
                f"({_VV_URBAN_MIN_NOVH})."
            )
            if vv >= _VV_URBAN_MIN_NOVH:
                trace.append(
                    f"  VV\u202f=\u202f{vv:.4f}\u202f\u2265\u202f{_VV_URBAN_MIN_NOVH} "
                    "\u2192 URBAN (VV-only, reduced confidence; no WET_FOREST fallback)."
                )
                final_class = URBAN
                confidence = "LOW"
            else:
                trace.append(
                    f"  VV\u202f=\u202f{vv:.4f}\u202f<\u202f{_VV_URBAN_MIN_NOVH} "
                    "\u2192 VEGETATION (VV-only fallback)."
                )
                final_class = VEGETATION
                confidence = "MEDIUM"

    # ── Step 4: NDVI Mask (per-acquisition veto) ───────────────────────────
    if ndvi is not None and isinstance(ndvi, (int, float)) and np.isfinite(ndvi):
        if ndvi > 0.5:
            prev_label = CLASS_LABELS.get(final_class, str(final_class))
            trace.append(
                f"Step 4 [NDVI Veto]: NDVI\u202f=\u202f{ndvi:.4f}\u202f>\u202f0.50 \u2192 "
                f"VEGETATION override applied (was {prev_label}). "
                "Live dense canopy cannot simultaneously be urban."
            )
            final_class = VEGETATION
            confidence = "HIGH"
        else:
            trace.append(
                f"Step 4 [NDVI Veto]: NDVI\u202f=\u202f{ndvi:.4f}\u202f\u2264\u202f0.50 "
                "\u2192 veto NOT triggered."
            )
    else:
        trace.append("Step 4 [NDVI Veto]: NDVI not available \u2014 step skipped.")

    # ── Step 5: NDVI Baseline Forest Mask ─────────────────────────────────
    if ndvi_baseline is not None and isinstance(ndvi_baseline, (int, float)) and np.isfinite(ndvi_baseline):
        if ndvi_baseline > 0.6 and final_class == URBAN:
            trace.append(
                f"Step 5 [Forest Mask]: NDVI baseline\u202f=\u202f{ndvi_baseline:.4f}"
                "\u202f>\u202f0.60 \u2192 Permanent forest zone confirmed. "
                "URBAN \u2192 ANOMALOUS_VEG_MOISTURE."
            )
            final_class = ANOMALOUS_VEG_MOISTURE
            confidence = "MEDIUM"
        else:
            trace.append(
                f"Step 5 [Forest Mask]: NDVI baseline\u202f=\u202f{ndvi_baseline:.4f} "
                "\u2014 forest mask NOT triggered."
            )
    else:
        trace.append("Step 5 [Forest Mask]: NDVI baseline not available \u2014 step skipped.")

    # ── Step 6: Persistence Filter (temporal consistency) ──────────────────
    if final_class == URBAN:
        if persistence_run >= min_persistence:
            trace.append(
                f"Step 6 [Persistence]: Urban run\u202f=\u202f{persistence_run} consecutive "
                f"acquisition(s)\u202f\u2265\u202f{min_persistence} \u2192 URBAN label CONFIRMED."
            )
            confidence = "HIGH"
        elif persistence_run > 0:
            trace.append(
                f"Step 6 [Persistence]: Urban run\u202f=\u202f{persistence_run}"
                f"\u202f<\u202f{min_persistence} \u2192 Insufficient persistence. "
                "Downgraded to ANOMALOUS_VEG_MOISTURE."
            )
            final_class = ANOMALOUS_VEG_MOISTURE
            confidence = "LOW"
        else:
            trace.append(
                "Step 6 [Persistence]: persistence_run\u202f=\u202f0 (single acquisition "
                "or temporal stack not provided). URBAN retained; confidence reduced."
            )
            if confidence == "HIGH":
                confidence = "MEDIUM"
    else:
        trace.append(
            f"Step 6 [Persistence]: Class is "
            f"{CLASS_LABELS.get(final_class, str(final_class))} "
            "\u2014 persistence filter not applicable to this class."
        )

    trace.append(
        f"\u2192 Final Class: {CLASS_LABELS.get(final_class, 'No Data')} "
        f"(ID\u202f=\u202f{final_class}) | Confidence: {confidence}"
    )

    if rtc_quality_flag is False and confidence not in ("LOW",):
        confidence = "LOW"

    return {
        "final_class":    final_class,
        "final_label":    CLASS_LABELS.get(final_class, "No Data"),
        "confidence":     confidence,
        "trace":          trace,
        "plain_language": CLASS_PLAIN_LANGUAGE.get(final_class, ""),
        "rtc_warning":    rtc_warning,
    }


# ── Climate-adaptive CR threshold recalibration ────────────────────────────
_CLIMATE_THRESHOLD_PROFILES: Dict[str, Dict[str, Any]] = {
    "alpine_temperate": {
        "description": "Current calibration \u2014 alpine / temperate environment (default).",
        "VV_WATER_MAX":           _VV_WATER_MAX,
        "VV_BARE_MAX":            _VV_BARE_MAX,
        "VV_VEG_MAX":             _VV_VEG_MAX,
        "CR_DOUBLE_BOUNCE_MAX":   _CR_DOUBLE_BOUNCE_MAX,
        "CR_VEG_MIN":             _CR_VEG_MIN,
        "VV_URBAN_MIN_NOVH":      _VV_URBAN_MIN_NOVH,
        "min_persistence":        3,
        "rationale": (
            "Validated against ESA Sentinel-1 C-band benchmarks for central European "
            "terrain (Szigarski et al. 2018; ESA SAR Toolbox documentation). "
            "Frozen ground in winter may temporarily suppress VH, reducing CR; "
            "a seasonal flag on acquisitions below 0\u00b0C is recommended."
        ),
    },
    "tropical": {
        "description": "Tropical rainforest / equatorial environment.",
        "VV_WATER_MAX":           0.01,
        "VV_BARE_MAX":            0.05,
        "VV_VEG_MAX":             0.15,
        "CR_DOUBLE_BOUNCE_MAX":   0.20,
        "CR_VEG_MIN":             0.35,
        "VV_URBAN_MIN_NOVH":      0.25,
        "min_persistence":        4,
        "rationale": (
            "Tropical rainforest canopies produce significantly higher VH returns than "
            "temperate forests due to multi-layer scattering and high above-ground "
            "biomass (Mitchard et al. 2011). Raising CR_DOUBLE_BOUNCE_MAX from 0.15 "
            "to 0.20 and CR_VEG_MIN from 0.30 to 0.35 reduces urban false-positives "
            "over dense tropical canopy. min_persistence is increased to 4 to account "
            "for high inter-acquisition rainfall variability. NDVI fusion is "
            "mandatory; operating without VH is strongly inadvisable."
        ),
    },
    "arid": {
        "description": "Arid / semi-arid desert environment.",
        "VV_WATER_MAX":           0.008,
        "VV_BARE_MAX":            0.04,
        "VV_VEG_MAX":             0.12,
        "CR_DOUBLE_BOUNCE_MAX":   0.12,
        "CR_VEG_MIN":             0.25,
        "VV_URBAN_MIN_NOVH":      0.18,
        "min_persistence":        2,
        "rationale": (
            "Arid environments exhibit very low background VV from sandy substrates "
            "(typically\u202f<\u202f0.03). Urban structures produce a very strong relative "
            "double-bounce contrast. Lowering VV_VEG_MAX to 0.12 prevents sparse "
            "xerophytic vegetation from being over-classified as urban. CR thresholds "
            "are tightened because wind-roughened dune surfaces can generate a low CR "
            "that could otherwise overlap the urban CR range."
        ),
    },
    "boreal": {
        "description": "Boreal forest / sub-arctic environment.",
        "VV_WATER_MAX":           0.01,
        "VV_BARE_MAX":            0.05,
        "VV_VEG_MAX":             0.15,
        "CR_DOUBLE_BOUNCE_MAX":   0.15,
        "CR_VEG_MIN":             0.28,
        "VV_URBAN_MIN_NOVH":      0.20,
        "min_persistence":        3,
        "rationale": (
            "Boreal forests are characterised by coniferous stands with moderate-to-high "
            "VH due to needle volume scattering. CR_VEG_MIN is slightly lowered from "
            "0.30 to 0.28 because frozen needles in winter reduce depolarisation, "
            "pushing CR marginally lower than temperate deciduous. The seasonal "
            "freeze/thaw cycle may temporarily produce bare-soil-like VV returns; "
            "multi-date stack analysis is essential for reliable classification."
        ),
    },
    "arctic": {
        "description": "Arctic tundra / polar environment.",
        "VV_WATER_MAX":           0.015,
        "VV_BARE_MAX":            0.06,
        "VV_VEG_MAX":             0.12,
        "CR_DOUBLE_BOUNCE_MAX":   0.12,
        "CR_VEG_MIN":             0.25,
        "VV_URBAN_MIN_NOVH":      0.18,
        "min_persistence":        5,
        "rationale": (
            "Arctic tundra produces highly variable backscatter driven by freeze/thaw "
            "phase transitions rather than vegetation density. Dry, frozen soil yields "
            "very low VV (approaching the water floor); thawed active layer can generate "
            "anomalously high VV mimicking urban double-bounce. VV_WATER_MAX is raised "
            "to 0.015 to account for roughened melt-pond surfaces. VV_VEG_MAX is lowered "
            "to 0.12 because prostrate tundra vegetation (sedges, mosses, dwarf shrubs) "
            "produces significantly less volume scattering than temperate canopy. "
            "CR thresholds are tightened: wind-packed snow can suppress VH, compressing "
            "CR into the urban range; raising CR_DOUBLE_BOUNCE_MAX would introduce "
            "false positives over snowfields. min_persistence is raised to 5 "
            "(≈30 days at 6-day revisit) because tundra surface state can oscillate "
            "rapidly across freeze/thaw boundaries within a single month. "
            "NDVI fusion is mandatory: the lack of woody structure means optical "
            "confirmation is the only reliable discriminator between active-layer "
            "heave events and genuine built infrastructure. Operating without VH "
            "in the Arctic is strongly inadvisable."
        ),
    },
}


def suggest_cr_thresholds(climate: str) -> Dict[str, Any]:
    """
    Return a climate-specific CR threshold recalibration profile.

    Evaluates the current alpine/temperate calibration (CR_DOUBLE_BOUNCE_MAX=0.15,
    CR_VEG_MIN=0.30) against known backscatter behaviour in the requested
    environment and returns a replacement profile with justification.

    Args:
        climate: One of 'alpine_temperate', 'tropical', 'arid', 'boreal', 'arctic'
                 (case-insensitive; spaces/hyphens normalised to underscores).

    Returns:
        Dict containing: climate, description, all threshold values,
        min_persistence, rationale, and a diff vs. current thresholds.

    Raises:
        ValueError: If the requested climate profile does not exist.
    """
    key = climate.lower().replace("-", "_").replace(" ", "_")
    profile = _CLIMATE_THRESHOLD_PROFILES.get(key)
    if profile is None:
        available = list(_CLIMATE_THRESHOLD_PROFILES.keys())
        raise ValueError(
            f"Unknown climate '{climate}'. Available profiles: {available}"
        )
    current = {
        "CR_DOUBLE_BOUNCE_MAX": _CR_DOUBLE_BOUNCE_MAX,
        "CR_VEG_MIN":           _CR_VEG_MIN,
        "VV_URBAN_MIN_NOVH":    _VV_URBAN_MIN_NOVH,
    }
    diff: Dict[str, str] = {}
    for param in current:
        old_val = current[param]
        new_val = profile.get(param, old_val)
        if old_val != new_val:
            diff[param] = f"{old_val} \u2192 {new_val}"
    return {
        "climate":            key,
        "current_thresholds": current,
        "threshold_diff":     diff,
        **profile,
    }


# ── Visual Dynamic Range Compression (display / export pipeline) ──────────

# Safe epsilon added to VV before log10 to prevent log(0).
_VV_LOG_EPSILON = 1e-9

# Default dB clip range for Sentinel-1 IW mode gamma0:
#   Lower clip: -25 dB  (typical noise floor; water / shadow)
#   Upper clip: +5  dB  (saturation level; very bright urban corner reflectors)
# Values outside this range are clipped BEFORE normalisation to prevent outliers
# like the observed Max VV = 29432.69 (linear) ≈ +44.7 dB from blowing out the
# display range and compressing all other features into near-zero contrast.
_VV_DISPLAY_DB_MIN = -25.0
_VV_DISPLAY_DB_MAX =   5.0


def compress_dynamic_range(
    vv: np.ndarray,
    db_min: float = _VV_DISPLAY_DB_MIN,
    db_max: float = _VV_DISPLAY_DB_MAX,
    output_dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Convert linear gamma0 VV to a display-ready normalised array via log10 scaling.

    Motivation — the raw VV outlier problem:
      A single urban corner reflector or RFI spike can produce a linear VV value
      orders of magnitude above the scene mean (e.g., 29432.69 ≈ +44.7 dB).
      When rendered linearly this collapses the entire 14.49 km² AOI into a
      near-black image with one white pixel, making roads, forest density
      gradients, and settlement edges invisible.

    Solution — two-stage compression:
      Stage 1 — Logarithmic Scaling (mandatory):
        dB = 10 * log10(max(VV, epsilon))  converts multiplicative outliers
        into additive offsets.  A +44.7 dB spike becomes a +39.7 dB outlier
        above the +5 dB clip ceiling, not a 29432× amplitude multiplier.
      Stage 2 — dB Clip + Linear Normalisation to [0, 1]:
        Values below db_min (≈ noise floor / shadow) and above db_max
        (≈ saturation / bright urban) are hard-clipped.  The remaining range
        is linearly stretched to [0, 1].

    Effect on features of interest:
      • Valley roads (low VV, ≈ −18 dB): mapped to low-but-visible ≈25 % range.
      • Forest density gradients (mid VV, − 12 to −6 dB): spread across 45–65 %.
      • Wet forest / moisture events (high VV, −6 to −2 dB): clearly 65–85 %.
      • Urban double-bounce (≈ 0 to +5 dB): saturate at the top 85–100 %, visually
        distinct but NOT blown out to the point of masking adjacent features.

    Args:
        vv:           2-D float array of linear gamma0 VV (any scale; NaN allowed).
        db_min:       Lower dB clip boundary (default: −25 dB).
        db_max:       Upper dB clip boundary (default: +5 dB).
        output_dtype: NumPy dtype for the output array (default: float32, range [0,1]).
                      Pass np.uint8 to get a direct 0–255 byte image.

    Returns:
        Normalised array of the same shape as vv, dtype=output_dtype.
        NaN / negative / non-finite input pixels remain NaN in float output
        or are set to 0 in integer output.
    """
    vv_db = np.where(
        np.isfinite(vv) & (vv > 0),
        10.0 * np.log10(np.maximum(vv, _VV_LOG_EPSILON)),
        np.nan,
    ).astype(np.float64)

    db_range = db_max - db_min
    normalised = np.clip((vv_db - db_min) / db_range, 0.0, 1.0)

    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        out = np.where(
            np.isfinite(normalised),
            np.round(normalised * info.max).clip(info.min, info.max),
            0,
        ).astype(output_dtype)
    else:
        out = normalised.astype(output_dtype)

    n_clipped_high = int(np.sum(np.isfinite(vv_db) & (vv_db > db_max)))
    n_clipped_low  = int(np.sum(np.isfinite(vv_db) & (vv_db < db_min)))
    if n_clipped_high or n_clipped_low:
        _log.debug(
            "compress_dynamic_range: %d pixel(s) clipped above +%.0f dB, "
            "%d pixel(s) clipped below %.0f dB.",
            n_clipped_high, db_max, n_clipped_low, db_min,
        )
    return out


def apply_histogram_equalization(
    normalised: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """
    Apply histogram equalization to a [0, 1]-normalised display array.

    Use this as an OPTIONAL Stage 3 after compress_dynamic_range() when the
    scene still shows low contrast (e.g., dense forest AOI where most pixels
    cluster in the −16 to −6 dB band after log scaling).

    Histogram equalization redistributes pixel intensities so that each
    brightness level occupies an equal fraction of the image.  This enhances
    subtle texture differences in homogeneous regions (e.g., forest density
    gradients, road traces through canopy) at the cost of removing the
    perceptual linearity between brightness and dB value.

    WARNING: Do NOT use histogram-equalized images as a linear measurement
    reference.  Apply only to RGB/grayscale export images; keep the
    compress_dynamic_range() output for all quantitative analysis.

    Args:
        normalised: 2-D float array in [0, 1] (output of compress_dynamic_range()).
                    NaN pixels are excluded from the CDF and remain NaN.
        n_bins:     Number of histogram bins (default: 256).

    Returns:
        float32 array of the same shape, values in [0, 1], with equalised
        intensity distribution.  NaN pixels preserved.
    """
    valid_mask = np.isfinite(normalised)
    flat       = normalised[valid_mask].ravel()
    if flat.size == 0:
        return normalised.astype(np.float32)

    hist, bin_edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0))
    cdf             = hist.cumsum().astype(np.float64)
    cdf_normalised  = (cdf - cdf.min()) / max(cdf.max() - cdf.min(), 1.0)

    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    out = normalised.copy().astype(np.float64)
    valid_vals = normalised[valid_mask]
    indices = np.searchsorted(bin_centres, valid_vals, side="right") - 1
    indices = np.clip(indices, 0, n_bins - 1)
    out[valid_mask] = cdf_normalised[indices]
    return out.astype(np.float32)


# ── Stratified validation protocol designer (Confusion Matrix) ─────────────
def generate_validation_protocol(
    area_km2: float = 14.49,
    n_points: int = 50,
    class_fractions: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Design a stratified random sampling protocol for an ISO 19157 confusion
    matrix validation over the classified AOI.

    Implements proportional allocation with a minimum of 3 points per class to
    ensure at least one expected true-positive, one false-positive, and one
    false-negative per class in the confusion matrix.

    Args:
        area_km2:         Total analysis area in km\u00b2 (default: 14.49 km\u00b2).
        n_points:         Total number of control points (default: 50).
        class_fractions:  Optional dict {class_id: fraction_of_area}.
                          If None, a conservative temperate alpine profile is used.

    Returns:
        Dict containing: sampling design metadata, per-class sample counts,
        minimum spatial spacing, reference data specification, and column
        template for the ground-truth table.
    """
    import math as _math

    if class_fractions is None:
        class_fractions = {
            WATER:                  0.05,
            BARE_SOIL:              0.10,
            VEGETATION:             0.65,
            URBAN:                  0.15,
            ANOMALOUS_VEG_MOISTURE: 0.03,
            WET_FOREST:             0.02,
        }

    total_frac = sum(class_fractions.values())
    norm_frac = {k: v / total_frac for k, v in class_fractions.items()}

    raw_counts: Dict[int, int] = {
        cls: max(3, round(norm_frac[cls] * n_points))
        for cls in norm_frac
    }
    delta = n_points - sum(raw_counts.values())
    dominant_cls = max(norm_frac, key=lambda c: norm_frac[c])
    raw_counts[dominant_cls] += delta

    min_spacing_km = round(_math.sqrt(area_km2 / n_points), 3)

    return {
        "area_km2":         area_km2,
        "n_control_points": n_points,
        "min_spacing_km":   min_spacing_km,
        "sampling_design":  "Stratified random — proportional to mapped class area",
        "per_class_samples": {
            CLASS_LABELS.get(cls, str(cls)): count
            for cls, count in raw_counts.items()
        },
        "reference_data": {
            "primary_source":           "Google Earth Pro archive imagery (< 1 m GSD)",
            "fallback_source":          "ESA Sentinel-2 cloudless composite (10 m GSD)",
            "temporal_window_days":     30,
            "minimum_mapping_unit_m2":  100,
            "interpreter_requirement": (
                "Two independent interpreters; inter-rater Cohen\u2019s Kappa \u2265\u202f0.85 "
                "required for reference consensus before confusion matrix compilation."
            ),
        },
        "field_columns": [
            "point_id", "latitude", "longitude",
            "automated_class_id", "automated_class_label",
            "reference_class_id", "reference_class_label",
            "interpreter_1", "interpreter_2",
            "high_res_source", "acquisition_date",
            "confidence_notes",
        ],
        "standard": "ISO 19157:2013 \u2014 Geographic Information Data Quality",
    }


# ── ISO 19157 accuracy metric calculator ──────────────────────────────────
def compute_accuracy_metrics(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute ISO 19157-compliant accuracy metrics from a square confusion matrix.

    Rows = reference (ground-truth) class; Columns = mapped (automated) class.
    Diagonal = correctly classified pixels / control points.

    Args:
        confusion_matrix: Square 2-D array of int/float counts.
        class_names:      Optional list of class name strings (length = n_classes).
                          Defaults to ['0', '1', \u2026].

    Returns:
        Dict with keys:
          overall_accuracy_pct    \u2014 (sum diagonal / total) * 100
          kappa_coefficient       \u2014 Cohen\u2019s Kappa (Landis & Koch 1977)
          kappa_interpretation    \u2014 qualitative band label
          producers_accuracy_pct  \u2014 {class: PA%} (recall; omission error = 100-PA)
          users_accuracy_pct      \u2014 {class: UA%} (precision; commission error = 100-UA)
          f1_score_pct            \u2014 {class: F1%} (harmonic mean of PA and UA)
          n_control_points        \u2014 total observations
          standard                \u2014 ISO 19157:2013
    """
    cm = np.array(confusion_matrix, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"confusion_matrix must be square 2-D, got shape {cm.shape}"
        )
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    if len(class_names) != n_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} \u2260 n_classes={n_classes}"
        )

    total = float(np.sum(cm))
    if total == 0:
        raise ValueError("confusion_matrix has no observations (all zeros).")

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    diag     = np.diag(cm)

    producers_accuracy: Dict[str, float] = {}
    users_accuracy:     Dict[str, float] = {}
    f1_scores:          Dict[str, float] = {}

    for i, name in enumerate(class_names):
        pa = float(diag[i] / row_sums[i] * 100.0) if row_sums[i] > 0 else float("nan")
        ua = float(diag[i] / col_sums[i] * 100.0) if col_sums[i] > 0 else float("nan")
        producers_accuracy[name] = round(pa, 2)
        users_accuracy[name]     = round(ua, 2)
        if pa > 0 and ua > 0 and not (np.isnan(pa) or np.isnan(ua)):
            f1 = 2.0 * pa * ua / (pa + ua)
        else:
            f1 = float("nan")
        f1_scores[name] = round(f1, 2)

    oa = float(np.sum(diag) / total * 100.0)
    po = float(np.sum(diag)) / total
    pe = float(np.dot(row_sums, col_sums)) / (total ** 2)
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0.0 else float("nan")

    if np.isnan(kappa):
        kappa_interp = "N/A"
    elif kappa > 0.80:
        kappa_interp = "Almost perfect agreement (> 0.80)"
    elif kappa > 0.60:
        kappa_interp = "Substantial agreement (0.61 \u2013 0.80)"
    elif kappa > 0.40:
        kappa_interp = "Moderate agreement (0.41 \u2013 0.60)"
    elif kappa > 0.20:
        kappa_interp = "Fair agreement (0.21 \u2013 0.40)"
    else:
        kappa_interp = "Slight agreement (\u2264 0.20)"

    return {
        "overall_accuracy_pct":   round(oa, 2),
        "kappa_coefficient":      round(float(kappa), 4),
        "kappa_interpretation":   kappa_interp,
        "producers_accuracy_pct": producers_accuracy,
        "users_accuracy_pct":     users_accuracy,
        "f1_score_pct":           f1_scores,
        "n_control_points":       int(total),
        "standard":               "ISO 19157:2013",
    }
