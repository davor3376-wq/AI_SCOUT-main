# Project Gaia — SAR Classification Pipeline
# Formal Certification Report
# Job Reference: JOB-02C2C063

---

| Field              | Value                                                      |
|--------------------|------------------------------------------------------------|
| **Report ID**      | CERT-JOB-02C2C063-v1.0                                     |
| **Pipeline**       | AI Scout · SAR Land Cover Classifier v2 (3-Tier CR + GLO-30 RTC) |
| **AOI**            | 14.49 km² Alpine / Temperate Zone                          |
| **Standard**       | ISO 19157:2013 — Geographic Information Data Quality       |
| **Prepared by**    | [Auditor Name / Organisation]                              |
| **Review Date**    | [YYYY-MM-DD]                                               |
| **Status**         | ☐ EXPERIMENTAL (Pending Validation)  ☐ PRODUCTION RELEASE  |

---

## 1. Statistical Validation Protocol

### 1.1 Stratified Random Sampling Design

**Function:** `generate_validation_protocol(area_km2=14.49, n_points=100)`

A total of **100 control points** are allocated across the AOI using proportional
stratified random sampling (ISO 19157:2013 §6.4). Proportional allocation ensures
minority classes (Water, Anomalous Veg Moisture, Wet Forest) receive the ISO-mandated
minimum of 3 points to prevent degenerate per-class confusion matrix rows.

| Class ID | Class Label                        | Mapped Area (%) | Control Points |
|----------|------------------------------------|-----------------|----------------|
| 0        | Water / Smooth Surface             | 5               | 5              |
| 1        | Bare Soil / Low Vegetation         | 10              | 10             |
| 2        | Vegetation / Forest                | 65              | 65             |
| 3        | Urban / Built-up                   | 15              | 15             |
| 4        | Anomalous Vegetation / Moisture    | 3               | 3              |
| 5        | Wet Forest / Dielectric Anomaly    | 2               | 3 (floor)      |
| **Total**|                                    | **100**         | **101***       |

> *One extra point added to the minimum-floor classes; dominant class (Vegetation) adjusted downward by 1 to maintain n=100 net total. Final allocation computed at runtime by `generate_validation_protocol()`.

**Minimum spatial spacing:** √(14.49 / 100) ≈ **0.380 km** between any two points.

**Reference Data Specification:**

| Parameter                   | Requirement                                                   |
|-----------------------------|---------------------------------------------------------------|
| Primary source              | Google Earth Pro archive imagery (< 1 m GSD)                  |
| Fallback source             | ESA Sentinel-2 cloudless composite (10 m GSD)                 |
| Temporal window             | ± 30 days from SAR acquisition date                           |
| Minimum mapping unit        | 100 m²                                                        |
| Interpreter requirement     | 2 independent interpreters; inter-rater Cohen's κ ≥ 0.85 required before matrix compilation |
| Field columns               | point_id, lat, lon, automated_class_id, automated_class_label, reference_class_id, reference_class_label, interpreter_1, interpreter_2, high_res_source, acquisition_date, confidence_notes |

---

### 1.2 Confusion Matrix Structure

Rows = reference (ground-truth) class. Columns = mapped (automated) class.
Diagonal = correctly classified control points.

```
                  | Water | Bare | Veg | Urban | AVM | WetF | ROW TOTAL
------------------+-------+------+-----+-------+-----+------+----------
Water             |  r00  | r01  | r02 |  r03  | r04 | r05  |   n0.
Bare Soil         |  r10  |  …   |     |       |     |      |   n1.
Vegetation        |       |      |     |       |     |      |   n2.
Urban             |       |      |     |       |     |      |   n3.
Anom. Veg. Moist. |       |      |     |       |     |      |   n4.
Wet Forest        |       |      |     |       |     |      |   n5.
------------------+-------+------+-----+-------+-----+------+----------
COL TOTAL         |  n.0  | n.1  | n.2 |  n.3  | n.4 | n.5  |    N
```

**Computed via:** `compute_accuracy_metrics(confusion_matrix, class_names=[...])`

---

### 1.3 Accuracy Metrics

#### Overall Accuracy (OA)

```
OA = (Σ diagonal) / N × 100
```

The fraction of all control points correctly classified.

#### Cohen's Kappa Coefficient (κ)

```
κ = (Po − Pe) / (1 − Pe)

where:
  Po = OA / 100  (observed proportion correct)
  Pe = Σ (row_sum_i × col_sum_i) / N²  (chance-agreement proportion)
```

Kappa removes the probability of chance agreement, making it the authoritative
metric for imbalanced multi-class maps. Interpretation (Landis & Koch 1977):

| κ range    | Agreement band      |
|------------|---------------------|
| > 0.80     | Almost Perfect      |
| 0.61–0.80  | Substantial         |
| 0.41–0.60  | Moderate            |
| 0.21–0.40  | Fair                |
| ≤ 0.20     | Slight              |

#### Producer's Accuracy (PA) — Recall

```
PA_i = diagonal[i] / row_sum[i] × 100
```

Answers: *"Of all reference pixels in class i, what fraction did the classifier find?"*
Low PA = high omission error (the classifier misses real occurrences of this class).

#### User's Accuracy (UA) — Precision

```
UA_i = diagonal[i] / col_sum[i] × 100
```

Answers: *"Of all pixels the classifier labelled as class i, what fraction are actually class i?"*
Low UA = high commission error (the classifier over-labels this class).

**Critical classes for commercial release:** Urban / Built-up AND Wet Forest / Dielectric Anomaly.
Both must individually satisfy the per-class thresholds below.

---

### 1.4 Accuracy Thresholds — Production Promotion Gate

**Implemented in:** `CERTIFICATION_THRESHOLDS` · `check_certification_gate(metrics)`

All three gates must pass **simultaneously**:

| Gate                          | Minimum Required | Rationale                                                                       |
|-------------------------------|------------------|---------------------------------------------------------------------------------|
| Overall Accuracy (OA)         | **≥ 85.0 %**     | ASPRS Positional Accuracy Standards (2014) Tier 3; threshold for legal evidence |
| Kappa Coefficient (κ)         | **≥ 0.80**       | "Almost Perfect" band; eliminates chance-agreement inflation                     |
| Urban UA & PA (each)          | **≥ 75.0 %**     | Commercial-critical class; prevents false enforcement actions                   |
| Wet Forest UA & PA (each)     | **≥ 75.0 %**     | High-confusion class with Urban; must not collapse into it                      |

> Failure of any single gate → **RETAIN_EXPERIMENTAL** watermark.
> All gates pass → **PROMOTE_TO_PRODUCTION** watermark.
> Text constants are defined in `CERTIFICATION_THRESHOLDS["watermark_text"]` /
> `["release_text"]` and must be embedded in PDF footers by the ReportLab
> pipeline (`app/reporting`).

---

## 2. Visual Processing Optimisation

### 2.1 Root Cause — Max VV = 29432.69 (linear)

Converting to dB: `10 × log10(29432.69) ≈ +44.7 dB`

The nominal Sentinel-1 IW mode gamma0 RTC backscatter range is −25 to +5 dB.
The +44.7 dB outlier is a **single urban corner reflector or RFI spike**, not a
scene-representative value. Linear rendering compresses the entire 14.49 km² AOI
into a 0–29432 scale where:

- Valley roads (≈ 0.013 linear, −18.9 dB) render at 0.00004 of full scale → **invisible**
- Forest density gradients (0.025–0.10 linear, −16 to −10 dB) → **invisible**
- All meaningful texture → **invisible**

### 2.2 Two-Stage Compression Pipeline

**Implemented in:** `compress_dynamic_range()` + `apply_histogram_equalization()`

```
LINEAR VV  →  [Stage 1: Log10 Scale]  →  dB VV  →  [Stage 2: Clip + Normalise]  →  [0,1]
                                                   (optional Stage 3: Hist. EQ)  →  [0,1] equalised
```

**Stage 1 — Logarithmic Scaling (mandatory):**

```python
vv_db = 10 * log10(max(VV, 1e-9))
```

- Max outlier: 29432.69 → +44.7 dB (an additive offset, not a ×29432 amplitude multiplier)
- Noise floor: 0.003 → −25.2 dB

**Stage 2 — dB Clip + Linear Normalisation:**

```
db_min = −25 dB  (noise floor / shadow)
db_max =   +5 dB (bright urban saturation)
normalised = clip((vv_db − db_min) / (db_max − db_min), 0, 1)
```

**Display mapping after compression:**

| Feature                           | Linear VV range        | dB range       | Normalised brightness |
|-----------------------------------|------------------------|----------------|-----------------------|
| Shadow / NO_DATA                  | < 0.001                | < −30 dB       | 0 % (clipped floor)   |
| Calm water                        | 0.001–0.010            | −30 to −20 dB  | 0–21 %                |
| Valley roads                      | 0.010–0.020            | −20 to −17 dB  | 21–27 %               |
| Low vegetation / bare soil        | 0.010–0.050            | −20 to −13 dB  | 21–40 %               |
| Forest / moderate backscatter     | 0.050–0.100            | −13 to −10 dB  | 40–50 %               |
| Dense forest / wet canopy         | 0.100–0.316            | −10 to −5 dB   | 50–67 %               |
| Wet forest / moisture events      | 0.316–0.562            | −5 to −2.5 dB  | 67–75 %               |
| Urban double-bounce               | 0.562–3.162            | −2.5 to +5 dB  | 75–100 % (visible top) |
| RFI / corner reflector outlier    | > 3.162 (e.g., 29432)  | > +5 dB        | 100 % (clipped ceiling)|

**Stage 3 — Histogram Equalisation (optional):**

Apply `apply_histogram_equalization()` **only** to greyscale/RGB export images
when forest-dominated AOIs still show insufficient texture contrast after log scaling.
**Never** apply to the quantitative analysis raster — it destroys the dB linearity.

### 2.3 Usage in the Pipeline

```python
from app.analytics.sar_classifier import compress_dynamic_range, apply_histogram_equalization

display = compress_dynamic_range(vv_array, db_min=-25, db_max=5, output_dtype=np.uint8)

display_eq = apply_histogram_equalization(
    compress_dynamic_range(vv_array, output_dtype=np.float32)
)
```

---

## 3. Legal & Disclaimer Hardening

### 3.1 Technical Limitations — Radar Geometric Distortions

**Implemented in:** `LEGAL_DISCLAIMERS["radar_artifacts"]`

> **TECHNICAL LIMITATIONS — RADAR GEOMETRIC DISTORTIONS**
>
> This product is derived from Sentinel-1 C-band SAR imagery processed with
> Radiometric Terrain Correction (RTC) using the GLO-30 Copernicus DEM (30 m resolution).
> In mountainous and alpine terrain the following geometric distortions may be present
> and are **NOT corrected by RTC**:
>
> **RADAR SHADOWING:** Terrain features on the far side of steep ridges produce zero
> radar return. These zones appear as NO_DATA (Class 255) or are falsely classified as
> WATER. Valley floors and north-facing slopes in alpine environments are most susceptible.
>
> **FORESHORTENING:** Slopes facing the radar appear compressed in the range direction,
> artificially concentrating backscatter and potentially inflating Urban or Wet Forest
> classification confidence.
>
> **LAYOVER:** Very steep terrain facing the satellite may cause the ridge top to be
> imaged before its base, reversing spatial position. Urban structures on or adjacent
> to steep slopes may exhibit displaced or duplicated signatures. **Classifications
> within identified layover zones MUST be treated as UNRELIABLE** and must be excluded
> from legal, planning, or regulatory decisions without independent verification.
>
> Users must consult the Sentinel-1 incidence angle map and GLO-30 slope/aspect derivatives
> to delineate shadow and layover zones before presenting this product as evidence.

### 3.2 Mandatory ESA / Copernicus Attribution

**Implemented in:** `LEGAL_DISCLAIMERS["copernicus_attribution"]`

Must appear verbatim in every report footer:

> "Contains modified Copernicus Sentinel data [YEAR], processed by ESA."

And for DEM sourcing:

> "Copernicus DEM GLO-30 © DLR e.V. 2021 and © Airbus Defence and Space GmbH 2021,
> provided under COPERNICUS by the European Union and ESA; all rights reserved."

Omission may violate Regulation (EU) No 1159/2013 and the Sentinel data licence.

### 3.3 Classification Uncertainty Disclaimer

**Implemented in:** `LEGAL_DISCLAIMERS["classification_uncertainty"]`

Must accompany any pixel-level evidence submitted to legal proceedings. Requires the
per-pixel confidence rating (HIGH / MEDIUM / LOW) from `explain_pixel()` as an annex.

---

## 4. Explainability Trace — Standardised 6-Step Protocol

**Implemented in:** `explain_pixel()` with new `rtc_quality_flag` and `rtc_local_inc_angle_deg` parameters.

Every pixel classified by the pipeline must be auditable via:

```python
trace = explain_pixel(
    vv=0.18, vh=0.022, ndvi=0.42, ndvi_baseline=0.55,
    persistence_run=4, min_persistence=3,
    rtc_quality_flag=True, rtc_local_inc_angle_deg=38.2
)
```

### Step Definition Table

| Step | Label                     | Decision Gate                                               | Confidence Effect                  |
|------|---------------------------|-------------------------------------------------------------|------------------------------------|
| 1    | **Validity + RTC Quality**| VV finite & ≥ 0; GLO-30 RTC QA flag; local incidence angle | FLAGGED RTC → forces LOW           |
| 2    | **VV Primary Gate**       | Specular (< 0.01) / Bare (< 0.05) / Veg (< 0.15) / High-VV | Unambiguous regimes → HIGH         |
| 3    | **CR Threshold**          | CR = VH/VV: < 0.15 → Urban; [0.15, 0.30) → Wet Forest; ≥ 0.30 → Vegetation | Tier 2 → MEDIUM |
| 4    | **NDVI Mask**             | Per-acquisition NDVI > 0.50 → force VEGETATION override     | Override → HIGH                    |
| 5    | **Forest Mask**           | NDVI baseline > 0.60 AND URBAN → ANOMALOUS_VEG_MOISTURE     | Override → MEDIUM                  |
| 6    | **Persistence Filter**    | Urban run ≥ min_persistence (3) → CONFIRMED; < → downgrade  | Confirmed → HIGH; Downgraded → LOW |

### Output Contract

```python
{
    "final_class":    3,                          # int class ID
    "final_label":    "Urban / Built-up",
    "confidence":     "HIGH",                     # HIGH | MEDIUM | LOW | N/A
    "trace": [                                    # always 6+ entries
        "Step 1 [Validity + RTC Quality]: ...",
        "Step 2 [VV Gate]: ...",
        "Step 3 [CR Check]: ...",
        "Step 4 [NDVI Veto]: ...",
        "Step 5 [Forest Mask]: ...",
        "Step 6 [Persistence]: ...",
        "→ Final Class: Urban / Built-up (ID = 3) | Confidence: HIGH"
    ],
    "plain_language": "Persistent high-intensity double-bounce return ...",
    "rtc_warning":    None                        # str if RTC suspect, else None
}
```

**Legal use requirement:** Any pixel submitted as evidence must include its full `trace`
list and the `rtc_warning` field. A non-None `rtc_warning` **disqualifies** the pixel
from sole-evidence status (see §3.1).

---

## 5. Multi-Biome Regional Configuration Profiles

**Implemented in:** `_CLIMATE_THRESHOLD_PROFILES` · `suggest_cr_thresholds(climate)`

Five calibrated profiles. Call `suggest_cr_thresholds("tropical")` to retrieve any profile
with a full diff against the current alpine/temperate defaults.

| Profile             | `CR_DOUBLE_BOUNCE_MAX` | `CR_VEG_MIN` | `VV_VEG_MAX` | `min_persistence` | Key Driver                                    |
|---------------------|------------------------|--------------|--------------|-------------------|-----------------------------------------------|
| `alpine_temperate`  | 0.15 (default)         | 0.30         | 0.15         | 3                 | Validated ESA C-band benchmark                |
| `tropical`          | **0.20** ↑             | **0.35** ↑   | 0.15         | **4** ↑           | Multi-layer canopy inflates VH; high rainfall |
| `arid`              | **0.12** ↓             | **0.25** ↓   | **0.12** ↓   | **2** ↓           | Low background VV; strong urban contrast      |
| `boreal`            | 0.15                   | **0.28** ↓   | 0.15         | 3                 | Frozen-needle VH suppression in winter        |
| `arctic`            | **0.12** ↓             | **0.25** ↓   | **0.12** ↓   | **5** ↑           | Freeze/thaw phase transitions; snow VH issues |

### Tropical Profile Rationale

Raised `CR_VEG_MIN` (0.30 → 0.35) because multi-layer tropical canopy produces
above-average VH via multi-path volume scattering and high above-ground biomass
(Mitchard et al. 2011). NDVI fusion is **mandatory**; VH-absent operation is
strongly inadvisable.

### Arid Profile Rationale

Sandy substrate background VV < 0.03 creates strong relative contrast for urban
double-bounce. Both CR thresholds are tightened to prevent wind-roughened dune
surfaces (low CR) from overlapping the urban CR range. Reduced `min_persistence`
(3 → 2) because permanent infrastructure in arid zones is more stable between
acquisitions.

### Arctic Profile Rationale

Freeze/thaw phase transitions dominate backscatter over vegetation density.
`VV_WATER_MAX` raised to 0.015 for roughened melt-pond surfaces. Prostrate tundra
(sedges, mosses, dwarf shrubs) produces much less volume scattering than temperate
canopy, requiring a lower `VV_VEG_MAX` (0.15 → 0.12). Wind-packed snow can suppress
VH into the urban CR range; tightened thresholds prevent false-positive urban
detection over snowfields. `min_persistence` raised to 5 (≈ 30 days at 6-day revisit)
because tundra surface state oscillates rapidly across freeze/thaw boundaries.

---

## 6. Certification Gate Summary

**Execute against your validation confusion matrix:**

```python
from app.analytics.sar_classifier import compute_accuracy_metrics, check_certification_gate

metrics = compute_accuracy_metrics(confusion_matrix_array, class_names=[...])
result  = check_certification_gate(metrics)

print(result["recommendation"])  # PROMOTE_TO_PRODUCTION | RETAIN_EXPERIMENTAL
print(result["watermark"])
```

| Gate                          | Required   | Measured | Pass? |
|-------------------------------|------------|----------|-------|
| Overall Accuracy (OA)         | ≥ 85.0 %   | [VALUE]  | ☐ Y ☐ N |
| Kappa Coefficient (κ)         | ≥ 0.80     | [VALUE]  | ☐ Y ☐ N |
| Urban — User's Accuracy       | ≥ 75.0 %   | [VALUE]  | ☐ Y ☐ N |
| Urban — Producer's Accuracy   | ≥ 75.0 %   | [VALUE]  | ☐ Y ☐ N |
| Wet Forest — User's Accuracy  | ≥ 75.0 %   | [VALUE]  | ☐ Y ☐ N |
| Wet Forest — Producer's Acc.  | ≥ 75.0 %   | [VALUE]  | ☐ Y ☐ N |
| **ALL GATES**                 |            |          | ☐ Y ☐ N |

**Certification Decision:** ☐ PROMOTE TO PRODUCTION   ☐ RETAIN EXPERIMENTAL

**Auditor Signature:** __________________________ **Date:** __________

---

## 7. Go-To-Market (GTM) Checklist — Final Release Build

### 7.1 Statistical Validation (Must-Have before GTM)

- [ ] 100 control points sampled using `generate_validation_protocol(n_points=100)`
- [ ] Two independent interpreters; inter-rater κ ≥ 0.85 documented
- [ ] Confusion matrix populated and committed to `/data/stats/validation_confusion_matrix.csv`
- [ ] `compute_accuracy_metrics()` run; output JSON archived at `/reports/accuracy_metrics.json`
- [ ] `check_certification_gate()` returns `"PROMOTE_TO_PRODUCTION"` — screenshot or log attached
- [ ] Per-class F1 scores for Urban and Wet Forest reported and ≥ 75 %

### 7.2 Visual Output Quality

- [ ] `compress_dynamic_range(db_min=-25, db_max=5)` applied to all VV export images
- [ ] Max VV outlier (29432.69 linear / +44.7 dB) clipped at +5 dB ceiling — verified visually
- [ ] Valley roads visible in display image (≥ 10 % contrast above shadow floor)
- [ ] Forest density gradients distinguishable across at least 3 tonal bands
- [ ] Optional `apply_histogram_equalization()` evaluated for forest-dense tiles only
- [ ] QUANTITATIVE rasters (for analysis) confirmed to use raw log-dB, NOT equalized output

### 7.3 Legal & Disclaimer Compliance

- [ ] `LEGAL_DISCLAIMERS["radar_artifacts"]` text appears in all PDF report footers
- [ ] `LEGAL_DISCLAIMERS["copernicus_attribution"]` appears verbatim — year variable populated
- [ ] `LEGAL_DISCLAIMERS["classification_uncertainty"]` included in report appendix
- [ ] Radar shadow / layover zone map (from incidence angle + DEM slope) generated and embedded
- [ ] All shadow/layover pixels masked with `rtc_quality_flag=False` before `explain_pixel()` calls
- [ ] Report cover page clearly shows EXPERIMENTAL or PRODUCTION watermark from `check_certification_gate()`

### 7.4 Explainability Trace Integration

- [ ] `explain_pixel()` API endpoint operational and accessible via REST
- [ ] Response contract validated: all 7 keys present (`final_class`, `final_label`, `confidence`, `trace`, `plain_language`, `rtc_warning`, `plain_language`)
- [ ] `rtc_quality_flag` and `rtc_local_inc_angle_deg` populated from GLO-30 processing metadata
- [ ] `rtc_warning` non-None triggers visual red-flag indicator in the UI pixel inspector
- [ ] 6 trace steps present in every response (no step skipped silently)
- [ ] Sample evidence packet (3 Urban, 3 Wet Forest, 2 Anomalous pixels) reviewed and approved by QA lead

### 7.5 Multi-Biome Portability

- [ ] Five profiles confirmed accessible: `alpine_temperate`, `tropical`, `arid`, `boreal`, `arctic`
- [ ] `suggest_cr_thresholds()` returns non-empty `threshold_diff` for all non-default profiles
- [ ] Arctic profile `min_persistence=5` documented in product changelog
- [ ] Regional profile selection exposed in job configuration (`tasks.json` or API parameter)
- [ ] Regression tests updated to cover Arctic and Tropical threshold combinations

### 7.6 Provenance & Audit Trail (AGENTS.md Mandate)

- [ ] All processed outputs include SHA-256 hash (Role 12 — Integrity Lead)
- [ ] Provenance JSON captures Sentinel-1 Orbit ID, acquisition datetime, and RTC DEM version
- [ ] PDF Evidence Pack generated by ReportLab includes hash in footer (Role 11 — PDF Lead)
- [ ] No API keys in logs (Role Operational Rule §3)
- [ ] No hardcoded paths; all paths use `os.path.join()` (Role Operational Rule §2)
- [ ] STAC-compliant metadata catalogue entry created (Role 13 — Export Lead)
- [ ] End-to-end integration test (Ingest → Classify → Report) passes (Role 15 — Final QA)

### 7.7 Final Sign-Off

| Role                      | Name | Signature | Date |
|---------------------------|------|-----------|------|
| QA Auditor                |      |           |      |
| Pipeline Lead (Squad Beta)|      |           |      |
| Reporting Lead (Squad Gamma)|    |           |      |
| Product Owner             |      |           |      |

---

*End of Certification Report — CERT-JOB-02C2C063-v1.0*
