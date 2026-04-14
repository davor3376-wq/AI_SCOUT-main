# AI Scout — Evidence-Grade Environmental Monitoring

> **"If it isn't in the provenance log, it didn't happen."**

Most satellite analytics tools give you data. AI Scout gives you **evidence** — legally defensible, auditable, court-ready intelligence from raw Sentinel imagery.

Every pixel is explainable. Every output is hashed. Every step is logged.

---

## Why AI Scout

| What others deliver | What AI Scout delivers |
|---|---|
| Processed rasters | SHA-256 signed Evidence Packs |
| Index values | Per-pixel 6-step explainability trace |
| Classification maps | ISO 19157:2013 certified accuracy metrics |
| Raw outputs | Confidence ratings (HIGH / MEDIUM / LOW) per pixel |
| One-size-fits-all | 5 calibrated climate profiles (alpine → arctic → tropical) |

---

## What It Does

**Input:** Area of Interest (GeoJSON) + date range + sensor config

**Output:** A PDF Evidence Pack containing:
- Classified land cover map (6 classes)
- Per-pixel confidence ratings and explainability traces
- SHA-256 integrity hash of every source file in the footer
- Mandatory Copernicus/ESA attribution
- EXPERIMENTAL or PRODUCTION watermark based on certified accuracy gates
- STAC-compliant metadata catalogue entry

**Pipeline: Ingest → Analyse → Certify → Report**

```
Sentinel-1 SAR (VV/VH)  ──┐
                          ├──▶  SAR Classifier  ──▶  Certification Gate  ──▶  PDF Evidence Pack
Sentinel-2 Optical (NDVI)─┘         │                      │
                                     ▼                      ▼
                              Provenance JSON          SHA-256 Hash
```

---

## Land Cover Classes

| ID | Class | Key Signal |
|----|-------|-----------|
| 0 | Water / Smooth Surface | Specular VV < 0.01 |
| 1 | Bare Soil / Low Vegetation | Low VV, low CR |
| 2 | Vegetation / Forest | VV 0.05–0.15, NDVI > 0.50 |
| 3 | Urban / Built-up | High VV, CR < 0.15, persistent |
| 4 | Anomalous Vegetation / Moisture | Forest baseline + urban signal |
| 5 | Wet Forest / Dielectric Anomaly | CR 0.15–0.30 |

---

## Certification Gates

A report is only stamped **PRODUCTION** when all four gates pass simultaneously:

| Gate | Threshold | Standard |
|------|-----------|---------|
| Overall Accuracy | ≥ 85.0 % | ASPRS 2014 Tier 3 (legal evidence threshold) |
| Kappa (κ) | ≥ 0.80 | "Almost Perfect" — eliminates chance-agreement inflation |
| Urban UA & PA | ≥ 75.0 % each | Prevents false enforcement actions |
| Wet Forest UA & PA | ≥ 75.0 % each | High-confusion class; must not collapse into Urban |

Fail any gate → report carries **EXPERIMENTAL** watermark and cannot be used as sole evidence.

---

## Per-Pixel Explainability

Every classified pixel carries a full 6-step audit trace. Required for any pixel submitted to legal proceedings.

```python
trace = explain_pixel(
    vv=0.18, vh=0.022, ndvi=0.42, ndvi_baseline=0.55,
    persistence_run=4, min_persistence=3,
    rtc_quality_flag=True, rtc_local_inc_angle_deg=38.2
)
# → final_class, final_label, confidence (HIGH/MEDIUM/LOW),
#   trace[6 steps], plain_language, rtc_warning
```

A non-`None` `rtc_warning` **disqualifies** the pixel from sole-evidence status — the pipeline tells you this automatically.

---

## Climate Profiles

Five calibrated threshold profiles. Select at job configuration time:

| Profile | Best for |
|---------|---------|
| `alpine_temperate` | Default — validated ESA C-band benchmark |
| `tropical` | Multi-layer canopy; mandatory NDVI fusion |
| `arid` | Sandy substrate; strong urban contrast |
| `boreal` | Frozen-needle VH suppression in winter |
| `arctic` | Freeze/thaw transitions; snow VH ambiguity |

```python
from app.analytics.sar_classifier import suggest_cr_thresholds
profile = suggest_cr_thresholds("tropical")
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements/omni_genesis.txt

# 2. Configure Sentinel Hub credentials
cp .env.example .env  # add your SH_CLIENT_ID and SH_CLIENT_SECRET

# 3. Run a mission (defaults to Vienna AOI, last 30 days)
python main.py

# Output: /data/raw/, /data/processed/, /data/stats/, /reports/Evidence_Pack_*.pdf
```

**Programmatic use:**

```python
from main import main
from sentinelhub import BBox, CRS

files = await main(
    bbox=BBox([16.2, 48.1, 16.5, 48.3], CRS.WGS84),
    time_interval=("2026-03-01", "2026-04-01")
)
```

---

## Output Structure

```
/data/raw/        {date}_{sensor}_{tileID}.tif          # COG format
                  {date}_{sensor}_{tileID}_provenance.json
/data/processed/  {date}_{index}_analysis.tif
/data/stats/      {date}_{index}_zonal.csv
/reports/         Evidence_Pack_{ID}.pdf                # SHA-256 signed
```

---

## Architecture

15-agent swarm across three squads:

**Squad Alpha — The Harvesters** (Roles 1–5): Sentinel Hub OAuth2, SAR + optical download, provenance logging, ingest QA

**Squad Beta — The Alchemists** (Roles 6–10): NDVI/NDWI calculation, temporal differencing, cloud/shadow masking (SCL band), zonal statistics, index QA

**Squad Gamma — The Notaries** (Roles 11–15): ReportLab PDF generation, SHA-256 integrity, STAC metadata, methodology auto-text, end-to-end integration test

See [AGENTS.md](AGENTS.md) for full swarm protocol and I/O contracts.

---

## Legal & Attribution

All outputs include mandatory Copernicus/ESA attribution per Regulation (EU) No 1159/2013:

> "Contains modified Copernicus Sentinel data [YEAR], processed by ESA."
> "Copernicus DEM GLO-30 © DLR e.V. 2021 and © Airbus Defence and Space GmbH 2021."

Radar geometric distortions (shadowing, foreshortening, layover) are disclosed in every report footer. Classifications within identified layover zones are automatically flagged UNRELIABLE.

---

## Certification & GTM

Before production release, run the full certification protocol defined in [CERTIFICATION_REPORT.md](CERTIFICATION_REPORT.md):

- 100 stratified control points (ISO 19157:2013 §6.4)
- Two independent interpreters, inter-rater κ ≥ 0.85
- All six accuracy gates must pass
- SHA-256 provenance chain verified end-to-end
