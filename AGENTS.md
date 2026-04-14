# AI Scout: System Blueprint & Swarm Protocol

**AI Scout** is an evidence-grade environmental monitoring platform.
**Mission:** Transform raw satellite data into legally defensible, auditable intelligence.
**Motto:** "If it isn't in the provenance log, it didn't happen."

---

## 🏗️ System Architecture & I/O Contracts

### 1. Ingestion Layer (`/app/ingestion`)
* **Owner:** Squad Alpha (The Harvesters)
* **Input:** AOI (GeoJSON), Date Range, Sensor Config.
* **Output:** * Raw Imagery: `/data/raw/{date}_{sensor}_{tileID}.tif` (COG format).
    * Metadata: `/data/raw/{date}_{sensor}_{tileID}_provenance.json`.
* **Mandate:** Handle Sentinel Hub OAuth2, retries, and rate limits. **Never** fail silently.

### 2. Analysis Engine (`/app/analytics`)
* **Owner:** Squad Beta (The Alchemists)
* **Input:** Raw GeoTIFFs from `/data/raw`.
* **Output:** * Processed Raster: `/data/processed/{date}_{index}_analysis.tif`.
    * Stats: `/data/stats/{date}_{index}_zonal.csv`.
* **Mandate:** Calculations (NDVI, NDWI) must account for cloud masks (SCL band). If cloud cover > 20%, flag data as "Low Confidence."

### 3. Reporting & Evidence (`/app/reporting`)
* **Owner:** Squad Gamma (The Notaries)
* **Input:** Processed Rasters + Stats CSVs + Provenance JSONs.
* **Output:** Final PDF Evidence Pack (`/reports/Evidence_Pack_{ID}.pdf`).
* **Mandate:** The PDF must include a SHA-256 hash of the source data in the footer. Use ReportLab for generation.

---

## 🤖 Swarm Roles (The 15-Agent Roster)

### Squad Alpha: The Harvesters (Roles 1-5)
* **Role 1 (Auth Lead):** Manage `sentinelhub` config and token refresh.
* **Role 2 (SAR Spec):** Handle Sentinel-1 GRD (VV/VH) downloads.
* **Role 3 (Optical Spec):** Handle Sentinel-2 L2A (B04/B08/SCL) downloads.
* **Role 4 (Archivist):** Ensure `provenance.json` captures *exact* Orbit ID and Time.
* **Role 5 (Ingest QA):** Verify downloads are not corrupt (0-byte checks).

### Squad Beta: The Alchemists (Roles 6-10)
* **Role 6 (Index Lead):** Implement optimized NDVI/NDWI using NumPy.
* **Role 7 (Change Lead):** Implement temporal differencing (Time A vs Time B).
* **Role 8 (Masking Lead):** robust cloud/shadow masking using the SCL band.
* **Role 9 (Stats Lead):** Calculate min/max/mean pixel values within the AOI.
* **Role 10 (Math QA):** Verify index values fall between -1.0 and 1.0.

### Squad Gamma: The Notaries (Roles 11-15)
* **Role 11 (PDF Lead):** ReportLab layout engine (Header, Map, Footer).
* **Role 12 (Integrity Lead):** Hashing logic (SHA-256) for all files.
* **Role 13 (Export Lead):** STAC-compliant metadata cataloging.
* **Role 14 (Scribe):** Auto-generate methodology text based on utilized bands.
* **Role 15 (Final QA):** End-to-end integration test (Ingest -> Report).

---

## 🛠️ Operational Rules (CRITICAL)

1.  **Strict Branching:** * Naming convention: `feat/squad-{alpha|beta|gamma}-role-{ID}-{task}`.
    * *Example:* `feat/squad-alpha-role-2-s1-download`.
    * **NEVER commit to main.**
2.  **Shared Files Protocol:**
    * Do NOT edit `requirements.txt` directly. Create `requirements/{role}.txt`. The Architect will merge them.
    * Do NOT edit `bridge.py`. That is the Commander's channel.
3.  **Code Style:**
    * Python 3.10+.
    * Type Hints are mandatory (e.g., `def calc_ndvi(band: np.array) -> np.array:`).
    * Docstrings must explain *why*, not just *what*.

## 🚀 Deployment Checklist
Before marking a task complete:
1.  Run local tests.
2.  Ensure no hardcoded paths (use `os.path.join`).
3.  Verify no API keys are logged.
