# Pin to Debian 12 (bookworm) — stable package names for GDAL, spatialindex, etc.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libspatialindex-dev \
        libgeos-dev libproj-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker layer-caches them
COPY requirements/omni_genesis.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create persistent data directories (overridden by volume mounts in prod)
RUN mkdir -p data/raw data/processed data/stats data/stac_catalog data/cache results reports

# Drop root — run as unprivileged user
RUN useradd -m -u 1000 scout && chown -R scout:scout /app
USER scout

EXPOSE 8000

# PORT is injected by Railway; WORKERS defaults to 2.
CMD ["sh", "-c", "alembic upgrade head; uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
