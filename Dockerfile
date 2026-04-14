# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libspatialindex-dev \
        libgeos-dev libproj-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements/omni_genesis.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal32 libspatialindex6 libgeos-c1v5 libproj25 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Create persistent data directories (overridden by volume mounts in prod)
RUN mkdir -p data/raw data/processed data/stats data/stac_catalog data/cache results reports

# Drop root — run as unprivileged user
RUN useradd -m -u 1000 scout && chown -R scout:scout /app
USER scout

EXPOSE 8000

# Run DB migrations then start the server.
# WORKERS defaults to 2; override via env var.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}"]
