"""
Centralized configuration for AI SCOUT.

All tunables read from environment variables with sensible defaults.
Import from here instead of scattering os.environ.get() across modules.
"""

import os

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
SCOUT_DB_PATH: str = os.environ.get("SCOUT_DB_PATH", "jobs.db")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: str = os.environ.get("SCOUT_RAW_DIR", os.path.join("data", "raw"))
PROCESSED_DATA_DIR: str = os.environ.get("SCOUT_PROCESSED_DIR", os.path.join("data", "processed"))
RESULTS_DIR: str = os.environ.get("SCOUT_RESULTS_DIR", "results")
CACHE_DIR: str = os.environ.get("SCOUT_CACHE_DIR", os.path.join("data", "cache"))
STAC_DIR: str = os.environ.get("SCOUT_STAC_DIR", os.path.join("data", "stac_catalog"))
STATS_DIR: str = os.environ.get("SCOUT_STATS_DIR", os.path.join("data", "stats"))

# ---------------------------------------------------------------------------
# Auth & Security
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.environ.get("SCOUT_SECRET_KEY", "")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES") or "480")
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS") or "7")
MIN_PASSWORD_LENGTH: int = int(os.environ.get("SCOUT_MIN_PASSWORD_LENGTH") or "10")

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS: str = os.environ.get("SCOUT_ALLOWED_ORIGINS", "")

# ---------------------------------------------------------------------------
# Cleanup / Retention
# ---------------------------------------------------------------------------
DATA_RETENTION_DAYS: int = int(os.environ.get("DATA_RETENTION_DAYS") or "90")

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
DEFAULT_RATE_LIMIT: str = os.environ.get("SCOUT_RATE_LIMIT", "200/minute")

# ---------------------------------------------------------------------------
# Stripe / Billing
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY: str = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET: str = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
SCOUT_FRONTEND_URL: str = os.environ.get("SCOUT_FRONTEND_URL", "http://localhost:3000")

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development").lower()
SCOUT_DOMAIN: str = os.environ.get("SCOUT_DOMAIN", "localhost")

# ---------------------------------------------------------------------------
# Billing mode
# ---------------------------------------------------------------------------
# stripe   — full Stripe Checkout flow (default)
# grant    — Stripe checkout disabled; admins assign credits via POST /billing/topup
# disabled — billing router not loaded at all (suitable for fully grant-funded orgs)
BILLING_MODE: str = os.environ.get("BILLING_MODE", "stripe").lower()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "json").lower()
