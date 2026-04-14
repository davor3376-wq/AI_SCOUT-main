"""
Shared pytest configuration for AI SCOUT.

Sets all required environment variables *before* any application module is
imported.  This file is loaded automatically by pytest before collecting any
tests, so settings here apply to the whole test session.
"""

import os

# ── Must be set before any app import ────────────────────────────────────────
os.environ.setdefault("SCOUT_SECRET_KEY", "test-secret-key-minimum-32-chars-xx!")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("LOG_FORMAT", "plain")
os.environ.setdefault("SCOUT_ADMIN_USERNAME", "admin")
os.environ.setdefault("SCOUT_ADMIN_PASSWORD", "AdminPass1!")
# Prevent Sentinel Hub real network calls during unit tests
os.environ.setdefault("SH_CLIENT_ID", "")
os.environ.setdefault("SH_CLIENT_SECRET", "")
