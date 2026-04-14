"""
Vercel Python entrypoint.

NOTE: Vercel is suitable for the frontend SPA only. The full AI SCOUT backend
requires persistent storage, background workers, GDAL, and PostgreSQL — none of
which are available in Vercel's serverless runtime.

For the backend use Railway (see railway.json) with the Dockerfile.
This file exists only to satisfy Vercel's entrypoint detection if this repo
is also used for frontend hosting.
"""

from app.api.main import app  # noqa: F401  (re-exported as the ASGI handler)
