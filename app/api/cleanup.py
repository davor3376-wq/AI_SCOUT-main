"""
Data retention — deletes raw and processed files older than DATA_RETENTION_DAYS.

Called:
  - Automatically on startup (async background task, non-blocking)
  - Manually via DELETE /admin/cleanup (admin only)

Directories cleaned:
  data/raw/       satellite imagery TIFs + provenance JSON
  data/processed/ NDVI/NDWI analysis TIFs
  results/        GIF timelapses, mobile JPG summaries
                  (PDFs and KML are intentionally kept — they are evidence)
"""

import os
import glob
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from app.core.config import DATA_RETENTION_DAYS, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# PDFs and KML are excluded — they are the legal evidence artefacts
_CLEANUP_PATTERNS = [
    os.path.join(RAW_DATA_DIR, "*.tif"),
    os.path.join(RAW_DATA_DIR, "*.json"),
    os.path.join(PROCESSED_DATA_DIR, "*.tif"),
    os.path.join(RESULTS_DIR, "*.gif"),
    os.path.join(RESULTS_DIR, "*.jpg"),
]


def run_cleanup(retention_days: int = DATA_RETENTION_DAYS) -> Dict[str, Any]:
    """
    Delete files older than *retention_days*.
    Returns a summary dict.
    """
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=retention_days)).timestamp()
    deleted: list[str] = []
    errors: list[str] = []

    for pattern in _CLEANUP_PATTERNS:
        for filepath in glob.glob(pattern):
            try:
                if os.path.getmtime(filepath) < cutoff_ts:
                    os.remove(filepath)
                    deleted.append(filepath)
            except Exception as exc:
                logger.warning(f"Could not delete {filepath}: {exc}")
                errors.append(filepath)

    logger.info(
        f"Cleanup complete — deleted {len(deleted)} files, "
        f"{len(errors)} errors, retention={retention_days}d"
    )
    return {
        "deleted_count": len(deleted),
        "error_count": len(errors),
        "retention_days": retention_days,
        "cutoff_utc": datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat(),
    }
