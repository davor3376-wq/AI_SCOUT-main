import asyncio
import logging
import os
import argparse
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union, List

from app.ingestion.s2_client import S2Client
from app.analytics import processor
from app.reporting.pdf_gen import PDFReportGenerator
from app.reporting.integrity import IntegrityChecker
from sentinelhub import BBox, CRS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main(
    bbox: Optional[BBox] = None,
    time_interval: Optional[Tuple[Union[str, datetime], Union[str, datetime]]] = None
) -> List[str]:
    """
    Executes the mission for a given BBox and time interval.
    Returns a list of generated analysis files (NDVI TIFs).
    """
    logger.info("Mission starting...")

    # Default BBox (Vienna) if not provided
    if bbox is None:
        # 16.2, 48.1 to 16.5, 48.3
        bbox = BBox(bbox=[16.2, 48.1, 16.5, 48.3], crs=CRS.WGS84)

    # Default Time Interval (Last 30 days) if not provided
    if time_interval is None:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)
        time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    # 1. Ingestion
    client = S2Client()
    downloaded_files = []
    try:
        downloaded_files = client.download_data(bbox=bbox, time_interval=time_interval)
        logger.info(f"Ingestion: {len(downloaded_files)} file(s) downloaded.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise e

    if not downloaded_files:
        logger.warning("No files downloaded. Aborting mission.")
        return []

    # 2. Analytics
    processed_files = []
    try:
        processed_files = processor.run(input_files=downloaded_files)
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise e

    # 3. Reporting
    try:
        PDFReportGenerator().generate_pdf()
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")

    # 4. Integrity
    try:
        IntegrityChecker().generate_integrity_file()
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")

    logger.info(f"Mission complete: {len(processed_files)} output(s).")
    return processed_files

if __name__ == "__main__":
    # Simple CLI handling for basic testing
    # In a real scenario, we might use argparse to parse --bbox and --time
    asyncio.run(main())
