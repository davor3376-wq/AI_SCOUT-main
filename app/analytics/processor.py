"""
Analytics Processor.
Orchestrates the analysis of raw satellite imagery.
"""
import os
import glob
import logging
import rasterio
import numpy as np
from pathlib import Path

from app.analytics.indices import calculate_ndvi, calculate_ndwi, calculate_nbr
from app.analytics.masking import get_cloud_mask, get_cloud_mask_from_qa60, calculate_cloud_coverage
from app.analytics.alerting import calculate_alert_level
from app.reporting.notifications import send_alert_sync
from app.core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

INPUT_DIR = RAW_DATA_DIR
OUTPUT_DIR = PROCESSED_DATA_DIR

def process_scene(filepath: str, requested_indices=None):
    """
    Processes a single raw Sentinel-2 TIF file.
    Calculates NDVI (and NDWI if possible) and saves the results.
    Returns a list of generated output file paths.
    """
    filename = os.path.basename(filepath)
    # Expected format: {date}_{sensor}_{tile_id}.tif
    # Example: 20231025_S2_T33UUE.tif
    parts = filename.replace(".tif", "").split("_")
    if len(parts) < 3:
        logger.warning(f"Skipping file with unexpected name format: {filename}")
        return []

    date_str = parts[0]
    sensor = parts[1]
    # tile_id = parts[2] # Unused for now

    if sensor != "S2":
        logger.info(f"Skipping non-S2 file: {filename}")
        return []

    logger.debug(f"Processing {filename}...")

    generated_files = []

    with rasterio.open(filepath) as src:
        # Check band count
        # Current format: B03, B04, B08, B12, SCL (5 bands)
        # Previous format: B03, B04, B08, SCL       (4 bands)
        # Legacy format:   B04, B08, SCL             (3 bands)
        if src.count < 3:
            logger.error(f"File {filename} has insufficient bands ({src.count}). Expected at least 3.")
            return []

        # Read bands
        green = None
        swir2 = None

        if src.count >= 5:
            # Current format: B03, B04, B08, B12, SCL
            green = src.read(1)
            red   = src.read(2)
            nir   = src.read(3)
            swir2 = src.read(4)
            scl   = src.read(5)
        elif src.count == 4:
            # Previous format: B03, B04, B08, SCL
            green = src.read(1)
            red   = src.read(2)
            nir   = src.read(3)
            scl   = src.read(4)
        else:
            # Legacy format: B04, B08, SCL
            red = src.read(1)
            nir = src.read(2)
            scl = src.read(3)

        # Generate Cloud Mask
        cloud_mask = get_cloud_mask(scl)
        cloud_cover_pct = calculate_cloud_coverage(cloud_mask)

        if cloud_cover_pct > 20.0:
            logger.warning(f"High cloud cover ({cloud_cover_pct:.1f}%) in {filename} — Low Confidence.")

        # Determine which indices to compute (default: all available)
        want = set(requested_indices) if requested_indices else {"NDVI", "NDWI", "NBR"}

        results = []

        # NDVI
        if "NDVI" in want:
            logger.debug("Calculating NDVI...")
            ndvi = calculate_ndvi(red, nir)
            ndvi[cloud_mask] = np.nan

            alert_level = calculate_alert_level(ndvi, cloud_cover_pct)
            if alert_level == "HIGH":
                logger.warning(f"HIGH ALERT for {filename}. Sending notification.")
                send_alert_sync(
                    f"WATCHDOG ALERT\nFile: {filename}\n"
                    f"NDVI Mean: {np.nanmean(ndvi):.2f}\nCloud Cover: {cloud_cover_pct:.2f}%"
                )

            path = save_result(ndvi, src.profile, date_str, "NDVI")
            if path:
                results.append(path)

        # NDWI
        if "NDWI" in want:
            if green is not None:
                logger.debug("Calculating NDWI...")
                ndwi = calculate_ndwi(green, nir)
                ndwi[cloud_mask] = np.nan
                path = save_result(ndwi, src.profile, date_str, "NDWI")
                if path:
                    results.append(path)
            else:
                logger.warning("Green band (B03) not available — skipping NDWI.")

        # NBR
        if "NBR" in want:
            if swir2 is not None:
                logger.debug("Calculating NBR...")
                nbr = calculate_nbr(nir, swir2)
                nbr[cloud_mask] = np.nan
                path = save_result(nbr, src.profile, date_str, "NBR")
                if path:
                    results.append(path)
            else:
                logger.warning("SWIR2 band (B12) not available — skipping NBR. Re-download with 5-band evalscript.")

        if results:
            indices_done = ", ".join(os.path.basename(p).split("_")[1] for p in results)
            logger.info(f"{filename} → {indices_done} (cloud {cloud_cover_pct:.1f}%)")

        return results

def save_result(data: np.ndarray, profile, date_str: str, index_name: str):
    """
    Saves the calculated index to a GeoTIFF.
    Returns the path to the saved file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Update profile for the output
    # Remove potential conflicting keys from source profile
    profile.pop('blockxsize', None)
    profile.pop('blockysize', None)

    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan,
        driver='GTiff',
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    )

    output_filename = f"{date_str}_{index_name}_analysis.tif"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)
        logger.debug(f"Saved {index_name} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return None

def run(input_files=None, requested_indices=None):
    """
    Main entry point to process files.

    Args:
        input_files: Optional list of specific file paths to process.
                     If None, processes all .tif files in data/raw.
        requested_indices: Optional list of index names to calculate
                           (e.g. ["NDVI", "NDWI"]). None means all available.

    Returns:
        List of paths to the processed output files.
    """
    if input_files is None:
        if not os.path.exists(INPUT_DIR):
            logger.warning(f"Input directory {INPUT_DIR} does not exist.")
            return []
        input_files = glob.glob(os.path.join(INPUT_DIR, "*.tif"))

    if not input_files:
        logger.warning("No TIF files found to process.")
        return []

    output_files = []
    for filepath in input_files:
        result = process_scene(filepath, requested_indices=requested_indices)
        if result:
            output_files.extend(result)

    logger.info(f"Analytics complete: {len(output_files)} output(s) from {len(input_files)} scene(s).")
    return output_files

if __name__ == "__main__":
    run()
