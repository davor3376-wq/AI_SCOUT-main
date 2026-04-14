import os
import glob
import logging
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from app.api.job_manager import JobManager

logger = logging.getLogger("ChronosEngine")

class AutoDifferencing:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.jm = JobManager()

    def find_previous_job(self, current_job_id: str, bbox: List[float], sensor: str) -> Optional[Dict[str, Any]]:
        """
        Finds the most recent completed job with a matching bbox and sensor.
        """
        jobs = self.jm.list_jobs()
        # Filter out current job and ensure completed
        jobs = [j for j in jobs if j["id"] != current_job_id and j["status"] == "COMPLETED"]

        # Filter by sensor
        jobs = [j for j in jobs if j.get("sensor") == sensor]

        best_job = None

        # Simple exact bbox match for now.
        for job in jobs:
            job_bbox = job.get("bbox")
            if not job_bbox:
                continue

            # Allow small float tolerance
            if len(job_bbox) == 4 and len(bbox) == 4:
                if np.allclose(job_bbox, bbox, atol=1e-3):
                     best_job = job
                     break # jobs are sorted by created_at desc

        return best_job

    def compute_difference(self, current_tif: str, previous_tif: str, date_str: str) -> Optional[str]:
        """
        Computes the difference between two NDVI TIFs.
        Saves the difference map.
        """
        try:
            with rasterio.open(current_tif) as src_curr, rasterio.open(previous_tif) as src_prev:
                # Read data
                # Assuming single band NDVI
                data_curr = src_curr.read(1)
                data_prev = src_prev.read(1)

                # Check dimensions and CRS. If different, reproject prev to curr.
                if (data_curr.shape != data_prev.shape or
                    src_curr.crs != src_prev.crs or
                    src_curr.transform != src_prev.transform):

                    logger.info("Aligning previous image to current image geometry...")

                    data_prev_aligned = np.full(data_curr.shape, np.nan, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src_prev, 1),
                        destination=data_prev_aligned,
                        src_transform=src_prev.transform,
                        src_crs=src_prev.crs,
                        dst_transform=src_curr.transform,
                        dst_crs=src_curr.crs,
                        resampling=Resampling.bilinear
                    )
                    data_prev = data_prev_aligned

                # Handle NaNs (difference = current - previous)
                # Keep NaN if either is NaN
                diff = data_curr - data_prev

                # Output path
                output_filename = f"{date_str}_NDVI_DIFF.tif"
                output_path = os.path.join(self.output_dir, output_filename)

                profile = src_curr.profile.copy()
                profile.update(dtype=rasterio.float32)

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(diff.astype(rasterio.float32), 1)

                logger.info(f"Saved Difference Map to {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Error computing difference: {e}")
            return None

class GifGenerator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.jm = JobManager()

    def generate_timelapse(self, current_job_id: str, bbox: List[float], current_files: List[str]) -> Optional[str]:
        """
        Generates a GIF timelapse from all jobs matching the bbox.
        """
        # Find all matching jobs (including current if it was already saved, but here we pass files explicitly)
        jobs = self.jm.list_jobs()
        matching_jobs = []

        for job in jobs:
            job_bbox = job.get("bbox")
            if not job_bbox:
                continue
            # Match bbox
            if len(job_bbox) == 4 and len(bbox) == 4 and np.allclose(job_bbox, bbox, atol=1e-3) and job["status"] == "COMPLETED":
                matching_jobs.append(job)

        # Sort by date (ascending)
        matching_jobs.sort(key=lambda x: x.get("start_date", x.get("created_at")))

        # Collect images (NDVI preferred)
        image_files = []

        # Add historical
        for job in matching_jobs:
            if job["id"] == current_job_id:
                continue

            processed = job.get("results", {}).get("processed_files", [])
            ndvi = next((f for f in processed if "NDVI" in f), None)
            if ndvi and os.path.exists(ndvi):
                date_label = job.get("start_date", "Unknown").split("T")[0]
                image_files.append((date_label, ndvi))

        # Add current
        current_ndvi = next((f for f in current_files if "NDVI" in f), None)
        if current_ndvi and os.path.exists(current_ndvi):
             # Try to extract date from filename or use "Current"
             # Filename format: {date}_{sensor}_{tile_id}_NDVI_analysis.tif or similar
             filename = os.path.basename(current_ndvi)
             parts = filename.split("_")
             date_label = parts[0] if len(parts) > 0 else "Current"
             image_files.append((date_label, current_ndvi))

        if len(image_files) < 2:
            logger.info("Not enough images for timelapse.")
            return None

        # Cap at the 24 most recent frames to bound memory use on long-running
        # monitoring jobs.  Older frames are dropped; the GIF stays navigable.
        MAX_FRAMES = 24
        if len(image_files) > MAX_FRAMES:
            logger.warning(
                f"Timelapse has {len(image_files)} frames — capping at {MAX_FRAMES} most recent."
            )
            image_files = image_files[-MAX_FRAMES:]

        output_filename = f"Timelapse_{current_job_id}.gif"
        output_path = os.path.join(self.output_dir, output_filename)

        try:
            norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
            cmap = plt.get_cmap("RdYlGn")

            # Build palette-mode frames one at a time so only two frames
            # (current + running list of P-mode images) live in memory at once.
            palette_frames: list[Image.Image] = []

            for _date_label, tif_path in image_files:
                with rasterio.open(tif_path) as src:
                    data = src.read(1)

                colored = cmap(norm(data))
                nan_mask = np.isnan(data)
                colored[nan_mask] = [1.0, 1.0, 1.0, 1.0]  # white for no-data

                # RGB → resize → quantize to P mode (palette GIF, ~4× smaller RAM)
                rgb = Image.fromarray(
                    (colored[:, :, :3] * 255).astype(np.uint8), "RGB"
                ).resize((512, 512), Image.LANCZOS)

                palette_frames.append(rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT))

                # Explicitly release the full-res arrays
                del data, colored, rgb

            palette_frames[0].save(
                output_path,
                save_all=True,
                append_images=palette_frames[1:],
                duration=800,
                loop=0,
                optimize=True,
            )

            logger.info(f"Saved timelapse ({len(palette_frames)} frames) to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating GIF: {e}", exc_info=True)
            return None
