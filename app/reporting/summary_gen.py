import os
import logging
import rasterio
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as colors

logger = logging.getLogger(__name__)

def generate_mobile_summary(input_files: list, output_dir: str = "results", job_id: str = "summary") -> str:
    """
    Generates a lightweight JPG summary for low-bandwidth agents.
    Prioritizes NDVI files, falls back to the first available file.
    """
    try:
        # Select best file (NDVI preferred)
        target_file = None
        for f in input_files:
            if "NDVI" in f:
                target_file = f
                break

        if not target_file and input_files:
            target_file = input_files[0]

        if not target_file:
            logger.warning("No input files to generate mobile summary.")
            return None

        output_filename = f"Mobile_Summary_{job_id}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        with rasterio.open(target_file) as src:
            # Read first band
            data = src.read(1)

            # Handling 10,000x Integer Scaling Check (Same logic as PDF Gen)
            valid_data = data[~np.isnan(data)]
            if valid_data.size > 0:
                max_val = np.max(valid_data)
                if max_val > 100:
                    data = data / 10000.0

            # Normalize and Colormap
            norm = colors.Normalize(vmin=-1.0, vmax=1.0)
            cmap = cm.get_cmap('RdYlGn')
            colored_data = cmap(norm(data))

            # Handle NaNs (Make black for JPG)
            nan_mask = np.isnan(data)
            colored_data[nan_mask] = [0, 0, 0, 1] # Black background

            # Convert to uint8
            img_data = (colored_data * 255).astype(np.uint8)

            # Create PIL Image
            img = Image.fromarray(img_data, 'RGBA').convert('RGB') # JPG doesn't support Alpha

            # Resize to max 800x800
            img.thumbnail((800, 800))

            # Add a "Mobile Summary" watermark text (optional, but good for context)
            # Skipping complex drawing for now, just saving optimized image.

            # Save with compression
            img.save(output_path, "JPEG", quality=60, optimize=True)

        logger.info(f"Mobile summary generated at {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate mobile summary: {e}")
        return None
