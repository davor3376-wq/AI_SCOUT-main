"""
User Upload Handler (Ingestion).
Responsible for parsing and validating user-uploaded geometries (GeoJSON, Shapefile, etc.).
"""
import os
import logging
import geopandas as gpd
from sentinelhub import BBox, CRS
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class UserUploadHandler:
    """
    Handles user uploads of AOI definitions.
    """

    def parse_file(self, filepath: str) -> List[BBox]:
        """
        Parses a vector file and returns a list of BBoxes.

        Args:
            filepath: Path to the uploaded file.

        Returns:
            List of SentinelHub BBox objects.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            gdf = gpd.read_file(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

        # Ensure CRS is WGS84 (EPSG:4326)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        bboxes = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            bounds = geom.bounds # (minx, miny, maxx, maxy)

            bbox = BBox(
                bbox=[bounds[0], bounds[1], bounds[2], bounds[3]],
                crs=CRS.WGS84
            )
            bboxes.append(bbox)

        return bboxes

    def validate_upload(self, filepath: str) -> bool:
        """
        Validates if the file is a valid vector format.
        """
        try:
            gpd.read_file(filepath)
            return True
        except Exception as e:
            logger.warning(f"File validation failed for {filepath}: {e}")
            return False
