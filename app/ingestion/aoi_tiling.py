"""
AOI Tiling Module (Infrastructure).
Responsible for splitting large Areas of Interest (AOI) into manageable tiles.
"""
from typing import List, Generator
from sentinelhub import BBox, Geometry
from shapely.geometry import box, Polygon
import math

def split_bbox(bbox: BBox, split_x: int = 2, split_y: int = 2) -> List[BBox]:
    """
    Splits a BBox into smaller BBoxes.

    Args:
        bbox: The BBox to split.
        split_x: Number of splits in X direction.
        split_y: Number of splits in Y direction.

    Returns:
        List of sub-BBoxes.
    """
    min_x, min_y = bbox.lower_left
    max_x, max_y = bbox.upper_right
    crs = bbox.crs

    width = (max_x - min_x) / split_x
    height = (max_y - min_y) / split_y

    sub_bboxes = []
    for i in range(split_x):
        for j in range(split_y):
            sub_min_x = min_x + i * width
            sub_min_y = min_y + j * height
            sub_max_x = sub_min_x + width
            sub_max_y = sub_min_y + height

            sub_bboxes.append(BBox(
                bbox=[sub_min_x, sub_min_y, sub_max_x, sub_max_y],
                crs=crs
            ))

    return sub_bboxes

def auto_tile_geometry(geometry: Geometry, max_area_sqkm: float = 100.0) -> List[BBox]:
    """
    Automatically tiles a geometry if it exceeds the max area.
    This is a simplified implementation that works on the bounding box of the geometry.

    Args:
        geometry: SentinelHub Geometry object.
        max_area_sqkm: Maximum area per tile in square kilometers.

    Returns:
        List of BBoxes covering the geometry.
    """
    bbox = geometry.bbox

    # Approximate area calculation (very rough, treating degrees as meters/111km)
    # Ideally should project to UTM, but for tiling logic, rough estimate is okay.
    # 1 deg ~ 111km.

    min_x, min_y = bbox.lower_left
    max_x, max_y = bbox.upper_right

    dx = max_x - min_x
    dy = max_y - min_y

    # Area in sq deg
    area_sqdeg = dx * dy

    # Approx conversion
    area_sqkm_approx = area_sqdeg * (111 * 111) * math.cos(math.radians((min_y + max_y)/2))

    if area_sqkm_approx <= max_area_sqkm:
        return [bbox]

    # Calculate splits
    # number of tiles needed
    n_tiles = math.ceil(area_sqkm_approx / max_area_sqkm)

    # Square root to find split factor
    split_factor = math.ceil(math.sqrt(n_tiles))

    return split_bbox(bbox, split_x=split_factor, split_y=split_factor)
