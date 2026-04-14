"""
STAC Catalog Manager (Infrastructure).
Responsible for indexing ingested files using the SpatioTemporal Asset Catalog (STAC) specification.
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pystac
from pystac import Catalog, Item, Asset, CatalogType
from sentinelhub import BBox

logger = logging.getLogger(__name__)

class StacCatalogManager:
    """
    Manages a local STAC Catalog for ingested data.
    """

    def __init__(self, catalog_root: str = "data/stac_catalog"):
        """
        Initialize the STAC Catalog Manager.

        Args:
            catalog_root: Path to the root directory of the STAC Catalog.
        """
        self.catalog_root = catalog_root
        self.catalog_path = os.path.join(catalog_root, "catalog.json")
        self._ensure_catalog()

    def _ensure_catalog(self):
        """
        Ensures the STAC Catalog exists. If not, creates it.
        """
        if not os.path.exists(self.catalog_path):
            os.makedirs(self.catalog_root, exist_ok=True)
            catalog = Catalog(
                id="ai-scout-catalog",
                description="Catalog for AI Scout Environmental Monitoring Platform",
                title="AI Scout Catalog"
            )
            catalog.normalize_hrefs(self.catalog_root)
            catalog.save(catalog_type=CatalogType.SELF_CONTAINED)

    def get_catalog(self) -> Catalog:
        """Loads the catalog."""
        return Catalog.from_file(self.catalog_path)

    def add_item(self,
                 item_id: str,
                 filepath: str,
                 bbox: BBox,
                 time: datetime,
                 properties: Dict[str, Any] = None):
        """
        Adds an item (image) to the STAC Catalog.

        Args:
            item_id: Unique identifier for the item.
            filepath: Path to the image file (relative to repo root or absolute).
            bbox: Bounding box of the item.
            time: Acquisition time.
            properties: Additional metadata properties.
        """
        catalog = self.get_catalog()

        # Calculate geometry from bbox
        min_x, min_y = bbox.lower_left
        max_x, max_y = bbox.upper_right

        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
                [min_x, min_y]
            ]]
        }

        # Create STAC Item
        item = Item(
            id=item_id,
            geometry=geometry,
            bbox=[min_x, min_y, max_x, max_y],
            datetime=time,
            properties=properties or {}
        )

        # Add Asset
        # Assuming filepath is relative to the execution root, we need to make it accessible to the catalog.
        # Ideally, STAC items are near their assets.
        # For this local catalog, we will point to the file.

        # Determine media type based on extension
        media_type = pystac.MediaType.GEOTIFF
        if filepath.endswith(".json"):
            media_type = pystac.MediaType.JSON
        elif filepath.endswith(".png"):
            media_type = pystac.MediaType.PNG

        item.add_asset(
            key="data",
            asset=Asset(
                href=os.path.abspath(filepath),
                media_type=media_type,
                title="Main Data Asset"
            )
        )

        catalog.add_item(item)

        # Normalize and save
        catalog.normalize_hrefs(self.catalog_root)
        catalog.save(catalog_type=CatalogType.SELF_CONTAINED)
        logger.info(f"Added item {item_id} to STAC Catalog.")
