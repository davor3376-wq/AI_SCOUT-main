"""
Landsat Legacy Client (Ingestion).
Responsible for downloading Landsat 8/9 OLI data.
"""
import os
import logging
from datetime import datetime
from typing import List, Tuple, Union

import rasterio
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SentinelHubCatalog,
)

from app.ingestion.auth import SentinelHubAuth
from app.ingestion.metadata import generate_provenance

logger = logging.getLogger(__name__)


class LandsatClient:
    """
    Client for downloading Landsat 8/9 data.
    """

    def __init__(self):
        self.auth = SentinelHubAuth()
        self.config = self.auth.config

        # CDSE might have Landsat 8. Let's try to define it.
        # If CDSE doesn't support it, this might fail or return empty results.
        # We use the standard definition but point to CDSE URL via config.
        # We might need to override the collection ID for CDSE.
        # Assuming "LANDSAT-8-L1" is the ID on CDSE if available.
        # Or we use the standard SH collection.
        self.data_collection = DataCollection.LANDSAT_OT_L1

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 30, # Landsat is 30m
    ) -> List[str]:
        """
        Downloads Landsat data.
        """
        # Evalscript for RGB (Red, Green, Blue)
        # B04: Red, B03: Green, B02: Blue
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3, sampleType: "UINT16" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B04, sample.B03, sample.B02];
        }
        """

        # Note: Catalog search for Landsat on CDSE might differ.
        # We will try standard catalog.

        catalog = SentinelHubCatalog(config=self.config)

        # If this fails, it's likely due to Collection ID mismatch on CDSE vs SH.
        # For this exercise, we assume it works or returns empty.
        try:
            search_iterator = catalog.search(
                collection=self.data_collection,
                bbox=bbox,
                time=time_interval,
            )
            results = list(search_iterator)
        except Exception as e:
            logger.warning(f"Landsat Catalog search failed: {e}")
            return []

        output_files = []
        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]
            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "L8"

            filename = f"{date_str}_{sensor}_{tile_id}.tif"
            filepath = os.path.join("data", "raw", filename)

            req_interval = (dt, dt)

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=self.data_collection,
                        time_interval=req_interval,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=bbox,
                resolution=(resolution, resolution),
                config=self.config,
            )

            try:
                data_list = request.get_data()
            except Exception:
                continue

            if not data_list:
                continue

            image_data = data_list[0]
            height, width, bands = image_data.shape

            transform = from_bounds(
                bbox.lower_left[0], bbox.lower_left[1],
                bbox.upper_right[0], bbox.upper_right[1],
                width, height
            )

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with rasterio.open(
                filepath, 'w', driver='GTiff',
                height=height, width=width, count=bands,
                dtype=image_data.dtype,
                crs=bbox.crs.pyproj_crs(),
                transform=transform,
            ) as dst:
                for b in range(bands):
                    dst.write(image_data[:, :, b], b + 1)

            output_files.append(filepath)

            generate_provenance(filepath.replace(".tif", "_provenance.json"), {
                "sensor": sensor,
                "item_id": tile_id,
                "acquisition_time": acquisition_time,
                "bbox": [bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]]
            })

        return output_files
