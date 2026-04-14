"""
Commercial Planet Client (Ingestion).
Responsible for downloading PlanetScope data via Sentinel Hub BYOC or TPDI.
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
from app.ingestion.retry import with_retry

logger = logging.getLogger(__name__)


@with_retry(max_attempts=3, base_delay=2.0)
def _fetch(request: SentinelHubRequest) -> list:
    return request.get_data()


class PlanetClient:
    """
    Client for downloading PlanetScope data.
    """

    def __init__(self, collection_id: str = None):
        """
        Initialize Planet Client.

        Args:
            collection_id: The BYOC collection ID for the Planet subscription.
                           If None, checks env var PLANET_COLLECTION_ID.
        """
        self.auth = SentinelHubAuth()
        self.config = self.auth.config

        self.collection_id = collection_id or os.environ.get("PLANET_COLLECTION_ID")

        if not self.collection_id:
            # Placeholder or raise error
            logger.warning("No Planet Collection ID provided.")
            self.data_collection = None
        else:
            self.data_collection = DataCollection.define_byoc(self.collection_id)

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 3, # PlanetScope is ~3m
    ) -> List[str]:
        """
        Downloads Planet data.
        """
        if not self.data_collection:
            logger.warning("No Planet Collection ID configured.")
            return []

        # PlanetScope Bands: Blue, Green, Red, NIR
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["Blue", "Green", "Red", "NIR"],
            output: { bands: 4, sampleType: "UINT16" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.Blue, sample.Green, sample.Red, sample.NIR];
        }
        """

        catalog = SentinelHubCatalog(config=self.config)
        try:
            search_iterator = catalog.search(
                collection=self.data_collection,
                bbox=bbox,
                time=time_interval,
            )
            results = list(search_iterator)
        except Exception as e:
            logger.exception(f"Planet catalog search failed: {e}")
            results = []

        output_files = []
        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]
            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "PLANET"

            filename = f"{date_str}_{sensor}_{tile_id}.tif"
            filepath = os.path.join("data", "raw", filename)

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=self.data_collection,
                        time_interval=(dt, dt),
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
                data_list = _fetch(request)
            except Exception as e:
                logger.exception(f"Planet download failed for tile {tile_id}: {e}")
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
