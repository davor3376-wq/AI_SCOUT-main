"""
MODIS Daily Client (Ingestion).
Responsible for downloading MODIS MCD43A4 data.
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


class ModisClient:
    """
    Client for downloading MODIS data.
    """

    def __init__(self):
        self.auth = SentinelHubAuth()
        self.config = self.auth.config
        self.data_collection = DataCollection.MODIS

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 500,
    ) -> List[str]:
        """
        Downloads MODIS data.
        """
        # MODIS Bands: B01 (Red), B04 (Green), B03 (Blue) for True Color (roughly)
        # Actually MODIS bands are:
        # B1: Red, B2: NIR, B3: Blue, B4: Green
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B01", "B04", "B03"],
            output: { bands: 3, sampleType: "UINT16" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.B01, sample.B04, sample.B03];
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
            logger.exception(f"MODIS catalog search failed: {e}")
            results = []

        # If catalog returns nothing (MODIS might not be indexed same way in all endpoints),
        # we might need to handle it.

        output_files = []
        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]
            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "MODIS"

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
                logger.exception(f"MODIS download failed for tile {tile_id}: {e}")
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
