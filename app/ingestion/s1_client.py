"""
Sentinel-1 Client (Role 2).
Responsible for downloading Sentinel-1 GRD (VV/VH) data and generating metadata.
"""
import os
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SentinelHubCatalog,
    CRS,
    bbox_to_dimensions,
)

from app.ingestion.auth import SentinelHubAuth
from app.ingestion.metadata import generate_provenance
from app.ingestion.retry import with_retry

logger = logging.getLogger(__name__)


@with_retry(max_attempts=3, base_delay=2.0)
def _fetch(request: SentinelHubRequest) -> list:
    return request.get_data()


class S1Client:
    """
    Client for downloading Sentinel-1 GRD data.
    """

    def __init__(self):
        """
        Initialize the S1Client.
        """
        self.auth = SentinelHubAuth()
        self.config = self.auth.config

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 10,
    ) -> List[str]:
        """
        Downloads Sentinel-1 data for the given BBox and time interval.

        Args:
            bbox: Bounding box of the area of interest.
            time_interval: Tuple of (start_date, end_date).
            resolution: Resolution in meters (default 10).

        Returns:
            List of paths to downloaded files.
        """
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "VH"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.VV, sample.VH];
        }
        """

        catalog = SentinelHubCatalog(config=self.config)
        search_iterator = catalog.search(
            collection=DataCollection.SENTINEL1_IW,
            bbox=bbox,
            time=time_interval,
        )

        output_files = []
        last_error: Optional[Exception] = None
        try:
            results = list(search_iterator)
        except Exception as e:
            logger.exception(f"S1 catalog search failed: {e}")
            return []

        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]
            orbit_id = props.get("sat:absolute_orbit")

            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "S1"

            filename = f"{date_str}_{sensor}_{tile_id}.tif"
            filepath = os.path.join("data", "raw", filename)

            req_interval = (dt, dt)

            w, h = bbox_to_dimensions(bbox, resolution=resolution)
            _MAX = 2500
            if w > _MAX or h > _MAX:
                scale = _MAX / max(w, h)
                w, h = max(1, int(w * scale)), max(1, int(h * scale))
            size = (w, h)
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL1_IW,
                        time_interval=req_interval,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF)
                ],
                bbox=bbox,
                size=size,
                config=self.config,
            )

            try:
                data_list = _fetch(request)
            except Exception as e:
                logger.exception(f"S1 download failed for {tile_id}: {e}")
                last_error = e
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
                filepath,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=bands,
                dtype=image_data.dtype,
                crs=bbox.crs.pyproj_crs(),
                transform=transform,
            ) as dst:
                for b in range(bands):
                    dst.write(image_data[:, :, b], b + 1)

            output_files.append(filepath)

            meta_filepath = filepath.replace(".tif", "_provenance.json")
            provenance = {
                "orbit_id": orbit_id,
                "acquisition_time": acquisition_time,
                "sensor": sensor,
                "bbox": [bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]],
                "item_id": item["id"],
                "properties": props
            }
            generate_provenance(meta_filepath, provenance)

        if not output_files and last_error is not None:
            raise last_error
        return output_files
