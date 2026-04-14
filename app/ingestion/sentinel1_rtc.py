"""
Sentinel-1 RTC Client (Role 2 - Enhanced).
Responsible for downloading Sentinel-1 GRD (VV, VH) data with Radiometric Terrain Correction.
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
    bbox_to_dimensions,
)

from app.ingestion.auth import SentinelHubAuth
from app.ingestion.metadata import generate_provenance
from app.ingestion.retry import with_retry
from app.core.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


@with_retry(max_attempts=3, base_delay=2.0)
def _fetch(request: SentinelHubRequest) -> list:
    return request.get_data()


class S1RTCClient:
    """
    Client for downloading Sentinel-1 IW GRD data with RTC.
    """

    def __init__(self):
        """
        Initialize the S1RTCClient.
        """
        self.auth = SentinelHubAuth()
        self.config = self.auth.config

        # Define DataCollection for CDSE S1 IW
        self.data_collection = DataCollection.SENTINEL1_IW.define_from(
            "CDSE_S1_IW",
            service_url="https://sh.dataspace.copernicus.eu"
        )

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 10,
    ) -> List[str]:
        """
        Downloads Sentinel-1 data.
        """
        # Evalscript for VV, VH (Linear gamma0)
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "VH", "dataMask"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.VV, sample.VH];
        }
        """

        catalog = SentinelHubCatalog(config=self.config)

        search_iterator = catalog.search(
            collection=self.data_collection,
            bbox=bbox,
            time=time_interval,
        )

        output_files = []
        last_error: Optional[Exception] = None
        try:
            results = list(search_iterator)
        except Exception as e:
            logger.exception(f"S1 catalog search failed: {e}")
            raise

        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]
            orbit_id = props.get("sat:absolute_orbit")
            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "S1_RTC"

            filename = f"{date_str}_{sensor}_{tile_id}.tif"
            filepath = os.path.join(RAW_DATA_DIR, filename)

            # Request specific time
            req_interval = (dt, dt)

            # Processing options for RTC.
            # GAMMA0_TERRAIN: radiometric terrain correction (gamma-naught).
            # demInstance COPERNICUS_30: forces the 30 m Copernicus DEM (GLO-30)
            # for terrain flattening, preventing topographic brightening on
            # sensor-facing slopes from being misinterpreted as built-up density.
            processing_options = {
                "backCoeff":   "GAMMA0_TERRAIN",
                "orthorectify": True,
                "demInstance": "COPERNICUS_30",
            }

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
                        data_collection=self.data_collection,
                        time_interval=req_interval,
                        other_args={"processing": processing_options}
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

            # Generate Metadata
            meta_filepath = filepath.replace(".tif", "_provenance.json")
            provenance = {
                "orbit_id": orbit_id,
                "acquisition_time": acquisition_time,
                "sensor": sensor,
                "bbox": [bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]],
                "item_id": tile_id,
                "processing": processing_options
            }
            generate_provenance(meta_filepath, provenance)

        if not output_files and last_error is not None:
            raise last_error
        return output_files
