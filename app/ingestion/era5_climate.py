"""
ERA5 Climate Client (Ingestion).
Responsible for downloading ERA5 reanalysis data.
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


class Era5Client:
    """
    Client for downloading ERA5 data.
    """

    def __init__(self):
        self.auth = SentinelHubAuth()
        self.config = self.auth.config
        # ERA5 is available as a BYOC-like collection or standard.
        # Use standard definition.
        # DataCollection.ERA5 might not be in the enum, so we define it.
        # ERA5 ID on Sentinel Hub is usually 'era5'.
        self.data_collection = DataCollection.define("ERA5", api_id="era5", service_url="https://services.sentinel-hub.com")

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 10000, # ERA5 is coarse (~30km)
    ) -> List[str]:
        """
        Downloads ERA5 data.
        """
        # Variables: Temperature (2m), Total Precipitation
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["temperature_2m", "total_precipitation"],
            output: { bands: 2, sampleType: "FLOAT32" }
          };
        }

        function evaluatePixel(sample) {
          return [sample.temperature_2m, sample.total_precipitation];
        }
        """

        # ERA5 usually doesn't need catalog search for individual scenes like optical.
        # It's a continuous dataset.
        # But we still iterate by time steps if we want daily files.
        # However, for ERA5 we might just request the interval.
        # Let's assume we want one file per request interval.

        start_date, end_date = time_interval

        # Normalise: accept both str and datetime
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        date_str = start_date.strftime("%Y%m%d")
        sensor = "ERA5"
        filename = f"{date_str}_{sensor}_climate.tif"
        filepath = os.path.join("data", "raw", filename)

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.data_collection,
                    time_interval=time_interval,
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
            logger.exception(f"ERA5 request failed: {e}")
            return []

        if not data_list:
            return []

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
            dst.write(image_data[:, :, 0], 1) # Temp
            dst.write(image_data[:, :, 1], 2) # Precip

        generate_provenance(filepath.replace(".tif", "_provenance.json"), {
            "sensor": sensor,
            "time_interval": [str(start_date), str(end_date)],
            "bbox": [bbox.lower_left[0], bbox.lower_left[1], bbox.upper_right[0], bbox.upper_right[1]],
            "bands": ["temperature_2m", "total_precipitation"]
        })

        return [filepath]
