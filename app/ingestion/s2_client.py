"""
Sentinel-2 Client (Role 3).
Responsible for downloading Sentinel-2 L2A (B04, B08, SCL) data and generating metadata.
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


class S2Client:
    """
    Client for downloading Sentinel-2 L2A data.
    """

    def __init__(self):
        """
        Initialize the S2Client.
        """
        self.auth = SentinelHubAuth()
        self.config = self.auth.config

        # Define DataCollection for CDSE as per requirements
        self.data_collection = DataCollection.SENTINEL2_L2A.define_from(
            "CDSE_S2_L2A",
            service_url="https://sh.dataspace.copernicus.eu"
        )

    def download_data(
        self,
        bbox: BBox,
        time_interval: Tuple[Union[str, datetime], Union[str, datetime]],
        resolution: int = 10,
    ) -> List[str]:
        """
        Downloads Sentinel-2 data for the given BBox and time interval.

        Args:
            bbox: Bounding box of the area of interest.
            time_interval: Tuple of (start_date, end_date).
            resolution: Resolution in meters (default 10).

        Returns:
            List of paths to downloaded files.
        """
        # Evalscript for B03, B04, B08, B12, SCL
        # B03: Green (for NDWI)
        # B04: Red (for NDVI)
        # B08: NIR (for NDVI / NDWI / NBR)
        # B12: SWIR2 at 20 m, resampled to 10 m by Sentinel Hub (for NBR)
        # SCL: Scene Classification Layer (for cloud masking)
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B03", "B04", "B08", "B12", "SCL"],
            output: { bands: 5, sampleType: "UINT16" }
          };
        }

        function evaluatePixel(sample) {
          return [
            sample.B03 * 10000,
            sample.B04 * 10000,
            sample.B08 * 10000,
            sample.B12 * 10000,
            sample.SCL
          ];
        }
        """

        # Use Catalog to get available scenes and metadata
        catalog = SentinelHubCatalog(config=self.config)

        # Search using the CDSE collection definition if possible,
        # or use standard collection for catalog search if compatible?
        # SentinelHubCatalog usually works with standard collections or collection IDs.
        # DataCollection.SENTINEL2_L2A should map to the correct collection ID "sentinel-2-l2a".
        # The define_from changes the service URL, which is crucial for download.
        # For Catalog search, the config should already handle the base URL if set correctly.
        # Let's try using the self.data_collection.

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
            logger.exception(f"S2 catalog search failed: {e}")
            raise

        for item in results:
            props = item["properties"]
            acquisition_time = props["datetime"]

            # Additional S2 metadata
            orbit_id = props.get("sat:absolute_orbit") # Might be null or different key
            cloud_cover = props.get("eo:cloud_cover")

            tile_id = item["id"]

            dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y%m%d")
            sensor = "S2"

            filename = f"{date_str}_{sensor}_{tile_id}.tif"
            filepath = os.path.join(RAW_DATA_DIR, filename)

            # Request specific time
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
                        data_collection=self.data_collection,
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
                logger.exception(f"S2 download failed for {tile_id}: {e}")
                last_error = e
                continue

            if not data_list:
                continue

            image_data = data_list[0]
            height, width, bands = image_data.shape

            # SCL is usually UINT8, but B04/B08 are UINT16.
            # Output sampleType is UINT16, so SCL will be cast to UINT16.
            # This is fine.

            transform = from_bounds(
                bbox.lower_left[0], bbox.lower_left[1],
                bbox.upper_right[0], bbox.upper_right[1],
                width, height
            )

            # Ensure dir exists
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
                "item_id": item["id"],
                "cloud_cover": cloud_cover,
                "properties": props
            }
            generate_provenance(meta_filepath, provenance)

        if not output_files and last_error is not None:
            raise last_error
        return output_files
