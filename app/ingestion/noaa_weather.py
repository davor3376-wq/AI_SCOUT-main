"""
NOAA Weather Client (Ingestion).
Responsible for fetching weather data from NOAA APIs (e.g., GFS, NAM).
"""
import os
import logging
import requests
from datetime import datetime
from typing import List, Tuple, Union, Dict, Any
from sentinelhub import BBox

logger = logging.getLogger(__name__)


class NoaaWeatherClient:
    """
    Client for fetching NOAA weather data.
    """

    BASE_URL = "https://api.weather.gov" # Example endpoint

    def __init__(self):
        pass

    def get_forecast(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetches forecast for a point.
        """
        # 1. Get grid point
        points_url = f"{self.BASE_URL}/points/{lat},{lon}"
        try:
            resp = requests.get(points_url)
            if resp.status_code != 200:
                logger.warning(f"NOAA API error {resp.status_code} for {points_url}")
                return {}
            data = resp.json()
            properties = data.get("properties", {})
            forecast_url = properties.get("forecast")

            if not forecast_url:
                return {}

            # 2. Get forecast
            forecast_resp = requests.get(forecast_url)
            if forecast_resp.status_code != 200:
                return {}

            return forecast_resp.json()

        except Exception as e:
            logger.exception(f"Failed to fetch NOAA data: {e}")
            return {}

    def download_data(self, bbox: BBox, time_interval: Tuple[datetime, datetime]) -> List[str]:
        """
        Placeholder for downloading raster weather data (e.g., GRIB files).
        In a real scenario, this would interface with a NOAA archive or distribution server.
        """
        logger.info("NOAA raster download not yet implemented — fetching point forecast for AOI centre.")

        center_lat = (bbox.lower_left[1] + bbox.upper_right[1]) / 2
        center_lon = (bbox.lower_left[0] + bbox.upper_right[0]) / 2

        forecast = self.get_forecast(center_lat, center_lon)

        # Save forecast as JSON
        if forecast:
            start_str = time_interval[0].strftime("%Y%m%d")
            filename = f"{start_str}_NOAA_forecast.json"
            filepath = os.path.join("data", "raw", filename)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            import json
            with open(filepath, 'w') as f:
                json.dump(forecast, f, indent=2)

            return [filepath]
        return []
