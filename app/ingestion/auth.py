"""
Authentication module for Sentinel Hub.
Responsible for managing Sentinel Hub OAuth2 token management with auto-refresh.
"""

from threading import Lock
from typing import Optional

from sentinelhub import SHConfig


class SentinelHubAuth:
    """
    Singleton class to manage Sentinel Hub configuration and authentication.
    """

    _instance: Optional["SentinelHubAuth"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "SentinelHubAuth":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SentinelHubAuth, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the SentinelHubAuth instance.
        Configures SHConfig for CDSE (Copernicus Data Space Ecosystem).
        SHConfig automatically reads SH_CLIENT_ID and SH_CLIENT_SECRET from environment variables.
        """
        if getattr(self, "_initialized", False):
            return

        self._config = SHConfig()

        # Configure for CDSE (Copernicus Data Space Ecosystem) as per project requirements
        self._config.sh_base_url = "https://sh.dataspace.copernicus.eu"
        self._config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self._config.sh_timeout = 120  # seconds — prevents silent hangs on slow responses

        self._initialized = True

    @property
    def config(self) -> SHConfig:
        """
        Returns the Sentinel Hub configuration object.

        Returns:
            SHConfig: The configured Sentinel Hub configuration.
        """
        return self._config
