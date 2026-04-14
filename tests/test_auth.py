import os
import unittest
from unittest.mock import patch
import sys

# Ensure the root directory is in sys.path
sys.path.append(os.getcwd())

from app.ingestion.auth import SentinelHubAuth

class TestSentinelHubAuth(unittest.TestCase):

    def setUp(self):
        # Reset singleton instance for each test
        SentinelHubAuth._instance = None

    @patch.dict(os.environ, {"SH_CLIENT_ID": "test_id", "SH_CLIENT_SECRET": "test_secret"})
    def test_singleton_config(self):
        auth1 = SentinelHubAuth()
        auth2 = SentinelHubAuth()

        self.assertIs(auth1, auth2)

        config = auth1.config
        self.assertEqual(config.sh_client_id, "test_id")
        self.assertEqual(config.sh_client_secret, "test_secret")
        self.assertEqual(config.sh_base_url, "https://sh.dataspace.copernicus.eu")
        self.assertEqual(config.sh_token_url, "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token")

if __name__ == '__main__':
    unittest.main()
