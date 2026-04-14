"""
Cache Manager (Infrastructure).
Responsible for managing local caches of thumbnails and metadata.
"""
import os
import shutil
from typing import Optional

class CacheManager:
    """
    Manages file cache.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_path(self, key: str) -> str:
        """Returns the cache path for a key."""
        return os.path.join(self.cache_dir, key)

    def exists(self, key: str) -> bool:
        """Checks if key exists in cache."""
        return os.path.exists(self.get_path(key))

    def store(self, key: str, source_path: str):
        """Stores a file in cache."""
        dest = self.get_path(key)
        shutil.copy2(source_path, dest)

    def retrieve(self, key: str) -> Optional[str]:
        """Retrieves a file path if it exists."""
        if self.exists(key):
            return self.get_path(key)
        return None

    def clear(self):
        """Clears the cache."""
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
