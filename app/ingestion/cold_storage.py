"""
Cold Storage Manager (Infrastructure).
Responsible for archiving data to long-term storage (S3/Vault).
"""
import os
import logging
import shutil

logger = logging.getLogger(__name__)


class ColdStorageManager:
    """
    Manages cold storage.
    Currently mocks S3 by moving files to a 'vault' directory.
    """

    def __init__(self, vault_path: str = "data/vault"):
        self.vault_path = vault_path
        os.makedirs(self.vault_path, exist_ok=True)

    def archive_file(self, filepath: str) -> str:
        """
        Archives a file to the vault.

        Args:
            filepath: Path to the file to archive.

        Returns:
            Path to the archived file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        filename = os.path.basename(filepath)
        dest_path = os.path.join(self.vault_path, filename)

        shutil.copy2(filepath, dest_path)
        logger.info(f"Archived {filename} to {dest_path}")

        return dest_path

    def retrieve_file(self, filename: str, restore_path: str) -> str:
        """
        Retrieves a file from the vault.
        """
        src_path = os.path.join(self.vault_path, filename)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"File not found in vault: {filename}")

        os.makedirs(os.path.dirname(restore_path), exist_ok=True)
        shutil.copy2(src_path, restore_path)

        return restore_path
