import os
import hashlib
import logging

logger = logging.getLogger(__name__)

class IntegrityChecker:
    def __init__(self, root_dir=".", output_file="integrity_hashes.txt"):
        self.root_dir = root_dir
        self.output_file = output_file
        self.exclude_dirs = {".git", "venv", "__pycache__", ".github", ".idea", ".vscode"}

    def generate_integrity_file(self):
        logger.info("Generating integrity hashes...")
        hashes = {}

        for root, dirs, files in os.walk(self.root_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                # Skip the output file itself if it exists
                if file == self.output_file:
                    continue
                if file.endswith(".pyc"): # Skip compiled python
                    continue

                filepath = os.path.join(root, file)
                file_hash = self._calculate_sha256(filepath)
                if file_hash:
                    hashes[filepath] = file_hash

        try:
            with open(self.output_file, "w") as f:
                for filepath, h in sorted(hashes.items()):
                    f.write(f"{h}  {filepath}\n")
            logger.info(f"Integrity hashes saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to write integrity file: {e}")

    def _calculate_sha256(self, filepath):
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {filepath}: {e}")
            return None

if __name__ == "__main__":
    checker = IntegrityChecker()
    checker.generate_integrity_file()
