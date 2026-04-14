"""
Metadata module for generating provenance logs.
Responsible for saving metadata and provenance information for downloaded data.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np


class MetadataJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder to handle datetime and numpy types.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_provenance(output_path: str, provenance_data: Dict[str, Any]) -> None:
    """
    Generates a provenance JSON file.

    Args:
        output_path: The full path to the output JSON file.
        provenance_data: Dictionary containing the metadata/provenance information.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add processing timestamp if not present
    if "processing_timestamp" not in provenance_data:
        provenance_data["processing_timestamp"] = datetime.now(timezone.utc)

    with open(output_path, "w") as f:
        json.dump(provenance_data, f, cls=MetadataJSONEncoder, indent=4)
