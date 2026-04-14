import pytest
import os
import json
import numpy as np
from unittest.mock import MagicMock, patch
from app.supervisor import load_tasks, check_alpha_health

def test_load_tasks(tmp_path):
    tasks_file = tmp_path / "tasks.json"
    data = {
        "tasks": [
            {
                "name": "TestTask",
                "bbox": [10, 20, 30, 40]
            }
        ]
    }
    with open(tasks_file, 'w') as f:
        json.dump(data, f)

    with patch("app.supervisor.TASK_FILE", str(tasks_file)):
        tasks = load_tasks()
        assert len(tasks) == 1
        assert tasks[0]["name"] == "TestTask"

def test_check_alpha_health_success(tmp_path):
    # Create dummy tif
    tif_path = tmp_path / "test.tif"
    with open(tif_path, 'wb') as f:
        f.write(b"dummy")

    # Create dummy metadata
    meta_path = tmp_path / "test_provenance.json"
    bbox = [10.0, 20.0, 30.0, 40.0]
    meta = {"bbox": bbox}
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    with patch("rasterio.open") as mock_open:
        mock_src = MagicMock()
        # Return data with mean > 100
        mock_src.read.return_value = np.array([[200, 200], [200, 200]])
        mock_open.return_value.__enter__.return_value = mock_src

        result = check_alpha_health(str(tif_path), bbox)
        assert result is True

def test_check_alpha_health_fail_mean(tmp_path):
    tif_path = tmp_path / "test.tif"
    with open(tif_path, 'wb') as f:
        f.write(b"dummy")

    # Metadata ok
    meta_path = tmp_path / "test_provenance.json"
    bbox = [10.0, 20.0, 30.0, 40.0]
    with open(meta_path, 'w') as f:
        json.dump({"bbox": bbox}, f)

    with patch("rasterio.open") as mock_open:
        mock_src = MagicMock()
        # Return data with mean <= 100
        mock_src.read.return_value = np.array([[50, 50], [50, 50]])
        mock_open.return_value.__enter__.return_value = mock_src

        result = check_alpha_health(str(tif_path), bbox)
        assert result is False

def test_check_alpha_health_fail_bbox(tmp_path):
    tif_path = tmp_path / "test.tif"
    with open(tif_path, 'wb') as f:
        f.write(b"dummy")

    # Metadata mismatch
    meta_path = tmp_path / "test_provenance.json"
    bbox = [0.0, 0.0, 1.0, 1.0] # Mismatch
    with open(meta_path, 'w') as f:
        json.dump({"bbox": bbox}, f)

    with patch("rasterio.open") as mock_open:
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[200, 200], [200, 200]])
        mock_open.return_value.__enter__.return_value = mock_src

        expected_bbox = [10.0, 20.0, 30.0, 40.0]
        result = check_alpha_health(str(tif_path), expected_bbox)
        assert result is False
