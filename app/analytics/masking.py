"""
Masking Module (Role 8).
Responsible for masking clouds, shadows, and snow using the SCL band.
"""
import numpy as np

def get_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """
    Generates a boolean mask for clouds, shadows, and snow from the SCL band.

    SCL (Scene Classification Layer) values:
    0: No Data
    1: Saturated or Defective
    2: Dark Area Pixels
    3: Cloud Shadows
    4: Vegetation
    5: Bare Soils
    6: Water
    7: Unclassified
    8: Cloud Medium Probability
    9: Cloud High Probability
    10: Thin Cirrus
    11: Snow

    Args:
        scl: A numpy array containing SCL values.

    Returns:
        A boolean numpy array where True indicates a masked pixel (cloud, shadow, snow, etc).
    """
    # Define the values to mask
    # 3: Cloud Shadows
    # 8: Cloud Medium
    # 9: Cloud High
    # 10: Cirrus
    # 11: Snow
    # We also mask 0 (No Data) and 1 (Saturated) as they are not valid observations.
    masked_values = [0, 1, 3, 8, 9, 10, 11]

    # Create the mask
    # np.isin returns a boolean array of the same shape as scl
    mask = np.isin(scl, masked_values)

    return mask

def get_cloud_mask_from_qa60(qa60: np.ndarray) -> np.ndarray:
    """
    Generates a boolean mask from the QA60 band.

    Bit 10: Opaque clouds (1024)
    Bit 11: Cirrus clouds (2048)
    """
    # Check for bit 10 or bit 11
    # We use bitwise AND
    opaque_clouds = (qa60 & (1 << 10)) > 0
    cirrus_clouds = (qa60 & (1 << 11)) > 0

    return opaque_clouds | cirrus_clouds

def calculate_cloud_coverage(mask: np.ndarray) -> float:
    """
    Calculates the percentage of masked pixels.

    Args:
        mask: A boolean numpy array where True indicates a masked pixel.

    Returns:
        Float representing the percentage (0.0 to 100.0) of masked pixels.
    """
    if mask.size == 0:
        return 0.0

    return (np.count_nonzero(mask) / mask.size) * 100.0
