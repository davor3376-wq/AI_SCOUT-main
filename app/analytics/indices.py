"""
Indices Module (Role 6).
Responsible for calculating spectral indices (NDVI, NDWI) using high-performance NumPy.
"""
import numpy as np

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Why: NDVI is a standardized index allowing you to generate an image displaying
    greenness (relative biomass). This index takes advantage of the contrast of
    the characteristics of two bands from a multispectral raster dataset:
    the chlorophyll pigment absorptions in the red band and the high reflectivity
    of plant materials in the near-infrared (NIR) band.

    Args:
        red: Red band array (e.g., B04).
        nir: Near-Infrared band array (e.g., B08).

    Returns:
        Numpy array containing NDVI values (float32), ranging from -1.0 to 1.0.
    """
    # Ensure inputs are float to allow NaNs and correct division
    red_f = red.astype(np.float32)
    nir_f = nir.astype(np.float32)

    numerator = nir_f - red_f
    denominator = nir_f + red_f

    # Initialize result with NaN or a safe value
    ndvi = np.zeros_like(numerator)

    # Avoid division by zero
    # We create a mask where denominator is not zero
    valid_mask = denominator != 0

    # Perform division only where valid
    ndvi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Where denominator is 0, usually result is undefined.
    # We can set it to NaN or 0. Since 0 usually implies no reflection,
    # but (0-0)/(0+0) is tricky.
    # Let's use NaN for invalid calculations to distinguish from valid 0.
    ndvi[~valid_mask] = np.nan

    return ndvi

def calculate_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Burn Ratio (NBR).

    NBR = (NIR - SWIR2) / (NIR + SWIR2)

    Why: NBR uses the contrast between the high reflectance of healthy vegetation
    in the NIR band (B08) and the high reflectance of burned / bare soil in the
    SWIR2 band (B12) to map fire severity and post-fire recovery. Healthy
    vegetation yields high positive NBR (≈ 0.1 – 0.9); active burn scars and
    severely burned areas yield strongly negative NBR (≈ −0.1 – −0.9).
    dNBR = pre-fire NBR − post-fire NBR is the standard severity metric (Key &
    Benson 2006; USGS Fire Effects Monitoring Program).

    Args:
        nir:   Near-Infrared band array (Sentinel-2 B08, 10 m).
        swir2: Short-Wave Infrared 2 band array (Sentinel-2 B12, 20 m —
               must be resampled to match nir resolution before passing in).

    Returns:
        Numpy array containing NBR values (float32), ranging from -1.0 to 1.0.
        Pixels where NIR + SWIR2 == 0 are set to NaN.
    """
    nir_f   = nir.astype(np.float32)
    swir2_f = swir2.astype(np.float32)

    numerator   = nir_f - swir2_f
    denominator = nir_f + swir2_f

    nbr = np.zeros_like(numerator)
    valid_mask = denominator != 0
    nbr[valid_mask]  = numerator[valid_mask] / denominator[valid_mask]
    nbr[~valid_mask] = np.nan

    return nbr


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculates the Normalized Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    Why: NDWI is used to monitor changes related to water content in water bodies.
    As water bodies strongly absorb light in visible to infrared electromagnetic spectrum,
    NDWI uses green and near-infrared bands to highlight water bodies.
    It is sensitive to built-up land and can result in over-estimation of water bodies.

    Args:
        green: Green band array (e.g., B03).
        nir: Near-Infrared band array (e.g., B08).

    Returns:
        Numpy array containing NDWI values (float32), ranging from -1.0 to 1.0.
    """
    # Ensure inputs are float
    green_f = green.astype(np.float32)
    nir_f = nir.astype(np.float32)

    numerator = green_f - nir_f
    denominator = green_f + nir_f

    # Initialize result
    ndwi = np.zeros_like(numerator)

    # Avoid division by zero
    valid_mask = denominator != 0

    # Perform division
    ndwi[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Handle invalid pixels
    ndwi[~valid_mask] = np.nan

    return ndwi
