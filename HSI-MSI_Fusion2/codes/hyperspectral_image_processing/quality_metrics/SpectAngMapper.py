import numpy as np
import sys
from math import acos

def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    # shape of the image should be like this (rows, cols, bands)
    # Please note that: The interpretation of a 3-dimension array read from rasterio is: (bands, rows, columns) while
    # image processing software like scikit-image, pillow and matplotlib are generally ordered: (rows, columns, bands)
    # in order efficiently swap the axis order one can use reshape_as_raster, reshape_as_image from rasterio.plot
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )

    assert org_img.shape == pred_img.shape, msg
    
def sam(org_img: np.ndarray, pred_img: np.ndarray, convert_to_degree: bool = True):
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
    _assert_image_shapes_equal(org_img, pred_img, "SAM")

    # Spectral angles are first computed for each pair of pixels
    numerator = np.sum(np.multiply(pred_img, org_img), axis=2)
    denominator = np.linalg.norm(org_img, axis=2) * np.linalg.norm(pred_img, axis=2)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = sam_angles * 180.0 / np.pi

    # The original paper states that SAM values are expressed as radians, while e.g. Lanares
    # et al. (2018) use degrees. We therefore made this configurable, with degree the default
    return np.mean(np.nan_to_num(sam_angles))