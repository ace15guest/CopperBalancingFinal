"""
Basic array utility functions.

This module contains general-purpose array manipulation utilities that don't
fit into more specific categories like alignment, interpolation, etc.
"""

import numpy as np
from skimage.transform import resize
from typing import Dict
from src.config import Config


def shrink_array(arr, out_shape, order=1):
    """
    Resize an array to out_shape.
    Good for downsampling when there are no NaNs.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    out_shape : (H, W)
        Target shape.
    order : int
        Interpolation order: 1=bilinear (smooth), 3=bicubic (sharper).

    Returns
    -------
    np.ndarray
        Resized array with values scaled to preserve range.
    """
    return resize(
        arr, out_shape,
        order=order,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(arr.dtype)


def rescale_to_shared_minmax(calculated_array, akro_array):
    """
    Affinely rescale A and B so both end up with the same min/max:
      min(A') = min(B') = min(A,B)
      max(A') = max(B') = max(A,B)
    
    Parameters
    ----------
    calculated_array : np.ndarray
        First array to rescale.
    akro_array : np.ndarray
        Second array to rescale (used as reference for min/max).
    
    Returns
    -------
    a2 : np.ndarray
        Rescaled calculated_array.
    b2 : np.ndarray
        Rescaled akro_array.
    minmax : tuple
        (global_min, global_max) used for rescaling.
    """
    calculated_array = np.asarray(calculated_array, dtype=np.float64)
    akro_array = np.asarray(akro_array, dtype=np.float64)

    gmin = np.nanmin(akro_array)
    gmax = np.nanmax(akro_array)
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise ValueError("Inputs contain no finite values.")
    if gmax == gmin:
        raise ValueError("Global range is zero; cannot rescale.")

    def scale(x):
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if xmax == xmin:
            raise ValueError("One input has zero range; cannot map to a nonzero global range.")
        return (x - xmin) / (xmax - xmin) * (gmax - gmin) + gmin

    a2 = scale(calculated_array)
    b2 = scale(akro_array)
    return a2, b2, (gmin, gmax)


def multiple_layers_weighted(layer_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Weighted sum of 2D arrays using copper weight parsed from key names.
    Keys must contain a token like '..._1oz_...', '..._0.5oz_...', '..._0_5oz_...'.

    Args:
        layer_dict: {name: 2D np.ndarray}

    Returns:
        2D np.ndarray: weighted sum cropped to the smallest (rows, cols).

    Raises:
        ValueError: if any key lacks a parsable '<number>oz' token or arrays aren't 2D.
    
    Examples
    --------
    >>> layers = {
    ...     'top_1oz': np.ones((100, 100)),
    ...     'bottom_0.5oz': np.ones((100, 100)) * 2
    ... }
    >>> result = multiple_layers_weighted(layers)
    >>> result.shape
    (100, 100)
    """
    if not layer_dict:
        raise ValueError("layer_dict is empty.")

    # Smallest common shape
    min_rows = min(arr.shape[0] for arr in layer_dict.values())
    min_cols = min(arr.shape[1] for arr in layer_dict.values())

    out = np.zeros((min_rows, min_cols), dtype=np.float32)
    missing = []

    for name, arr in layer_dict.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"Layer '{name}' must be a 2D numpy array.")
        m = Config._OZ_RE.search(name)
        if not m:
            missing.append(name)
            continue
        w = float(m.group(1).replace("_", "."))  # "0_5" -> 0.5
        out += w * arr[:min_rows, :min_cols].astype(np.float32, copy=False)

    if missing:
        raise ValueError("Could not parse copper weight from keys: " + ", ".join(missing))

    return out

# Made with Bob
