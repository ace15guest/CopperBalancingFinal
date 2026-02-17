"""
Border and mask operations for array processing.

This module provides functions for detecting borders, creating masks,
and working with array boundaries.
"""

import numpy as np


def _first_true_index(arr_1d):
    """
    Return the index of the first True in a 1D boolean array.
    If none, return len(arr_1d).
    """
    idx = np.where(arr_1d)[0]
    if len(idx) == 0:
        return len(arr_1d)
    return int(idx[0])


def _last_true_index(arr_1d):
    """
    Return the index of the last True in a 1D boolean array.
    If none, return -1.
    """
    idx = np.where(arr_1d)[0]
    if len(idx) == 0:
        return -1
    return int(idx[-1])


def _mode_int_ignore_nan(arr):
    """
    Return the most common integer in arr, ignoring NaNs.
    If all NaN or empty, return 0.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0
    vals, counts = np.unique(finite.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def find_border_idx(array):
    """
    Find the bounding box of non-zero/non-NaN values in a 2D array.

    Parameters
    ----------
    array : 2D numpy array
        Input array to analyze.

    Returns
    -------
    min_row_idx : int
        First row with valid data.
    min_col_idx : int
        First column with valid data.
    max_row_idx : int
        Last row with valid data.
    max_col_idx : int
        Last column with valid data.
    """
    valid = (array != 0) & np.isfinite(array)
    
    # Row-wise: find first and last rows with any True
    row_has_valid = valid.any(axis=1)
    min_row_idx = _first_true_index(row_has_valid)
    max_row_idx = _last_true_index(row_has_valid)
    
    # Col-wise: find first and last cols with any True
    col_has_valid = valid.any(axis=0)
    min_col_idx = _first_true_index(col_has_valid)
    max_col_idx = _last_true_index(col_has_valid)
    
    return min_row_idx, min_col_idx, max_row_idx, max_col_idx


def border_mask_from_rect(shape, min_row, max_row, min_col, max_col):
    """
    Create a boolean mask that is True at the border of a rectangular region.

    The border includes:
    - All pixels in rows [0, min_row) and (max_row, shape[0])
    - All pixels in cols [0, min_col) and (max_col, shape[1])

    Parameters
    ----------
    shape : tuple of ints
        Shape of the output mask (rows, cols).
    min_row : int
        First row of the interior region.
    max_row : int
        Last row of the interior region.
    min_col : int
        First column of the interior region.
    max_col : int
        Last column of the interior region.

    Returns
    -------
    mask : 2D boolean array
        True at border positions, False in interior.
    """
    mask = np.zeros(shape, dtype=bool)
    
    # Top and bottom borders
    if min_row > 0:
        mask[:min_row, :] = True
    if max_row < shape[0] - 1:
        mask[max_row + 1:, :] = True
    
    # Left and right borders
    if min_col > 0:
        mask[:, :min_col] = True
    if max_col < shape[1] - 1:
        mask[:, max_col + 1:] = True
    
    return mask


def get_border_mask(array):
    """
    Get a boolean mask of the border region around valid data.

    This is a convenience function that combines find_border_idx
    and border_mask_from_rect.

    Parameters
    ----------
    array : 2D numpy array
        Input array to analyze.

    Returns
    -------
    mask : 2D boolean array
        True at border positions (outside the bounding box of valid data).
    """
    min_row_idx, min_col_idx, max_row_idx, max_col_idx = find_border_idx(array)
    return border_mask_from_rect(array.shape, min_row_idx, max_row_idx, min_col_idx, max_col_idx)


def center_crop_by_area(arr: np.ndarray, pct_area: float, *, round_to_even=False) -> np.ndarray:
    """
    Return a centered crop whose AREA is pct_area of the original.

    Parameters
    ----------
    arr : np.ndarray
        2D array to crop.
    pct_area : float
        Fraction of original area to keep (0 < pct_area <= 1).
    round_to_even : bool, optional
        If True, round dimensions to nearest even number.

    Returns
    -------
    cropped : np.ndarray
        Centered crop of the input array.

    Examples
    --------
    >>> arr = np.ones((100, 100))
    >>> cropped = center_crop_by_area(arr, 0.25)  # 50x50 crop
    >>> cropped.shape
    (50, 50)
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if not (0 < pct_area <= 1):
        raise ValueError("pct_area must be in (0, 1]")
    
    H, W = arr.shape
    scale = np.sqrt(pct_area)
    new_H = int(np.round(H * scale))
    new_W = int(np.round(W * scale))
    
    if round_to_even:
        new_H = (new_H // 2) * 2
        new_W = (new_W // 2) * 2
    
    # Ensure at least 1x1
    new_H = max(1, new_H)
    new_W = max(1, new_W)
    
    # Center crop
    r0 = (H - new_H) // 2
    c0 = (W - new_W) // 2
    
    return arr[r0:r0 + new_H, c0:c0 + new_W]

# Made with Bob
