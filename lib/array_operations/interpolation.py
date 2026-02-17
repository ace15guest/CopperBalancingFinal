"""
Interpolation and fill operations for arrays with missing data.

This module provides various methods for filling NaN values and invalid
regions in arrays, including IDW, nearest neighbor, and iterative methods.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt


def fill_nans_nd(arr, method="linear", max_iters=200, tol=1e-5, kernel_size=3):
    """
    Fill NaNs in an N-D array using:
      - method='linear'   : scipy.interpolate.griddata with nearest fallback
      - method='nearest'  : scipy.ndimage.distance_transform_edt
      - method='iterative': N-D neighbor averaging via scipy.ndimage.convolve

    Parameters
    ----------
    arr : np.ndarray
        N-D array with NaN values to fill.
    method : str, optional
        Fill method: 'linear', 'nearest', or 'iterative'.
    max_iters : int, optional
        Maximum iterations for iterative method.
    tol : float, optional
        Convergence tolerance for iterative method.
    kernel_size : int, optional
        Kernel size for iterative method (must be odd).

    Returns
    -------
    filled : np.ndarray
        Array with NaNs filled.
    """
    arr = np.asarray(arr, dtype=float)
    mask = np.isnan(arr)
    
    if not np.any(mask):
        return arr.copy()
    
    if method == "nearest":
        # Use distance transform to find nearest valid value
        indices = ndi.distance_transform_edt(mask, return_indices=True)[1]
        filled = arr.copy()
        # For each dimension, use the indices to gather values
        filled[mask] = arr[tuple(indices[:, mask])]
        return filled
    
    elif method == "iterative":
        # Iterative neighbor averaging
        filled = arr.copy()
        filled[mask] = 0.0  # Initialize NaNs to 0
        
        # Create averaging kernel
        kernel = np.ones([kernel_size] * arr.ndim, dtype=float)
        kernel /= kernel.sum()
        
        for iteration in range(max_iters):
            old = filled.copy()
            
            # Convolve to get neighbor averages
            smoothed = ndi.convolve(filled, kernel, mode='nearest')
            
            # Update only NaN positions
            filled[mask] = smoothed[mask]
            
            # Check convergence
            diff = np.abs(filled[mask] - old[mask]).max()
            if diff < tol:
                break
        
        return filled
    
    elif method == "linear":
        # Use scipy.interpolate.griddata for linear interpolation
        from scipy.interpolate import griddata
        
        # Get coordinates of valid and invalid points
        valid = ~mask
        if arr.ndim == 2:
            rows, cols = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
            points_valid = np.column_stack([rows[valid].ravel(), cols[valid].ravel()])
            points_invalid = np.column_stack([rows[mask].ravel(), cols[mask].ravel()])
            values = arr[valid]
            
            # Try linear interpolation
            try:
                filled_vals = griddata(points_valid, values, points_invalid, method='linear')
                # Fill any remaining NaNs with nearest
                still_nan = np.isnan(filled_vals)
                if np.any(still_nan):
                    filled_vals[still_nan] = griddata(
                        points_valid, values, points_invalid[still_nan], method='nearest'
                    )
            except Exception:
                # Fallback to nearest
                filled_vals = griddata(points_valid, values, points_invalid, method='nearest')
            
            filled = arr.copy()
            filled[mask] = filled_vals
            return filled
        else:
            # For N-D, fall back to nearest
            return fill_nans_nd(arr, method='nearest')
    
    else:
        raise ValueError(f"Unknown method: {method}")


def idw_fill_2d(arr: np.ndarray, mask: np.ndarray, *, threshold: float = 0.1, power: float = 2.0, k: int = 8, max_radius: float | None = None,
    treat_nan_as_invalid: bool = True, fallback: float | None = None) -> np.ndarray:
    """
    Fill masked positions in a 2D array using Inverse Distance Weighting (IDW).
    Requires SciPy (scipy.spatial.cKDTree).

    Parameters
    ----------
    arr : np.ndarray
        2D numeric array (rows x cols).
    mask : np.ndarray
        Boolean mask, same shape as `arr`. True where you want to fill.
    threshold : float, default=0.1
        Validity threshold. A cell is valid if abs(value) > threshold.
    power : float, default=2.0
        IDW power parameter (p). Larger p down-weights distant neighbors more.
    k : int, default=8
        Number of nearest neighbors to use per masked cell.
    max_radius : float or None, default=None
        Maximum neighbor distance. If None, no radius limit.
        If set, neighbors farther than `max_radius` are ignored.
    treat_nan_as_invalid : bool, default=True
        If True, NaNs are considered invalid (not used as sources; will be filled if masked).
    fallback : float or None, default=None
        If no neighbors are found (e.g., outside `max_radius`), use this value.
        If None, leave the original value unchanged.

    Returns
    -------
    out : np.ndarray
        Array with masked invalid positions filled (float dtype).

    Notes
    -----
    - If a masked cell coincides with a valid source (distance == 0),
      the source value is copied directly.
    - If some of the k neighbors are missing (KDTree returns inf distances),
      they contribute zero weight.
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if mask.shape != arr.shape:
        raise ValueError(f"mask shape {mask.shape} must match arr shape {arr.shape}")

    try:
        from scipy.spatial import cKDTree  # fast KDTree
    except Exception as e:
        raise ImportError(
            "SciPy not available. Install scipy to use idw_fill_2d."
        ) from e

    n_rows, n_cols = arr.shape
    out = arr.astype(float, copy=True)

    # Validity mask
    valid = np.abs(arr) > threshold
    if treat_nan_as_invalid:
        valid &= np.isfinite(arr)

    # Targets to fill: masked & invalid
    target = mask & (~valid)
    if not np.any(target):
        return out

    # Source points (valid)
    src_r, src_c = np.where(valid)
    if src_r.size == 0:
        # No sources available
        return out

    src_points = np.column_stack((src_r.astype(float), src_c.astype(float)))
    src_vals = arr[valid].astype(float)

    # Build KDTree
    tree = cKDTree(src_points)

    # Target points
    tgt_r, tgt_c = np.where(target)
    tgt_points = np.column_stack((tgt_r.astype(float), tgt_c.astype(float)))

    # Query k neighbors (optionally limited by max_radius)
    distance_upper_bound = np.inf if max_radius is None else float(max_radius)
    dists, idxs = tree.query(
        tgt_points,
        k=min(k, src_points.shape[0]),
        distance_upper_bound=distance_upper_bound,
        workers=-1  # use all cores if available
    )

    # Ensure 2D shape for consistency when k==1
    if k == 1 or np.ndim(dists) == 1:
        dists = dists[:, np.newaxis]
        idxs  = idxs[:,  np.newaxis]

    # Handle exact matches: any distance == 0 → copy that source
    exact_match = (dists == 0.0)
    if np.any(exact_match):
        # For rows with exact matches, take the first exact neighbor
        row_has_exact = exact_match.any(axis=1)
        rows_exact = np.where(row_has_exact)[0]
        cols_exact = exact_match[rows_exact].argmax(axis=1)  # first exact neighbor per row
        src_indices_exact = idxs[rows_exact, cols_exact]
        out[tgt_r[rows_exact], tgt_c[rows_exact]] = src_vals[src_indices_exact]

    # For the rest, compute IDW weighted average
    remaining_rows = ~exact_match.any(axis=1)
    if np.any(remaining_rows):
        rr = np.where(remaining_rows)[0]

        d = dists[rr].astype(float)
        ii = idxs[rr].astype(int)

        # Neighbors beyond radius have idx == src_points.shape[0] and dist == inf
        # Mask them out by zero weights
        with np.errstate(divide='ignore'):
            w = 1.0 / (d ** power)  # IDW weights
        w[~np.isfinite(w)] = 0.0  # handle inf/NaN (including inf distances)

        # Gather neighbor values
        v = src_vals[ii]  # shape (n_remaining, k)

        # Weighted sum / sum of weights
        wsum = w.sum(axis=1)
        # Avoid division by zero
        nonzero_w = wsum > 0
        if np.any(nonzero_w):
            est = (w[nonzero_w] * v[nonzero_w]).sum(axis=1) / wsum[nonzero_w]
            out[tgt_r[rr[nonzero_w]], tgt_c[rr[nonzero_w]]] = est

        # Rows with no neighbors (wsum == 0): apply fallback
        if np.any(~nonzero_w) and fallback is not None:
            out[tgt_r[rr[~nonzero_w]], tgt_c[rr[~nonzero_w]]] = float(fallback)
        # else: leave unchanged

    return out


def nearest_border_fill_true_2d(arr: np.ndarray, mask: np.ndarray, threshold: float = 0.0, treat_nan_as_invalid: bool = True) -> np.ndarray:
    """
    Fill positions selected by `mask` in a 2D array by copying from the nearest valid
    (non-zero per threshold) cell anywhere in 2D (including diagonals).
    Requires SciPy (distance_transform_edt). Uses Euclidian distance rather than cardinal direction.

    Parameters
    ----------
    arr : np.ndarray
        2D array (rows x cols), numeric.
    mask : np.ndarray
        Boolean mask, same shape as arr. True where you want to fill (border).
    threshold : float
        A cell is considered valid if |value| > threshold. Default 0.0 (strict non-zero).
    treat_nan_as_invalid : bool
        If True, NaNs are considered invalid.

    Returns
    -------
    out : np.ndarray
        Filled array (float).
    """
    if arr.ndim != 2:
        raise ValueError("arr must be 2D")
    if mask.shape != arr.shape:
        raise ValueError(f"mask shape {mask.shape} must match arr shape {arr.shape}")

    out = arr.astype(float, copy=True)

    # Validity mask: finite & above threshold
    valid = np.abs(arr) > threshold
    if treat_nan_as_invalid:
        valid &= np.isfinite(arr)

    # Only fill mask positions that are invalid
    target = mask & (~valid)
    if not np.any(target):
        return out
    if not np.any(valid):
        # No source values to copy from
        return out

    # EDT semantics: returns indices of nearest *zero* in the input.
    # We pass (~valid) so that valid cells become zeros in this input.
    indices = ndi.distance_transform_edt(~valid, return_indices=True)[1]  # (i_idx, j_idx)
    gather = (indices[0][target], indices[1][target])
    out[target] = arr[gather]
    return out


def fill_border_with_percent_max(arr: np.ndarray, mask: np.ndarray, percent: float) -> np.ndarray:
    """
    Fill all positions where `mask` is True with `percent * max(arr)`.
    
    Parameters
    ----------
    arr : np.ndarray
        2D numeric array (rows x cols).
    mask : np.ndarray
        Boolean array, same shape as `arr`. True indicates border positions to fill.
    percent : float
        Fraction of the array maximum to use (e.g., 0.2 for 20%).

    Returns
    -------
    out : np.ndarray
        A copy of `arr` with masked positions set to percent * max(arr).
    """
    if arr.ndim != 2:
        raise ValueError("arr must be a 2D array")
    if mask.shape != arr.shape:
        raise ValueError(f"mask shape {mask.shape} must match arr shape {arr.shape}")
    if not (np.isfinite(percent) and percent >= 0):
        raise ValueError("percent must be a non-negative finite number")

    out = arr.astype(float, copy=True)
    # Use finite max; if all NaN, default to 0
    finite_vals = out[np.isfinite(out)]
    arr_max = finite_vals.max() if finite_vals.size > 0 else 0.0

    fill_value = percent * arr_max
    out[mask] = fill_value
    return out

# Made with Bob
