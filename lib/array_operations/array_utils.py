import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate, rescale
from skimage.registration import phase_cross_correlation
from scipy.ndimage import distance_transform_edt
import scipy.ndimage as ndi
from skimage.transform import SimilarityTransform, warp, rotate, rescale
from src.config import Config
from typing import Dict
# ----------------- Helpers -----------------

def sanitize(a):
    """Ensure finite float32 array [0..1]."""
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    amin, amax = a.min(), a.max()
    if np.isfinite(amax) and amax > 0:
        a = a / amax
    return a


def binarize_robust(a, sigma=1.0):
    """Blur + Otsu threshold, safe on flat images."""
    a = sanitize(a)
    a = gaussian_filter(a, sigma=sigma)
    if not np.isfinite(a).any() or np.all(a == a.flat[0]):
        return (a > 0).astype(np.float32)
    try:
        t = threshold_otsu(a)
    except Exception:
        t = np.nanmedian(a) if np.isfinite(a).any() else 0.0
    return (a >= t).astype(np.float32)


def dominant_angle_safe(binary):
    """Estimate orientation from image moments; safe on sparse data."""
    y, x = np.nonzero(binary)
    if len(x) < 100:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eig(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def auto_scale(ref, mov, scales, preblur=1.0):
    """Pick scale that maximizes NCC vs ref."""
    ref_b = binarize_robust(ref, sigma=preblur)
    best = (1.0, -np.inf)
    for s in scales:
        m = rescale(mov, s, anti_aliasing=True, preserve_range=True)
        m_b = binarize_robust(m, sigma=preblur)
        H = min(ref_b.shape[0], m_b.shape[0])
        W = min(ref_b.shape[1], m_b.shape[1])
        if H < 32 or W < 32:
            continue
        rb = ref_b[:H, :W] - ref_b[:H, :W].mean()
        mb = m_b[:H, :W] - m_b[:H, :W].mean()
        denom = (np.linalg.norm(rb) * np.linalg.norm(mb) + 1e-8)
        ncc = float((rb * mb).sum() / denom)
        if ncc > best[1]:
            best = (s, ncc)
    return best[0]


def phase_translate(ref, mov):
    """Return (dy, dx) shift to align mov onto ref."""
    if ref.size == 0 or mov.size == 0:
        return 0.0, 0.0
    try:
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
        return float(shift[0]), float(shift[1])
    except Exception:
        return 0.0, 0.0


# ----------------- Main aligner -----------------

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import rotate, rescale, SimilarityTransform, warp
from skimage.registration import phase_cross_correlation

def align_dat_to_gerber(
    gerber_arr,
    dat_arr,
    px_per_mm_gerber=None,
    px_per_mm_dat=None,
    scale_search=(0.7, 1.4, 21),
    flip_x=False,
    flip_y=False,
    # NEW: hole handling
    fill_mode="edge",        # 'edge' | 'reflect' | 'constant'
    fill_cval=np.nan,        # used only if fill_mode == 'constant'
    return_valid_mask=True,
):
    """
    Align DAT -> Gerber by rotation+scale+translation and resample directly
    onto the Gerber canvas WITHOUT zero padding.

    fill_mode:
      - 'edge'     : extend nearest edge values (no zeros, no NaNs)
      - 'reflect'  : reflect about the border
      - 'constant' : fill with fill_cval (e.g., np.nan to mask later)

    If return_valid_mask=True, also returns a boolean mask of pixels that
    come from valid DAT samples (True) vs extrapolated/fill (False).
    """
    # ---- helpers ----
    def sanitize(a):
        a = np.asarray(a, dtype=np.float32)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    def binarize_robust(a, sigma=1.0):
        a = sanitize(a)
        a = gaussian_filter(a, sigma=sigma)
        if not np.isfinite(a).any() or np.all(a == a.flat[0]):
            return (a > 0).astype(np.float32)
        try:
            t = threshold_otsu(a)
        except Exception:
            t = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
        return (a >= t).astype(np.float32)

    def dominant_angle_safe(binary):
        y, x = np.nonzero(binary)
        if len(x) < 100:
            return 0.0
        x = x - x.mean(); y = y - y.mean()
        cov = np.cov(np.vstack([x, y]))
        if not np.all(np.isfinite(cov)):
            return 0.0
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)]
        return float(np.degrees(np.arctan2(v[1], v[0])))

    def auto_scale(ref, mov, scales, preblur=1.0):
        ref_b = binarize_robust(ref, sigma=preblur)
        best = (1.0, -np.inf)
        for s in scales:
            m = rescale(mov, s, anti_aliasing=True, preserve_range=True)
            m_b = binarize_robust(m, sigma=preblur)
            H = min(ref_b.shape[0], m_b.shape[0])
            W = min(ref_b.shape[1], m_b.shape[1])
            if H < 32 or W < 32:
                continue
            rb = ref_b[:H, :W]; mb = m_b[:H, :W]
            rb0 = rb - rb.mean(); mb0 = mb - mb.mean()
            denom = (np.linalg.norm(rb0) * np.linalg.norm(mb0) + 1e-8)
            ncc = float((rb0 * mb0).sum() / denom)
            if ncc > best[1]:
                best = (s, ncc)
        return best[0]

    def phase_translate(ref, mov):
        try:
            shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10)
            return float(shift[0]), float(shift[1])
        except Exception:
            return 0.0, 0.0

    # ---- inputs ----
    G = sanitize(gerber_arr)
    D = sanitize(dat_arr)
    if flip_x: D = np.fliplr(D)
    if flip_y: D = np.flipud(D)
    if not np.any(G): raise ValueError("Gerber array has no finite data.")
    if not np.any(D): raise ValueError("DAT array has no finite data.")

    # ---- orientation + scale ----
    angG = dominant_angle_safe(binarize_robust(G))
    angD = dominant_angle_safe(binarize_robust(D))
    rot_needed = angG - angD

    # IMPORTANT: rotate with mode + cval so we don't introduce zeros
    Drot = rotate(D, angle=rot_needed, resize=True, preserve_range=True, order=1, mode=fill_mode, cval=fill_cval)

    if px_per_mm_gerber and px_per_mm_dat:
        scale = float(px_per_mm_gerber) / float(px_per_mm_dat)
    else:
        smin, smax, n = scale_search
        scales = np.linspace(smin, smax, n)
        scale = auto_scale(G, Drot, scales)

    Drs = rescale(Drot, scale, anti_aliasing=True, preserve_range=True, order=1)

    # ---- translation on overlap ----
    H = min(G.shape[0], Drs.shape[0])
    W = min(G.shape[1], Drs.shape[1])
    Gc = gaussian_filter(G[:H, :W], 1.0)
    Dc = gaussian_filter(Drs[:H, :W], 1.0)
    dy, dx = phase_translate(Gc, Dc)

    # ---- compose translation as a warp (no manual paste) ----
    # Use warp(..., mode=fill_mode, cval=fill_cval) so we never insert zeros.
    tform = SimilarityTransform(scale=1.0, rotation=0.0, translation=(dx, dy))
    aligned = warp(
        Drs, inverse_map=tform.inverse, output_shape=G.shape,
        order=1, mode=fill_mode, cval=fill_cval, preserve_range=True
    ).astype(np.float32)

    # ---- optional validity mask (where pixels truly came from DAT) ----
    valid_mask = None
    if return_valid_mask:
        ones = np.ones_like(D, dtype=np.float32)
        ones_rot = rotate(
            ones, angle=rot_needed, resize=True, preserve_range=True,
            order=0, mode='constant', cval=0.0)
        ones_rs = rescale(ones_rot, scale, anti_aliasing=False, preserve_range=True, order=0)
        mask = warp(
            ones_rs, inverse_map=tform.inverse, output_shape=G.shape,
            order=0, mode='constant', cval=0.0, preserve_range=True
        )
        valid_mask = (mask > 0.5)

    params = {
        "rotation_deg": float(rot_needed),
        "scale": float(scale),
        "shift": (float(dy), float(dx)),
        "flip_x": bool(flip_x),
        "flip_y": bool(flip_y),
    }

    if return_valid_mask:
        return aligned, params, valid_mask
    else:
        return aligned, params

def apply_alignment(mov_arr, params, out_shape, order=1, cval=0.0):
    """
    Apply a previously computed Gerber↔DAT alignment to another array.

    Parameters
    ----------
    mov_arr : 2D numpy array
        The "moving" dataset you want to transform (e.g. another DAT layer).
        This should be in the *same native coordinate system* as the one you
        originally aligned with `align_dat_to_gerber`.

    params : dict
        The dictionary returned by `align_dat_to_gerber`. Expected keys are:
          - 'rotation_deg' : float
              Rotation angle in degrees applied to DAT → Gerber.
          - 'scale' : float
              Scale factor applied to DAT → Gerber.
          - 'shift' : (dy, dx)
              Translation (row shift, col shift) needed to align to Gerber.
              Positive dy = down, positive dx = right.
          - 'flip_x' : bool (optional)
              Whether a left-right mirror was applied during alignment.
          - 'flip_y' : bool (optional)
              Whether an up-down mirror was applied during alignment.

    out_shape : tuple of ints
        Desired output shape (rows, cols). Usually you pass
        `gerber_arr.shape` so the aligned result is on the Gerber canvas.

    order : int, optional (default=1)
        Interpolation order used by `skimage.transform.warp`.
        0=nearest, 1=bilinear, 3=cubic. Higher = smoother but slower.

    cval : float, optional (default=0.0)
        Fill value used for areas that move outside the canvas after transform.

    Returns
    -------
    aligned : 2D numpy array, dtype=float32
        The input `mov_arr`, transformed by the rotation/scale/translation
        stored in `params` and resampled onto `out_shape`.

    Notes
    -----
    - Use this to apply *the exact same alignment* you computed once with
      `align_dat_to_gerber` to other arrays from the same DAT dataset.
    - That way all your DAT layers line up with the Gerber reference.
    """

    a = np.asarray(mov_arr, dtype=np.float32)
    p = {**{"rotation_deg":0.0,"scale":1.0,"shift":(0.0,0.0),
            "flip_x":False,"flip_y":False}, **params}

    # Apply flips if required
    if p["flip_x"]:
        a = np.fliplr(a)
    if p["flip_y"]:
        a = np.flipud(a)

    # Apply rotation (with resize so nothing gets cropped)
    if abs(p["rotation_deg"]) > 1e-6:
        a = rotate(a, angle=p["rotation_deg"], resize=True, preserve_range=True, order=order, cval=cval)

    # Apply scaling
    if abs(p["scale"] - 1.0) > 1e-6:
        a = rescale(a, p["scale"], anti_aliasing=True,
                    preserve_range=True, order=order)

    # Apply translation
    dy, dx = p["shift"]
    tform = SimilarityTransform(scale=1.0, rotation=0.0, translation=(dx, dy))
    aligned = warp(a, inverse_map=tform.inverse, output_shape=out_shape,
                   order=order, cval=cval, preserve_range=True)
    return aligned.astype(np.float32)

import numpy as np

def fill_nans_nd(arr, method="linear", max_iters=200, tol=1e-5, kernel_size=3):
    """
    Fill NaNs in an N-D array using:
      - method='linear'   : scipy.interpolate.griddata with nearest fallback
      - method='nearest'  : scipy.ndimage.distance_transform_edt
      - method='iterative': N-D neighbor averaging via scipy.ndimage.convolve

    Parameters
    ----------
    arr : np.ndarray, shape (D0, D1, ..., Dn)
        N-D array with NaNs to fill.
    method : {'linear','iterative','nearest'}
    max_iters : int
        For 'iterative', max passes.
    tol : float
        For 'iterative', stop when max |Δ| on formerly-NaN entries < tol.
    kernel_size : int
        For 'iterative', size of the averaging window along each axis (odd).

    Returns
    -------
    filled : np.ndarray
        Copy of arr with NaNs filled.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be a NumPy ndarray")

    if arr.ndim < 1:
        raise ValueError("arr must be at least 1-D")

    filled = arr.astype(np.float64, copy=True)  # work in float for NaNs
    nan_mask = np.isnan(filled)

    if not np.any(nan_mask):
        return arr.copy()

    if np.all(nan_mask):
        raise ValueError("All values are NaN; nothing to interpolate from.")

    if method == "nearest":
        known = ~nan_mask
        # distance_transform_edt measures distance to nearest zero; pass (~known)
        _, ind = distance_transform_edt(~known, return_indices=True)
        filled[nan_mask] = filled[tuple(ind[:, nan_mask])]
        return filled.astype(arr.dtype, copy=False)

    if method == "linear":
        from scipy.interpolate import griddata

        # Get coordinates of known and missing points in N-D
        coords_known = np.array(np.nonzero(~nan_mask)).T  # (M, N)
        vals_known = filled[~nan_mask]
        coords_miss  = np.array(np.nonzero(nan_mask)).T   # (K, N)

        lin_vals = griddata(coords_known, vals_known, coords_miss, method="linear")

        # Fallback for points outside convex hull
        need_nn = np.isnan(lin_vals)
        if np.any(need_nn):
            nn_vals = griddata(coords_known, vals_known, coords_miss[need_nn], method="nearest")
            lin_vals[need_nn] = nn_vals

        filled[nan_mask] = lin_vals
        return filled.astype(arr.dtype, copy=False)

    if method == "iterative":
        from scipy.ndimage import convolve

        if kernel_size % 2 != 1 or kernel_size < 3:
            raise ValueError("kernel_size should be an odd integer >= 3")

        data = filled.copy()
        data[nan_mask] = 0.0
        weights = (~nan_mask).astype(np.float64)

        # N-D uniform kernel (e.g., 3x3x...x3)
        kernel = np.ones((kernel_size,) * arr.ndim, dtype=np.float64)

        prev_vals = data[nan_mask]
        for _ in range(max_iters):
            num = convolve(data, kernel, mode="nearest")
            den = convolve(weights, kernel, mode="nearest")

            upd_mask = nan_mask & (den > 0)
            new_vals = num[upd_mask] / den[upd_mask]

            data[upd_mask] = new_vals
            weights[upd_mask] = 1.0  # become "known" for subsequent passes

            cur_vals = data[nan_mask]
            max_change = np.nanmax(np.abs(cur_vals - prev_vals))
            if max_change < tol:
                break
            prev_vals = cur_vals

        filled[nan_mask] = data[nan_mask]
        return filled.astype(arr.dtype, copy=False)

    raise ValueError("method must be 'linear', 'iterative', or 'nearest'")
from skimage.transform import resize

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
    """
    calculated_array = np.asarray(calculated_array, dtype=np.float64)
    akro_array = np.asarray(akro_array, dtype=np.float64)

    # gmin = min(np.nanmin(a), np.nanmin(b))
    # gmax = min(np.nanmax(a), np.nanmax(b))
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

def border_mask_from_rect(shape, min_row, max_row, min_col, max_col):
    """
    Create a boolean mask for the border in an array of shape (X, Y, Z),
    where the interior rectangle is [min_row:max_row] x [min_col:max_col] in XY.
    Everything outside this rectangle (for all Z) is marked as border (True).
    
    Parameters
    ----------
    shape : tuple
        (X, Y, Z)
    min_row, max_row : int
        Row bounds (inclusive/exclusive style explained below).
    min_col, max_col : int
        Column bounds (inclusive/exclusive style explained below).

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (X, Y, Z), True for border positions, False for interior.
    """
    X, Y= shape
    mask = np.ones(shape, dtype=bool)

    # Sanity clamp (optional)
    min_row = max(0, min_row+1)
    min_col = max(0, min_col+1)
    max_row = min(X, max_row)
    max_col = min(Y, max_col)

    # Mark interior rectangle as False (not border)
    # Convention: interior includes rows [min_row, max_row) and cols [min_col, max_col)
    mask[min_row:max_row, min_col:max_col] = False
    return mask


def _first_true_index(mask: np.ndarray, axis: int) -> np.ndarray:
    """
    Return the index of the first True along `axis` for each slice,
    or np.nan if a slice has no True.
    """
    # Argmax gives the first occurrence *if* there is at least one True.
    idx = mask.argmax(axis=axis)
    has_true = mask.any(axis=axis)
    # Where there is no True, set to np.nan
    out = idx.astype(float)
    out[~has_true] = np.nan
    return out

def _last_true_index(mask: np.ndarray, axis: int) -> np.ndarray:
    """
    Return the index of the last True along `axis` for each slice,
    or np.nan if a slice has no True.
    Implemented by reversing along `axis` and using first_true_index.
    """
    if axis == 0:
        rev = mask[::-1, ...]
        first_rev = _first_true_index(rev, axis=0)  # indices relative to reversed
        # Convert back: last_index = (size - 1) - first_rev
        n = mask.shape[0]
        out = (n - 1) - first_rev
    elif axis == 1:
        rev = mask[:, ::-1, ...]
        first_rev = _first_true_index(rev, axis=1)
        n = mask.shape[1]
        out = (n - 1) - first_rev
    else:
        raise ValueError("axis must be 0 (rows) or 1 (cols) for a 2D array")
    return out

def _mode_int_ignore_nan(vals: np.ndarray):
    """
    Return the integer mode of vals after dropping NaNs.
    If all are NaN or empty, return None.
    """
    clean = vals[~np.isnan(vals)]
    if clean.size == 0:
        return None
    # Convert to int safely
    as_int = clean.astype(int)
    # bincount on non-negative ints
    # shift if negatives can exist (not expected here)
    minv = as_int.min()
    if minv < 0:
        as_int = as_int - minv
    counts = np.bincount(as_int)
    mode_shifted = counts.argmax()
    mode_val = mode_shifted + (minv if minv < 0 else 0)
    return int(mode_val)

def find_border_idx(array: np.ndarray, threshold: float = 0.1):
    """
    Compute border indices (min_row, min_col, max_row, max_col) for a 2D array,
    using the same logic as the original implementation, but vectorized.

    Logic:
    - min_col: for each row in the top half, find first col with value > threshold, then subtract 1; take mode
    - max_col: for each row (all rows), find last col with value > threshold, then add 1; take mode
    - min_row: for each col in the left half, find first row with value > threshold, then subtract 1; take mode
    - max_row: for each col in the left half, find last row with value > threshold, then add 1; take mode

    Returns:
        (min_row_mode, min_col_mode, max_row_mode, max_col_mode)
        Each is an int or None if no valid indices were found.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array; got ndim={array.ndim}")
    n_rows, n_cols = array.shape

    # Threshold mask
    mask = array > threshold

    # Top half rows
    top_half = mask[: n_rows // 2, :]
    # min_col per top-half row → first True along axis=1, then minus 1
    first_nonzero_col_top = _first_true_index(top_half, axis=1) - 1
    min_col_mode = _mode_int_ignore_nan(first_nonzero_col_top)

    # All rows → last nonzero col per row → last True along axis=1, then +1
    last_nonzero_col_all = _last_true_index(mask, axis=1) + 1
    max_col_mode = _mode_int_ignore_nan(last_nonzero_col_all)

    # Left half columns
    left_half = mask[:, : n_cols // 2]
    # min_row per left-half col → first True along axis=0, then minus 1
    first_nonzero_row_left = _first_true_index(left_half, axis=0) - 1
    min_row_mode = _mode_int_ignore_nan(first_nonzero_row_left)

    # max_row per left-half col → last True along axis=0, then +1
    last_nonzero_row_left = _last_true_index(left_half, axis=0) + 1
    max_row_mode = _mode_int_ignore_nan(last_nonzero_row_left)

    return (min_row_mode, min_col_mode, max_row_mode, max_col_mode)

def get_border_mask(array):
    min_row_idx, min_col_idx, max_row_idx, max_col_idx = find_border_idx(array)
    return border_mask_from_rect(array.shape, min_row_idx, max_row_idx, min_col_idx, max_col_idx)

def idw_fill_2d(arr: np.ndarray, mask: np.ndarray, *, threshold: float = 0.1, power: float = 2.0, k: int = 8, max_radius: float | None = None,
    treat_nan_as_invalid: bool = True, fallback: float | None = None) -> np.ndarray:
    """
    Fill masked positions in a 2D array using Inverse Distance Weighting (IDW).
    Requires SciPy (scipy.spatial.cKDTree). If SciPy is unavailable, use
    `idw_fill_2d_numpy` (provided below).

    Parameters
    ----------
    arr : np.ndarray
        2D numeric array (rows x cols).
    mask : np.ndarray
        Boolean mask, same shape as `arr`. True where you want to fill.
    threshold : float, default=0.1
        Validity threshold. A cell is valid if abs(value) > threshold.
    power : float, default=2.0
        IDW power parameter (p). Larger p down-weights distant neighbors more. (Closer neighbors dominant more)
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
            "SciPy not available. Use `idw_fill_2d_numpy` (provided below) or install scipy."
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

def fill_border_with_percent_max(arr: np.ndarray, mask: np.ndarray, percent: int) -> np.ndarray:
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

if __name__ == "__main__":
    # dummy example arrays
    g = np.random.rand(400, 600)
    d = np.random.rand(200, 300)

    a = align_dat_to_gerber(g, d)
    # print("Transform params:", info)
    # print("Output shape:", aligned.shape)