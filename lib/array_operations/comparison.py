"""
Array comparison and 3D alignment operations.

This module provides functions for aligning and comparing 3D point clouds
or surface data, including rigid transformations and statistical metrics.
"""

import numpy as np
from typing import Dict


def _as_points(arr, ignore_zeros=True):
    """
    Convert a 2D array to Nx3 points (x, y, z).
    
    Parameters
    ----------
    arr : 2D ndarray
        Height map where arr[y, x] = z.
    ignore_zeros : bool
        If True, exclude points where z == 0.
    
    Returns
    -------
    points : Nx3 ndarray
        Array of (x, y, z) coordinates.
    """
    ny, nx = arr.shape
    y, x = np.mgrid[0:ny, 0:nx]
    z = arr.astype(float)
    
    if ignore_zeros:
        mask = (z != 0) & np.isfinite(z)
    else:
        mask = np.isfinite(z)
    
    points = np.column_stack([x[mask], y[mask], z[mask]])
    return points


def _kabsch_umeyama(P, Q, with_scaling=False):
    """
    Compute optimal rigid transformation (rotation, translation, optional scale)
    from point set P to Q using Kabsch/Umeyama algorithm.
    
    Parameters
    ----------
    P, Q : Nx3 ndarray
        Source and target point sets.
    with_scaling : bool
        If True, allow uniform scaling.
    
    Returns
    -------
    R : 3x3 ndarray
        Rotation matrix.
    t : 3-element ndarray
        Translation vector.
    s : float
        Scale factor (1.0 if with_scaling=False).
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    
    # Center the point sets
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute cross-covariance matrix
    H = P_centered.T @ Q_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    if with_scaling:
        var_P = np.var(P_centered, axis=0).sum()
        s = S.sum() / var_P if var_P > 0 else 1.0
    else:
        s = 1.0
    
    # Translation
    t = centroid_Q - s * (R @ centroid_P)
    
    return R, t, s


def _fit_plane(points):
    """
    Fit a plane z = ax + by + c to 3D points.
    
    Parameters
    ----------
    points : Nx3 ndarray
        Points (x, y, z).
    
    Returns
    -------
    coeffs : 3-element ndarray
        Plane coefficients [a, b, c].
    """
    X = points[:, :2]  # x, y
    z = points[:, 2]
    X_aug = np.column_stack([X, np.ones(len(X))])
    coeffs, *_ = np.linalg.lstsq(X_aug, z, rcond=None)
    return coeffs


def _detrend_plane(points):
    """
    Remove best-fit plane from 3D points.
    
    Parameters
    ----------
    points : Nx3 ndarray
        Points (x, y, z).
    
    Returns
    -------
    detrended : Nx3 ndarray
        Points with plane removed from z coordinates.
    """
    coeffs = _fit_plane(points)
    a, b, c = coeffs
    z_plane = a * points[:, 0] + b * points[:, 1] + c
    detrended = points.copy()
    detrended[:, 2] -= z_plane
    return detrended


def _downsample_pairs(P, Q, maxN, seed=None):
    """
    Randomly downsample paired point sets to maxN points.
    
    Parameters
    ----------
    P, Q : Nx3 ndarray
        Paired point sets.
    maxN : int
        Maximum number of points to keep.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    P_sub, Q_sub : Mx3 ndarray
        Downsampled point sets (M <= maxN).
    """
    N = P.shape[0]
    if N <= maxN:
        return P, Q
    
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, size=maxN, replace=False)
    return P[idx], Q[idx]


def align_and_compare(
    A: np.ndarray,
    B: np.ndarray,
    *,
    ignore_zeros: bool = True,
    detrend: bool = True,
    with_scaling: bool = False,
    maxN_align: int = 20000
) -> Dict[str, float]:
    """
    Align A->B with rigid SVD (optionally scaling), then compute stats.
    Assumes A and B represent the same grid (1-1 correspondence by index).
    
    Parameters
    ----------
    A, B : 2D ndarray
        Height maps to compare. Must have same shape.
    ignore_zeros : bool
        If True, exclude zero values from analysis.
    detrend : bool
        If True, remove best-fit plane from each surface before computing stats.
    with_scaling : bool
        If True, allow uniform scaling in alignment.
    maxN_align : int
        Maximum number of points to use for alignment (downsampled if needed).
    
    Returns
    -------
    stats : dict
        Dictionary containing:
        - Transform parameters: scale, R (rotation matrix), t (translation)
        - Distance metrics: rmse_3d, mae_3d, p95_3d, max_3d
        - Vertical metrics: rmse_z, mae_z, p95_z, max_z
        - Height relationship: pearson_r, slope, intercept, r2
        - Metadata: n (number of points), detrended, with_scaling
        - text: One-line summary string
    """
    # 1) Build point sets
    P = _as_points(A, ignore_zeros=ignore_zeros)  # Nx3 (x,y,z_A)
    Q = _as_points(B, ignore_zeros=ignore_zeros)  # Nx3 (x,y,z_B)

    # 2) Use common indices only (in case masks differed)
    # Map (x,y) -> z and intersect keys
    kP = {(int(px), int(py)) for px, py in P[:, :2]}
    kQ = {(int(qx), int(qy)) for qx, qy in Q[:, :2]}
    common = kP & kQ
    if not common:
        raise ValueError("No overlapping (x,y) samples between A and B after masking.")
    
    # Build aligned arrays in the same (x,y) order
    def _points_from_keys(arr, keys):
        z = arr.astype(float)
        pts = []
        for (x, y) in keys:
            pts.append((x, y, z[y, x]))
        return np.array(pts, float)

    keys_sorted = sorted(common)  # deterministic order
    Pfull = _points_from_keys(A, keys_sorted)
    Qfull = _points_from_keys(B, keys_sorted)

    # 3) (Optional) downsample for alignment only
    Pfit, Qfit = _downsample_pairs(Pfull, Qfull, maxN=maxN_align, seed=123)

    # 4) Kabsch / Umeyama (rigid, optional scale)
    R, t, s = _kabsch_umeyama(Pfit, Qfit, with_scaling=with_scaling)

    # Apply transform to ALL correspondences
    Paligned = s * (Pfull @ R.T) + t  # Nx3

    # 5) (Optional) detrend each surface before stats
    if detrend:
        Paligned = _detrend_plane(Paligned)
        Qd        = _detrend_plane(Qfull)
    else:
        Qd = Qfull

    # 6) Metrics
    diff3d = Paligned - Qd
    dists = np.linalg.norm(diff3d, axis=1)
    dz = Paligned[:, 2] - Qd[:, 2]

    def _pct(a, p): return float(np.percentile(a, p))

    rmse_3d = float(np.sqrt(np.mean(dists**2)))
    mae_3d  = float(np.mean(np.abs(dists)))
    p95_3d  = _pct(dists, 95)
    max_3d  = float(np.max(dists))

    rmse_z = float(np.sqrt(np.mean(dz**2)))
    mae_z  = float(np.mean(np.abs(dz)))
    p95_z  = _pct(np.abs(dz), 95)
    max_z  = float(np.max(np.abs(dz)))

    # Correlation + linear fit of heights
    zA = Paligned[:, 2]
    zB = Qd[:, 2]
    # Pearson r
    r = float(np.corrcoef(zA, zB)[0, 1]) if zA.size > 1 else np.nan
    # Linear regression zB ~ a*zA + b
    Areg = np.column_stack([zA, np.ones_like(zA)])
    a, b = np.linalg.lstsq(Areg, zB, rcond=None)[0]
    # R^2
    yhat = a * zA + b
    ss_res = float(np.sum((zB - yhat)**2))
    ss_tot = float(np.sum((zB - np.mean(zB))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    stats = {
        # transform
        "scale": s,
        "R00": float(R[0,0]), "R01": float(R[0,1]), "R02": float(R[0,2]),
        "R10": float(R[1,0]), "R11": float(R[1,1]), "R12": float(R[1,2]),
        "R20": float(R[2,0]), "R21": float(R[2,1]), "R22": float(R[2,2]),
        "t_x": float(t[0]), "t_y": float(t[1]), "t_z": float(t[2]),
        # distances
        "rmse_3d": rmse_3d, "mae_3d": mae_3d, "p95_3d": p95_3d, "max_3d": max_3d,
        # vertical-only
        "rmse_z": rmse_z, "mae_z": mae_z, "p95_z": p95_z, "max_z": max_z,
        # height relationship
        "pearson_r": r, "slope": float(a), "intercept": float(b), "r2": r2,
        # counts
        "n": int(Paligned.shape[0]),
        "detrended": bool(detrend),
        "with_scaling": bool(with_scaling),
    }

    # One-line summary you can drop into your Plotly annotation:
    stats_text = (
        f"n={stats['n']} | rmse₃ᴅ={stats['rmse_3d']:.4g} | p95₃ᴅ={stats['p95_3d']:.4g} | "
        f"rmse_z={stats['rmse_z']:.4g} | p95_z={stats['p95_z']:.4g} | "
        f"r={stats['pearson_r']:.3f} | zB≈{stats['slope']:.3f}·zA+{stats['intercept']:.3f} | "
        f"R²={stats['r2']:.3f} | scale={stats['scale']:.4f}"
        + (" | detrended" if detrend else "")
    )

    stats["text"] = stats_text
    return stats

# Made with Bob
