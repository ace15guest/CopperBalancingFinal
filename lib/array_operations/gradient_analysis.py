"""
Gradient computation and analysis for 2D surfaces.

This module provides functions for computing gradients using different methods
and comparing gradient fields between surfaces.
"""

import numpy as np


def analyze_gradients(
        Z_model,
        Z_ref,
        dx=1.0,
        dy=1.0,
        method="finite",
        window_size=3,
        make_plots=True,
):
    """
    High-level wrapper:
      - compute gradients using selected method
      - compare fields
      - return metrics + maps
    """
    # Gradients
    gx_m, gy_m = compute_gradients(Z_model, dx, dy, method, window_size)
    gx_r, gy_r = compute_gradients(Z_ref, dx, dy, method, window_size)

    # Compare
    metrics, angle_diff, mag_ratio = compare_gradient_fields(
        gx_m, gy_m, gx_r, gy_r
    )

    # Print summary
    # print("=== Gradient Comparison Metrics ===")
    # for k, v in metrics.items():
    #     print(f"{k:18s}: {v:.4f}")

    # Plot
    # if make_plots:
    #     plot_gradient_results(Z_model, Z_ref, angle_diff)

    return metrics, angle_diff, mag_ratio


def compute_gradients(
        Z,
        dx=1.0,
        dy=1.0,
        method="finite",
        window_size=3,
):
    """
    Compute gradients dz/dx and dz/dy for a 2D surface Z using either:
      - finite differences (np.gradient), or
      - local plane fitting over an N x N neighborhood.

    Parameters
    ----------
    Z : 2D ndarray
        Surface height data.
    dx, dy : float
        Spacing in x (columns) and y (rows).
    method : {"finite", "plane"}
        "finite" -> 2-neighbor finite difference
        "plane"  -> least-squares plane fit in an N x N window
    window_size : int
        Odd integer (3, 5, 7...). Size of plane-fit neighborhood.

    Returns
    -------
    gx, gy : 2D ndarray
        Gradient components dz/dx and dz/dy
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")

    if method == "finite":
        dZ_dy, dZ_dx = np.gradient(Z, dy, dx)
        return dZ_dx, dZ_dy

    elif method == "plane":
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")

        ny, nx = Z.shape
        half = window_size // 2
        gx = np.zeros_like(Z)
        gy = np.zeros_like(Z)

        # Precompute coordinates for the N x N window (centered at 0,0)
        wy = np.arange(-half, half + 1) * dy
        wx = np.arange(-half, half + 1) * dx
        WY, WX = np.meshgrid(wy, wx, indexing="ij")

        # Flatten for least squares
        X_flat = WX.ravel()
        Y_flat = WY.ravel()
        A_base = np.column_stack([X_flat, Y_flat, np.ones_like(X_flat)])

        for i in range(ny):
            for j in range(nx):
                r0 = max(0, i - half)
                r1 = min(ny, i + half + 1)
                c0 = max(0, j - half)
                c1 = min(nx, j + half + 1)

                patch = Z[r0:r1, c0:c1]
                pr, pc = patch.shape

                # Take matching rows from A_base
                A = A_base.reshape(window_size, window_size, 3)[:pr, :pc].reshape(-1, 3)
                z_vec = patch.ravel()

                coeffs, *_ = np.linalg.lstsq(A, z_vec, rcond=None)
                gx[i, j] = coeffs[0]
                gy[i, j] = coeffs[1]

        return gx, gy

    else:
        raise ValueError("method must be 'finite' or 'plane'")


def compare_gradient_fields(gx_m, gy_m, gx_r, gy_r, eps=1e-12):
    """
    Compare model vs reference gradient fields.

    Parameters
    ----------
    gx_m, gy_m : 2D ndarray
        Model gradient components (dz/dx, dz/dy).
    gx_r, gy_r : 2D ndarray
        Reference gradient components (dz/dx, dz/dy).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    metrics : dict
        Dictionary of summary statistics:
        - angle_mean_deg, angle_median_deg, angle_p95_deg
        - mag_ratio_mean, mag_ratio_median, mag_ratio_p05, mag_ratio_p95
    angle_diff : 2D ndarray
        Angle difference map (degrees) between model and reference gradients.
    mag_ratio : 2D ndarray
        Magnitude ratio map: |∇Z_ref| / |∇Z_model|
    """
    g_m = np.stack([gx_m, gy_m], axis=-1)
    g_r = np.stack([gx_r, gy_r], axis=-1)

    mag_m = np.linalg.norm(g_m, axis=-1)
    mag_r = np.linalg.norm(g_r, axis=-1)

    dot = np.sum(g_m * g_r, axis=-1)
    denom = mag_m * mag_r + eps
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    angle_diff = np.degrees(np.arccos(cos_theta))

    mag_ratio = mag_r / (mag_m + eps)

    # Summary metrics
    angle_flat = angle_diff.ravel()
    ratio_flat = mag_ratio.ravel()

    metrics = {
        "angle_mean_deg": float(np.mean(angle_flat)),
        "angle_median_deg": float(np.median(angle_flat)),
        "angle_p95_deg": float(np.percentile(angle_flat, 95)),

        "mag_ratio_mean": float(np.mean(ratio_flat)),
        "mag_ratio_median": float(np.median(ratio_flat)),
        "mag_ratio_p05": float(np.percentile(ratio_flat, 5)),
        "mag_ratio_p95": float(np.percentile(ratio_flat, 95)),
    }

    return metrics, angle_diff, mag_ratio

# Made with Bob
