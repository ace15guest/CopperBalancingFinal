"""
Blur Filters

This module implements box blur and Gaussian blur algorithms using separable convolution.
"""

import numpy as np
from typing import Optional
from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter


def blur_call(array: np.ndarray, blur_type: str, radius: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply blur filter to array based on specified type.
    
    Parameters:
        array (np.ndarray): Input 2D array to blur
        blur_type (str): Type of blur - 'box_blur' or 'gauss'
        radius (int): Window size for box blur (default: 5)
        sigma (float): Standard deviation for Gaussian blur (default: 1.0)
        
    Returns:
        np.ndarray: Blurred array
        
    Raises:
        ValueError: If blur_type is not recognized or array is not 2D
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array")
    
    if blur_type == "box_blur":
        return box_blur(array, window_size=radius)
    elif blur_type == "gauss":
        return blur_2d_gaussian(array, sigma=sigma)
    else:
        raise ValueError(f"Unknown blur_type: '{blur_type}'. Use 'box_blur' or 'gauss'")


def box_blur(array: np.ndarray, window_size: int = 5, mode: str = 'nearest') -> np.ndarray:
    """
    Apply box blur (uniform averaging) to array using separable convolution.
    
    Parameters:
        array (np.ndarray): Input array (2D)
        window_size (int): Size of blur kernel (default: 5)
        mode (str): Boundary handling mode - 'nearest', 'reflect', 'mirror', 'wrap', 'constant'
        
    Returns:
        np.ndarray: Blurred array (same shape as input)
        
    Raises:
        ValueError: If window_size is < 1 or array is not 2D
    """
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array")
    
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    
    # Apply uniform filter (box blur) using scipy's optimized implementation
    # This uses separable convolution internally for efficiency
    blurred = uniform_filter(array.astype(float), size=window_size, mode=mode)
    
    # Preserve original dtype
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        blurred = np.clip(blurred, info.min, info.max)
        return blurred.astype(array.dtype)
    
    return blurred


def blur_2d_gaussian(volume_2d: np.ndarray, sigma: float = 1.0, mode: str = 'nearest', truncate: float = 3.0, preserve_dtype: bool = True) -> np.ndarray:
    """
    Apply a 2D Gaussian blur using separable convolution.

    Parameters:
        volume_2d (np.ndarray): Input 2D array of shape (H, W)
        sigma (float): Standard deviation of the Gaussian kernel (default: 1.0)
            Can also be tuple (sy, sx) for different sigmas per axis
        mode (str): Boundary handling mode (default: 'nearest')
            Options: 'reflect', 'nearest', 'mirror', 'wrap', 'constant'
        truncate (float): Truncate filter at this many standard deviations (default: 3.0)
            Kernel radius is int(truncate * sigma) for each axis
        preserve_dtype (bool): If True, cast back to input dtype after filtering (default: True)

    Returns:
        np.ndarray: Blurred array of same shape as input
        
    Raises:
        ValueError: If volume_2d is not 2D or sigma is invalid
    """
    if volume_2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got {volume_2d.ndim}D array")
    
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
    elif isinstance(sigma, (tuple, list)):
        if len(sigma) != 2 or any(s <= 0 for s in sigma):
            raise ValueError(f"sigma tuple must have 2 positive values, got {sigma}")
    
    # Compute in float for accuracy
    arr = volume_2d.astype(float)
    blurred = gaussian_filter(arr, sigma=sigma, mode=mode, truncate=truncate)
    
    if preserve_dtype:
        # Clip if integer type to avoid wrap-around on casts
        if np.issubdtype(volume_2d.dtype, np.integer):
            info = np.iinfo(volume_2d.dtype)
            blurred = np.clip(blurred, info.min, info.max)
        return blurred.astype(volume_2d.dtype)
    
    return blurred