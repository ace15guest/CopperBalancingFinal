"""
PNG Loader

This module handles loading PNG files into numpy arrays.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def load_png(filepath: str, grayscale: bool = True) -> np.ndarray:
    """
    Load a PNG file into a numpy array.
    
    Parameters:
        filepath (str): Path to PNG file
        grayscale (bool, optional): Convert to grayscale. Default: True
        
    Returns:
        np.ndarray: Image array (H, W) if grayscale, (H, W, C) if color
    """
    # TODO: Implement PNG loading
    raise NotImplementedError("PNG loading not yet implemented")


def png_to_array(filepath: str, normalize: bool = False) -> np.ndarray:
    """
    Load PNG and optionally normalize to [0, 1] range.
    
    Parameters:
        filepath (str): Path to PNG file
        normalize (bool, optional): Normalize to [0, 1]. Default: False
        
    Returns:
        np.ndarray: Image array, normalized if requested
    """
    # TODO: Implement PNG to array with normalization
    raise NotImplementedError("PNG to array conversion not yet implemented")


def validate_png(filepath: str) -> bool:
    """
    Check if file is a valid PNG.
    
    Parameters:
        filepath (str): Path to file
        
    Returns:
        bool: True if valid PNG
    """
    # TODO: Implement PNG validation
    raise NotImplementedError("PNG validation not yet implemented")

# Made with Bob
