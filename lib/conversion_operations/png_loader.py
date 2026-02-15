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


def bitmap_to_array(bitmap_path, inverted=False):
    """Convert bitmap image to normalized numpy array.
    
    Args:
        bitmap_path: Path to the bitmap file
        inverted: If True, invert the pixel values
        
    Returns:
        numpy array or None if conversion fails
    """
    try:
        with Image.open(bitmap_path) as img:
            gray = img.convert('L')  # Convert to grayscale
            array = np.array(gray)
            
            # Normalize array, handling edge case of all-zero images
            max_val = np.max(array)
            if max_val > 0:
                array = array * 255.0 / max_val
            
            if inverted:
                array = 255 - array
            
            return array
    except Exception as e:
        print(f"Error converting bitmap at {bitmap_path}: {e}")
        return None


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
