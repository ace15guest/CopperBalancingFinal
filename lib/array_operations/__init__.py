"""
Array Operations Library

This module handles image processing and array manipulation operations:
- Blur filters (box and Gaussian)
- Array utilities (normalization, padding, cropping)
- Image processing operations (thresholding, masking, statistics)
"""

from .blur_filters import (
    box_blur,
    blur_2d_gaussian
)
from .array_utils import (
    normalize_array,
    pad_array,
    center_crop_by_area,
    validate_array,
    convert_dtype
)
from .image_processing import (
    threshold,
    apply_mask,
    calculate_statistics
)

__all__ = [
    # Blur filters
    'box_blur',
    'blur_2d_gaussian',
    # Array utilities
    'normalize_array',
    'pad_array',
    'center_crop_by_area',
    'validate_array',
    'convert_dtype',
    # Image processing
    'threshold',
    'apply_mask',
    'calculate_statistics',
]

# Made with Bob
