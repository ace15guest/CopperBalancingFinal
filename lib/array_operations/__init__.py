"""
Array Operations Library

This module handles image processing and array manipulation operations:
- Blur filters (box and Gaussian)
- Array utilities (alignment, normalization, interpolation, scaling)
- Image processing operations (thresholding, masking, statistics)
"""

from .blur_filters import (
    box_blur,
    blur_2d_gaussian,
    blur_call
)

from .array_utils import (
    # Alignment functions
    align_dat_to_gerber,
    apply_alignment,
    
    # Helper functions
    sanitize,
    binarize_robust,
    dominant_angle_safe,
    auto_scale,
    phase_translate,
    
    # NaN handling and interpolation
    fill_nans_nd,
    
    # Scaling and resizing
    shrink_array,
    rescale_to_shared_minmax,
    
    # Multi-layer operations
    multiple_layers_weighted
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
    'blur_call',
    
    # Alignment functions
    'align_dat_to_gerber',
    'apply_alignment',
    
    # Helper functions
    'sanitize',
    'binarize_robust',
    'dominant_angle_safe',
    'auto_scale',
    'phase_translate',
    
    # NaN handling and interpolation
    'fill_nans_nd',
    
    # Scaling and resizing
    'shrink_array',
    'rescale_to_shared_minmax',
    
    # Multi-layer operations
    'multiple_layers_weighted',
    
    # Image processing
    'threshold',
    'apply_mask',
    'calculate_statistics',
]

__version__ = '1.0.0'
__author__ = 'Bob'

# Made with Bob
