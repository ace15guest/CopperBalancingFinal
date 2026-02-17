"""
Array Operations Library

This module handles image processing and array manipulation operations:
- Blur filters (box and Gaussian)
- Alignment operations (DAT to Gerber registration)
- Border and mask operations
- Interpolation and fill operations
- Gradient analysis
- Array comparison and 3D alignment
- Basic array utilities (scaling, resizing, multi-layer operations)
- Image processing operations (thresholding, masking, statistics)
"""

from .blur_filters import (
    box_blur,
    blur_2d_gaussian,
    blur_call
)

from .alignment import (
    # Main alignment functions
    align_dat_to_gerber,
    apply_alignment,
    
    # Helper functions
    sanitize,
    binarize_robust,
    dominant_angle_safe,
    auto_scale,
    phase_translate,
)

from .border_operations import (
    find_border_idx,
    border_mask_from_rect,
    get_border_mask,
    center_crop_by_area,
)

from .interpolation import (
    fill_nans_nd,
    idw_fill_2d,
    nearest_border_fill_true_2d,
    fill_border_with_percent_max,
)

from .gradient_analysis import (
    analyze_gradients,
    compute_gradients,
    compare_gradient_fields,
)

from .comparison import (
    align_and_compare,
)

from .array_utils import (
    shrink_array,
    rescale_to_shared_minmax,
    multiple_layers_weighted,
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
    'sanitize',
    'binarize_robust',
    'dominant_angle_safe',
    'auto_scale',
    'phase_translate',
    
    # Border operations
    'find_border_idx',
    'border_mask_from_rect',
    'get_border_mask',
    'center_crop_by_area',
    
    # Interpolation
    'fill_nans_nd',
    'idw_fill_2d',
    'nearest_border_fill_true_2d',
    'fill_border_with_percent_max',
    
    # Gradient analysis
    'analyze_gradients',
    'compute_gradients',
    'compare_gradient_fields',
    
    # Comparison
    'align_and_compare',
    
    # Array utilities
    'shrink_array',
    'rescale_to_shared_minmax',
    'multiple_layers_weighted',
    
    # Image processing
    'threshold',
    'apply_mask',
    'calculate_statistics',
]

__version__ = '1.0.0'
__author__ = 'Bob'

# Made with Bob
