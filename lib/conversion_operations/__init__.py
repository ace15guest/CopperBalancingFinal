"""
Conversion Operations Library

This module handles all file format conversions and I/O operations:
- DAT file to array conversion
- Gerber to PNG conversion using gerbv
- PNG to numpy array loading
- NPZ file save/load operations
"""

from .dat_converter import load_dat_file, dat_to_array, validate_dat_format
from .gerber_converter import gerber_to_png_gerbv, batch_gerber_to_png, get_gerbv_path
from .png_loader import load_png, validate_png, bitmap_to_array
from .npz_handler import save_npz, load_npz, save_array_stack, load_array_stack

__all__ = [
    # DAT converter
    'load_dat_file',
    'dat_to_array',
    'validate_dat_format',
    # Gerber converter
    'gerber_to_png_gerbv',
    'batch_gerber_to_png',
    'get_gerbv_path',
    # PNG loader
    'load_png',
    'bitmap_to_array',

    'validate_png',
    # NPZ handler
    'save_npz',
    'load_npz',
    'save_array_stack',
    'load_array_stack',
]

# Made with Bob
