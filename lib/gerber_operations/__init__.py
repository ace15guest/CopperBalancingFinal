"""
Gerber Operations Library

This module handles Gerber-specific operations:
- Gerber file parsing and metadata extraction
- Layer management and organization
- Batch processing of Gerber files
"""

from .gerber_parser import (
    parse_gerber_header,
    extract_layer_info,
    get_layer_type,
    parse_layer_name
)
from .layer_manager import (
    organize_layers,
    get_layer_stack,
    validate_layer_stack,
    get_quadrant_layers
)
# from .batch_processor import (
#     process_directory,
#     process_layer_stack,
#     parallel_process
# )

__all__ = [
    # Gerber parser
    'parse_gerber_header',
    'extract_layer_info',
    'get_layer_type',
    'parse_layer_name',
    # Layer manager
    'organize_layers',
    'get_layer_stack',
    'validate_layer_stack',
    'get_quadrant_layers',
    # Batch processor
    # 'process_directory',
    # 'process_layer_stack',
    # 'parallel_process',
]

# Made with Bob
