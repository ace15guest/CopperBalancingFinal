"""
Configuration management for the Copper Balancing Image Processing application.

This module contains all configuration settings and constants used throughout the project.
"""

import re
from pathlib import Path


class Config:
    """Configuration class for project settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ASSETS_DIR = PROJECT_ROOT / "Assets"
    OUTPUT_DIR = PROJECT_ROOT / "Assets" / "DataOutput"
    
    # External tools
    GERBV_PATH = ASSETS_DIR / "gerbv" / "gerbv.exe"
    
    # Processing parameters
    DEFAULT_DPI = 300
    DEFAULT_BLUR_KERNEL = 5
    DEFAULT_GAUSSIAN_SIGMA = 1.0
    
    # Layer naming patterns (regex)
    LAYER_PATTERNS = {
        'signal': r'l\d+_signal',
        'plane': r'l\d+_plane',
    }
    
    # Match: 1oz, 0.5oz, 0_5oz, 1.0oz, optionally with spaces, and require '_' or end after 'oz'
    _OZ_RE = re.compile(r'(\d+(?:[._]\d+)?)\s*oz(?=$|_)', re.IGNORECASE)
    # Quadrant identifiers
    QUADRANTS = ['Q1', 'Q2', 'Q3', 'Q4', 'Global']
    # File extensions
    GERBER_EXTENSIONS = ['.gbr']
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']

    # Akrometrix Data Folders
    TOP_DAT_FOLDER =  ASSETS_DIR / 'AkroFiles' / 'TopDatFiles'
    TOP_DAT_FILES = [p for p in TOP_DAT_FOLDER.iterdir() if p.is_file()]
    BOT_AKFRO_FOLDER = ASSETS_DIR / 'AkroFiles' / 'BottomDatFiles'
    BOT_AKRO_FILES = [p for p in BOT_AKFRO_FOLDER.iterdir() if p.is_file()]
    ALL_AKRO_FILES = TOP_DAT_FILES + BOT_AKRO_FILES
    # Processing options
    USE_COMPRESSION = True
    PARALLEL_WORKERS = 4
    
    # Output file names
    DATA_OUTPUT_FILE = OUTPUT_DIR / 'data_out.csv'
    


    # Parameter Sweeping
    # Parameter sweep configurations
    PARAMETER_SWEEPS = {
    'window_sizes': [1, 3, 5, 7, 9],
    'edge_fills': ['idw', 'nearest', 'percent_max'],  # pixels
    'dx_choices': [1.0, 2, 5, 10.0],
    'dpis': [50, 100, 200, 300, 500, 700],
    'percent_max_fills': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'blur_kernels': [3, 5, 7, 9],
    'gradient_methods': ['finite', 'plane'],
    'gaussian_sigmas': [0.5, 1.0, 1.5, 2.0],
    'blur_types': ['box', 'gaussian'],
    'radii': [25, 50, 100, 250, 500],
    'sigmas': [0.5, 1.0, 1.1, 1.5],
    'percent_area_from_centers': [.25, .4, .6]    }

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "pngs").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "arrays").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "processed").mkdir(exist_ok=True)

    @classmethod
    def get_gerber_dirs(cls):
        cls.GERBER_FOLDERS = [f"{str(cls.ASSETS_DIR)}/gerbers/CU_Balancing_Gerber/{quad}" for quad in QUADRANTS]

    
    @classmethod
    def validate_gerbv(cls) -> bool:
        """Check if gerbv.exe exists."""
        return cls.GERBV_PATH.exists()

# Made with Bob
