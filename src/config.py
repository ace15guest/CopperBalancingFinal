"""
Configuration management for the Copper Balancing Image Processing application.

This module contains all configuration settings and constants used throughout the project.
"""

import os
from pathlib import Path


class Config:
    """Configuration class for project settings."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ASSETS_DIR = PROJECT_ROOT / "Assets"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    
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
    
    # Quadrant identifiers
    QUADRANTS = ['Q1', 'Q2', 'Q3', 'Q4', 'Global']
    
    # File extensions
    GERBER_EXTENSIONS = ['.gbr', '.gdo']
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    
    # Processing options
    USE_COMPRESSION = True
    PARALLEL_WORKERS = 4
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "pngs").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "arrays").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "processed").mkdir(exist_ok=True)
    
    @classmethod
    def validate_gerbv(cls) -> bool:
        """Check if gerbv.exe exists."""
        return cls.GERBV_PATH.exists()

# Made with Bob
