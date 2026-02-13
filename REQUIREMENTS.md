# Project Requirements

## Python Dependencies

### Core Dependencies
```
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
Pillow>=10.0.0,<11.0.0
```

### Development Dependencies
```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
```

### Optional Dependencies
```
matplotlib>=3.7.0      # For visualization
jupyter>=1.0.0         # For notebooks
```

## System Requirements

- Python 3.8 or higher
- Windows OS (for gerbv.exe compatibility)
- Minimum 4GB RAM (8GB recommended for large arrays)
- 1GB free disk space for output files

## External Tools

- **gerbv.exe**: Located in `Assets/gerbv/gerbv.exe`
  - Used for Gerber to PNG conversion
  - Version: (check Assets/gerbv/ for version info)

## Installation

### Standard Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Editable Installation
```bash
pip install -e .
```

## requirements.txt Content
```
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
Pillow>=10.0.0,<11.0.0
matplotlib>=3.7.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

## requirements-dev.txt Content
```
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
jupyter>=1.0.0
ipython>=8.0.0
```

## Compatibility Notes

- **NumPy**: Version 1.24+ required for modern array operations
- **SciPy**: Used for advanced filtering (Gaussian blur optimization)
- **Pillow**: PNG image I/O operations
- **Matplotlib**: Optional, for visualization and debugging
- **pytest**: Testing framework with coverage support

## Version Constraints

All version constraints use semantic versioning:
- `>=X.Y.Z`: Minimum version required
- `<X.0.0`: Maximum major version (avoid breaking changes)

## Platform-Specific Notes

### Windows
- gerbv.exe is Windows-only
- Path separators handled by pathlib

### Linux/Mac
- Would require gerbv installation via package manager
- May need wine to run gerbv.exe or native gerbv build

## Future Dependencies

Potential additions for enhanced functionality:
- `opencv-python`: Advanced image processing
- `scikit-image`: Additional filters
- `numba`: JIT compilation for performance
- `dask`: Parallel array processing for very large files