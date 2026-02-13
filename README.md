# Copper Balancing Image Processing

A modular Python framework for processing PCB Gerber files with advanced image processing operations, specifically designed for copper balancing analysis.

## Overview

This project provides a comprehensive toolkit for:
- Converting Gerber files to PNG images using gerbv
- Loading and manipulating image arrays
- Applying blur filters (Box and Gaussian)
- Batch processing multiple layers and quadrants
- Efficient NPZ file handling for processed data

## Features

- **Modular Architecture**: Three main libraries (conversion_operations, array_operations, gerber_operations)
- **Efficient Blur Algorithms**: Separable convolution for O(n) performance
- **Batch Processing**: Parallel processing support for large datasets
- **Type-Safe**: Full type hints throughout the codebase
- **Well-Documented**: Comprehensive API documentation and examples

## Quick Start

### Installation

```bash
# Clone the repository
cd CopperBalancingFinal

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from lib.conversion_operations import load_png, save_npz
from lib.array_operations import gaussian_blur

# Load an image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Apply Gaussian blur
blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)

# Save result
save_npz("output/blurred.npz", {
    'original': array,
    'blurred': blurred
})
```

### Convert Gerber to PNG

```python
from lib.conversion_operations import gerber_to_png

# Convert single Gerber file
png_path = gerber_to_png(
    gerber_path="Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr",
    output_path="output/layer2.png",
    dpi=300
)
```

### Batch Processing

```python
from lib.gerber_operations import get_quadrant_layers, process_layer_stack
from lib.array_operations import gaussian_blur

# Get all layers for Q1
layers = get_quadrant_layers("Assets/gerbers/Cu_Balancing_Gerber")['Q1']

# Process with blur
operations = [lambda arr: gaussian_blur(arr, 5, 1.5)]
results = process_layer_stack(layers, operations)
```

## Project Structure

```
CopperBalancingFinal/
├── lib/
│   ├── conversion_operations/    # File format conversions
│   │   ├── dat_converter.py
│   │   ├── gerber_converter.py
│   │   ├── png_loader.py
│   │   └── npz_handler.py
│   │
│   ├── array_operations/         # Image processing
│   │   ├── blur_filters.py
│   │   ├── array_utils.py
│   │   └── image_processing.py
│   │
│   └── gerber_operations/        # Gerber-specific ops
│       ├── gerber_parser.py
│       ├── layer_manager.py
│       └── batch_processor.py
│
├── src/
│   ├── main.py                   # Main entry point
│   └── config.py                 # Configuration
│
├── Assets/                       # Input data
│   ├── gerbers/                  # Gerber files
│   ├── gerbv/                    # gerbv.exe tool
│   └── processed_pngs/           # Existing processed data
│
├── output/                       # Generated outputs
├── tests/                        # Unit tests
├── examples/                     # Example scripts
└── docs/                         # Documentation
```

## Core Modules

### conversion_operations

Handles file I/O and format conversions:
- **dat_converter**: DAT file to numpy array
- **gerber_converter**: Gerber to PNG using gerbv
- **png_loader**: PNG to numpy array
- **npz_handler**: Efficient NPZ save/load

### array_operations

Image processing operations:
- **blur_filters**: Box and Gaussian blur implementations
- **array_utils**: Normalization, padding, cropping
- **image_processing**: Thresholding, masking, statistics

### gerber_operations

Gerber-specific functionality:
- **gerber_parser**: Extract layer metadata
- **layer_manager**: Organize layer stackups
- **batch_processor**: Parallel batch processing

## Blur Algorithms

### Box Blur
- Simple averaging filter with uniform weights
- Fast: O(n) with separable implementation
- Good for: Quick smoothing, noise reduction

```python
from lib.array_operations import box_blur
result = box_blur(array, kernel_size=5)
```

### Gaussian Blur
- Weighted averaging using Gaussian distribution
- Natural smoothing with edge preservation
- Configurable sigma parameter for blur strength

```python
from lib.array_operations import gaussian_blur
result = gaussian_blur(array, kernel_size=5, sigma=1.5)
```

## Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Detailed project structure and design
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[EXAMPLES.md](EXAMPLES.md)** - 20+ usage examples
- **[REQUIREMENTS.md](REQUIREMENTS.md)** - Dependencies and installation

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Pillow >= 10.0.0
- Windows OS (for gerbv.exe)

See [REQUIREMENTS.md](REQUIREMENTS.md) for complete details.

## Examples

### Example 1: Compare Blur Methods

```python
from lib.conversion_operations import load_png
from lib.array_operations import box_blur, gaussian_blur

array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Apply both methods
box_result = box_blur(array, kernel_size=5)
gauss_result = gaussian_blur(array, kernel_size=5, sigma=1.5)

# Compare
difference = abs(box_result - gauss_result)
print(f"Mean difference: {difference.mean():.4f}")
```

### Example 2: Process All Quadrants

```python
from lib.gerber_operations import get_quadrant_layers
from lib.array_operations import gaussian_blur
from lib.conversion_operations import save_npz

quadrants = get_quadrant_layers("Assets/gerbers/Cu_Balancing_Gerber")

for quadrant, layers in quadrants.items():
    print(f"Processing {quadrant}...")
    # Process layers...
```

### Example 3: Multi-Resolution Conversion

```python
from lib.conversion_operations import gerber_to_png

gerber_file = "Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr"

for dpi in [100, 200, 300, 500]:
    output = f"output/layer2_dpi{dpi}.png"
    gerber_to_png(gerber_file, output, dpi=dpi)
```

See [EXAMPLES.md](EXAMPLES.md) for 20+ complete examples.

## Configuration

Customize settings in [`src/config.py`](src/config.py):

```python
from src.config import Config

# Override defaults
Config.DEFAULT_DPI = 500
Config.DEFAULT_BLUR_KERNEL = 7
Config.DEFAULT_GAUSSIAN_SIGMA = 2.0
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lib --cov-report=html

# Run specific test module
pytest tests/test_array_operations/test_blur_filters.py
```

## Performance

- **Separable Convolution**: O(n) blur operations instead of O(n²)
- **Parallel Processing**: Multi-core support for batch operations
- **Efficient I/O**: Compressed NPZ format for storage
- **Vectorized Operations**: NumPy-based for maximum performance

## Workflow

```mermaid
graph LR
    A[Gerber Files] --> B[gerber_converter]
    B --> C[PNG Files]
    C --> D[png_loader]
    D --> E[NumPy Arrays]
    E --> F[blur_filters]
    F --> G[Processed Arrays]
    G --> H[npz_handler]
    H --> I[NPZ Output]
```

## Integration with Existing Code

This framework is designed to integrate with your existing DAT conversion and gerbv code:

```python
# Use your existing DAT converter
array = your_dat_converter("data/layer.dat")

# Apply new blur operations
from lib.array_operations import gaussian_blur
blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)

# Save with new NPZ handler
from lib.conversion_operations import save_npz
save_npz("output/result.npz", {'blurred': blurred})
```

## Roadmap

Future enhancements:
- [ ] Additional blur algorithms (bilateral, median)
- [ ] Edge detection and feature extraction
- [ ] Automated copper balance calculation
- [ ] GUI for interactive processing
- [ ] Support for additional Gerber formats
- [ ] Performance optimization with Numba/Cython

## Contributing

This is a structured framework designed for easy extension:

1. Add new operations to appropriate library module
2. Follow existing type hints and documentation style
3. Add unit tests for new functionality
4. Update API documentation

## License

[Specify your license here]

## Support

For questions or issues:
- See [EXAMPLES.md](EXAMPLES.md) for usage examples
- Check [API_REFERENCE.md](API_REFERENCE.md) for detailed API docs
- Review [PROJECT_PLAN.md](PROJECT_PLAN.md) for architecture details

## Acknowledgments

- Uses gerbv for Gerber file rendering
- Built with NumPy and SciPy for efficient array operations
- Designed for PCB copper balancing analysis workflows