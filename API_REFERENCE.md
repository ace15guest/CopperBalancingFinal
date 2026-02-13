# API Reference

Complete API documentation for the Copper Balancing Image Processing library.

## Table of Contents

1. [conversion_operations](#conversion_operations)
2. [array_operations](#array_operations)
3. [gerber_operations](#gerber_operations)
4. [Configuration](#configuration)

---

## conversion_operations

Module for file format conversions and I/O operations.

### dat_converter.py

#### `load_dat_file(filepath: str) -> np.ndarray`

Load a DAT file and convert it to a numpy array.

**Parameters:**
- `filepath` (str): Path to the DAT file

**Returns:**
- `np.ndarray`: Loaded array data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is invalid

**Example:**
```python
from lib.conversion_operations import load_dat_file

array = load_dat_file("data/layer1.dat")
print(array.shape)  # (1024, 1024)
```

#### `dat_to_array(dat_data: bytes, shape: tuple) -> np.ndarray`

Convert raw DAT bytes to numpy array with specified shape.

**Parameters:**
- `dat_data` (bytes): Raw binary data from DAT file
- `shape` (tuple): Target array shape (height, width)

**Returns:**
- `np.ndarray`: Reshaped array

**Example:**
```python
with open("data.dat", "rb") as f:
    data = f.read()
array = dat_to_array(data, (512, 512))
```

#### `validate_dat_format(filepath: str) -> bool`

Check if file is a valid DAT format.

**Parameters:**
- `filepath` (str): Path to file to validate

**Returns:**
- `bool`: True if valid DAT format

---

### gerber_converter.py

#### `gerber_to_png(gerber_path: str, output_path: str, dpi: int = 300) -> str`

Convert a Gerber file to PNG using gerbv.

**Parameters:**
- `gerber_path` (str): Path to input Gerber file (.gbr, .gdo)
- `output_path` (str): Path for output PNG file
- `dpi` (int, optional): Resolution in dots per inch. Default: 300

**Returns:**
- `str`: Path to generated PNG file

**Raises:**
- `FileNotFoundError`: If Gerber file or gerbv.exe not found
- `RuntimeError`: If conversion fails

**Example:**
```python
from lib.conversion_operations import gerber_to_png

png_path = gerber_to_png(
    "Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr",
    "output/layer2.png",
    dpi=300
)
```

#### `batch_gerber_to_png(gerber_dir: str, output_dir: str, dpi: int = 300) -> list`

Convert all Gerber files in a directory to PNG.

**Parameters:**
- `gerber_dir` (str): Directory containing Gerber files
- `output_dir` (str): Directory for output PNG files
- `dpi` (int, optional): Resolution. Default: 300

**Returns:**
- `list`: List of generated PNG file paths

**Example:**
```python
png_files = batch_gerber_to_png(
    "Assets/gerbers/Q1",
    "output/Q1_pngs",
    dpi=300
)
print(f"Converted {len(png_files)} files")
```

#### `get_gerbv_path() -> str`

Get the path to gerbv.exe from configuration.

**Returns:**
- `str`: Absolute path to gerbv.exe

---

### png_loader.py

#### `load_png(filepath: str, grayscale: bool = True) -> np.ndarray`

Load a PNG file into a numpy array.

**Parameters:**
- `filepath` (str): Path to PNG file
- `grayscale` (bool, optional): Convert to grayscale. Default: True

**Returns:**
- `np.ndarray`: Image array (H, W) if grayscale, (H, W, C) if color

**Example:**
```python
from lib.conversion_operations import load_png

# Load as grayscale
gray_array = load_png("image.png", grayscale=True)

# Load as color
color_array = load_png("image.png", grayscale=False)
```

#### `png_to_array(filepath: str, normalize: bool = False) -> np.ndarray`

Load PNG and optionally normalize to [0, 1] range.

**Parameters:**
- `filepath` (str): Path to PNG file
- `normalize` (bool, optional): Normalize to [0, 1]. Default: False

**Returns:**
- `np.ndarray`: Image array, normalized if requested

**Example:**
```python
# Load with normalization
array = png_to_array("image.png", normalize=True)
print(array.min(), array.max())  # 0.0 1.0
```

#### `validate_png(filepath: str) -> bool`

Check if file is a valid PNG.

**Parameters:**
- `filepath` (str): Path to file

**Returns:**
- `bool`: True if valid PNG

---

### npz_handler.py

#### `save_npz(filepath: str, arrays: dict, compressed: bool = True) -> None`

Save multiple arrays to NPZ file.

**Parameters:**
- `filepath` (str): Output NPZ file path
- `arrays` (dict): Dictionary of array_name: array pairs
- `compressed` (bool, optional): Use compression. Default: True

**Example:**
```python
from lib.conversion_operations import save_npz

save_npz("output/results.npz", {
    'original': original_array,
    'blurred': blurred_array,
    'metadata': metadata_array
}, compressed=True)
```

#### `load_npz(filepath: str) -> dict`

Load arrays from NPZ file.

**Parameters:**
- `filepath` (str): Path to NPZ file

**Returns:**
- `dict`: Dictionary of loaded arrays

**Example:**
```python
from lib.conversion_operations import load_npz

data = load_npz("output/results.npz")
original = data['original']
blurred = data['blurred']
```

#### `save_array_stack(filepath: str, arrays: list, metadata: dict = None) -> None`

Save a stack of arrays with optional metadata.

**Parameters:**
- `filepath` (str): Output NPZ file path
- `arrays` (list): List of numpy arrays
- `metadata` (dict, optional): Metadata dictionary

**Example:**
```python
layers = [layer1, layer2, layer3]
metadata = {'dpi': 300, 'quadrant': 'Q1'}
save_array_stack("output/stack.npz", layers, metadata)
```

#### `load_array_stack(filepath: str) -> tuple`

Load array stack and metadata.

**Parameters:**
- `filepath` (str): Path to NPZ file

**Returns:**
- `tuple`: (list of arrays, metadata dict)

---

## array_operations

Module for image processing and array manipulation.

### blur_filters.py

#### `box_blur(array: np.ndarray, kernel_size: int) -> np.ndarray`

Apply box blur (uniform averaging) to array.

**Parameters:**
- `array` (np.ndarray): Input array (2D)
- `kernel_size` (int): Size of blur kernel (must be odd)

**Returns:**
- `np.ndarray`: Blurred array (same shape as input)

**Raises:**
- `ValueError`: If kernel_size is even or < 1

**Algorithm:**
- Uses separable convolution for efficiency
- Time complexity: O(n) where n is array size
- Each pixel averaged with neighbors in kernel_size × kernel_size window

**Example:**
```python
from lib.array_operations import box_blur

blurred = box_blur(array, kernel_size=5)
```

#### `gaussian_blur(array: np.ndarray, kernel_size: int, sigma: float = None) -> np.ndarray`

Apply Gaussian blur to array.

**Parameters:**
- `array` (np.ndarray): Input array (2D)
- `kernel_size` (int): Size of blur kernel (must be odd)
- `sigma` (float, optional): Standard deviation. Default: kernel_size/6

**Returns:**
- `np.ndarray`: Blurred array (same shape as input)

**Raises:**
- `ValueError`: If kernel_size is even or < 1

**Algorithm:**
- Uses 2D Gaussian kernel: exp(-(x² + y²)/(2σ²))
- Separable implementation for efficiency
- Sigma controls blur strength (larger = more blur)

**Example:**
```python
from lib.array_operations import gaussian_blur

# Default sigma
blurred = gaussian_blur(array, kernel_size=5)

# Custom sigma for stronger blur
strong_blur = gaussian_blur(array, kernel_size=7, sigma=2.0)
```

#### `separable_blur(array: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray`

Apply separable blur using 1D kernel.

**Parameters:**
- `array` (np.ndarray): Input array (2D)
- `kernel_1d` (np.ndarray): 1D convolution kernel

**Returns:**
- `np.ndarray`: Blurred array

**Note:** This is a low-level function. Use `box_blur` or `gaussian_blur` instead.

#### `create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray`

Create a 1D Gaussian kernel.

**Parameters:**
- `kernel_size` (int): Kernel size (must be odd)
- `sigma` (float): Standard deviation

**Returns:**
- `np.ndarray`: Normalized 1D Gaussian kernel

**Example:**
```python
kernel = create_gaussian_kernel(5, 1.0)
print(kernel)  # [0.06, 0.24, 0.40, 0.24, 0.06]
```

#### `create_box_kernel(kernel_size: int) -> np.ndarray`

Create a 1D box (uniform) kernel.

**Parameters:**
- `kernel_size` (int): Kernel size

**Returns:**
- `np.ndarray`: Normalized 1D box kernel

---

### array_utils.py

#### `normalize_array(array: np.ndarray, min_val: float = 0, max_val: float = 1) -> np.ndarray`

Normalize array to specified range.

**Parameters:**
- `array` (np.ndarray): Input array
- `min_val` (float, optional): Target minimum. Default: 0
- `max_val` (float, optional): Target maximum. Default: 1

**Returns:**
- `np.ndarray`: Normalized array

**Example:**
```python
from lib.array_operations import normalize_array

# Normalize to [0, 1]
normalized = normalize_array(array)

# Normalize to [0, 255]
scaled = normalize_array(array, 0, 255)
```

#### `pad_array(array: np.ndarray, padding: int, mode: str = 'reflect') -> np.ndarray`

Add padding to array edges.

**Parameters:**
- `array` (np.ndarray): Input array
- `padding` (int): Number of pixels to pad on each side
- `mode` (str, optional): Padding mode ('reflect', 'constant', 'edge'). Default: 'reflect'

**Returns:**
- `np.ndarray`: Padded array

**Example:**
```python
from lib.array_operations import pad_array

padded = pad_array(array, padding=10, mode='reflect')
```

#### `crop_array(array: np.ndarray, bounds: tuple) -> np.ndarray`

Crop array to specified bounds.

**Parameters:**
- `array` (np.ndarray): Input array
- `bounds` (tuple): (top, left, bottom, right) pixel coordinates

**Returns:**
- `np.ndarray`: Cropped array

**Example:**
```python
from lib.array_operations import crop_array

# Crop to center 512x512 region
h, w = array.shape
top = (h - 512) // 2
left = (w - 512) // 2
cropped = crop_array(array, (top, left, top + 512, left + 512))
```

#### `validate_array(array: np.ndarray, expected_shape: tuple = None) -> bool`

Validate array properties.

**Parameters:**
- `array` (np.ndarray): Array to validate
- `expected_shape` (tuple, optional): Expected shape

**Returns:**
- `bool`: True if valid

#### `convert_dtype(array: np.ndarray, target_dtype: np.dtype) -> np.ndarray`

Convert array to target data type.

**Parameters:**
- `array` (np.ndarray): Input array
- `target_dtype` (np.dtype): Target data type

**Returns:**
- `np.ndarray`: Converted array

---

### image_processing.py

#### `threshold(array: np.ndarray, threshold_value: float) -> np.ndarray`

Apply binary threshold to array.

**Parameters:**
- `array` (np.ndarray): Input array
- `threshold_value` (float): Threshold value

**Returns:**
- `np.ndarray`: Binary array (0 or 1)

**Example:**
```python
from lib.array_operations import threshold

binary = threshold(array, threshold_value=0.5)
```

#### `apply_mask(array: np.ndarray, mask: np.ndarray) -> np.ndarray`

Apply binary mask to array.

**Parameters:**
- `array` (np.ndarray): Input array
- `mask` (np.ndarray): Binary mask (same shape)

**Returns:**
- `np.ndarray`: Masked array

#### `calculate_statistics(array: np.ndarray) -> dict`

Calculate array statistics.

**Parameters:**
- `array` (np.ndarray): Input array

**Returns:**
- `dict`: Statistics including mean, std, min, max, median

**Example:**
```python
from lib.array_operations import calculate_statistics

stats = calculate_statistics(array)
print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
```

---

## gerber_operations

Module for Gerber-specific operations.

### gerber_parser.py

#### `parse_gerber_header(filepath: str) -> dict`

Parse Gerber file header information.

**Parameters:**
- `filepath` (str): Path to Gerber file

**Returns:**
- `dict`: Header information

#### `extract_layer_info(filepath: str) -> dict`

Extract layer metadata from Gerber file.

**Parameters:**
- `filepath` (str): Path to Gerber file

**Returns:**
- `dict`: Layer information (number, type, weight)

**Example:**
```python
from lib.gerber_operations import extract_layer_info

info = extract_layer_info("Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr")
print(info)  # {'layer': 2, 'type': 'plane', 'weight': '1oz', 'quadrant': 'Q1'}
```

#### `get_layer_type(filename: str) -> str`

Determine layer type from filename.

**Parameters:**
- `filename` (str): Gerber filename

**Returns:**
- `str`: Layer type ('signal', 'plane', 'unknown')

#### `parse_layer_name(filename: str) -> tuple`

Parse layer information from filename.

**Parameters:**
- `filename` (str): Gerber filename

**Returns:**
- `tuple`: (layer_number, layer_type, weight, quadrant)

---

### layer_manager.py

#### `organize_layers(gerber_dir: str) -> dict`

Organize Gerber files by quadrant and layer.

**Parameters:**
- `gerber_dir` (str): Directory containing Gerber files

**Returns:**
- `dict`: Organized layer structure

**Example:**
```python
from lib.gerber_operations import organize_layers

layers = organize_layers("Assets/gerbers/Cu_Balancing_Gerber")
print(layers.keys())  # ['Q1', 'Q2', 'Q3', 'Q4', 'UL', 'UR', 'LL', 'LR']
```

#### `get_layer_stack(quadrant: str, gerber_dir: str) -> list`

Get ordered list of layers for a quadrant.

**Parameters:**
- `quadrant` (str): Quadrant identifier ('Q1', 'Q2', etc.)
- `gerber_dir` (str): Base Gerber directory

**Returns:**
- `list`: Ordered list of layer file paths

**Example:**
```python
from lib.gerber_operations import get_layer_stack

q1_layers = get_layer_stack('Q1', 'Assets/gerbers/Cu_Balancing_Gerber')
print(f"Q1 has {len(q1_layers)} layers")
```

#### `validate_layer_stack(layers: list) -> bool`

Validate that layer stack is complete and ordered.

**Parameters:**
- `layers` (list): List of layer file paths

**Returns:**
- `bool`: True if valid

#### `get_quadrant_layers(base_dir: str) -> dict`

Get all layers organized by quadrant.

**Parameters:**
- `base_dir` (str): Base directory containing quadrant subdirectories

**Returns:**
- `dict`: Quadrant to layer list mapping

---

### batch_processor.py

#### `process_directory(input_dir: str, output_dir: str, operations: list) -> dict`

Process all files in directory with specified operations.

**Parameters:**
- `input_dir` (str): Input directory
- `output_dir` (str): Output directory
- `operations` (list): List of functions to apply

**Returns:**
- `dict`: Processing results and statistics

**Example:**
```python
from lib.gerber_operations import process_directory
from lib.array_operations import gaussian_blur

operations = [
    lambda arr: gaussian_blur(arr, 5, 1.5)
]

results = process_directory(
    "Assets/gerbers/Q1",
    "output/Q1_processed",
    operations
)
```

#### `process_layer_stack(layers: list, operations: list) -> list`

Process a stack of layers with operations.

**Parameters:**
- `layers` (list): List of layer file paths
- `operations` (list): List of processing functions

**Returns:**
- `list`: Processed arrays

#### `parallel_process(files: list, operation: callable, workers: int = 4) -> list`

Process files in parallel.

**Parameters:**
- `files` (list): List of file paths
- `operation` (callable): Function to apply to each file
- `workers` (int, optional): Number of parallel workers. Default: 4

**Returns:**
- `list`: Processing results

---

## Configuration

### config.py

Configuration class for project settings.

#### Class: `Config`

**Attributes:**

- `GERBV_PATH` (str): Path to gerbv.exe
- `ASSETS_DIR` (str): Assets directory path
- `OUTPUT_DIR` (str): Output directory path
- `DEFAULT_DPI` (int): Default DPI for conversions
- `DEFAULT_BLUR_KERNEL` (int): Default blur kernel size
- `DEFAULT_GAUSSIAN_SIGMA` (float): Default Gaussian sigma
- `LAYER_PATTERNS` (dict): Regex patterns for layer types
- `QUADRANTS` (list): Valid quadrant identifiers

**Example:**
```python
from src.config import Config

print(Config.DEFAULT_DPI)  # 300
print(Config.QUADRANTS)    # ['Q1', 'Q2', 'Q3', 'Q4', ...]
```

---

## Type Hints

All functions use Python type hints for clarity:

```python
def function_name(
    param1: type1,
    param2: type2 = default
) -> return_type:
    pass
```

## Error Handling

All functions raise appropriate exceptions:
- `FileNotFoundError`: File not found
- `ValueError`: Invalid parameter value
- `RuntimeError`: Operation failed
- `TypeError`: Wrong type provided

## Performance Notes

- Blur operations use separable convolution: O(n) instead of O(n²)
- Batch operations support parallel processing
- Large arrays handled efficiently with numpy vectorization
- NPZ files use compression by default