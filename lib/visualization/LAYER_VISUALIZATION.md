# Layer Visualization Guide

This guide explains how to use the layer visualization functions to inspect individual layers and understand the results of operations like `center_crop_by_area`.

## Overview

The `layer_visualizer` module provides functions to:
- Visualize single 2D array layers with statistics
- Compare original and cropped arrays side-by-side
- Load and visualize layers from NPZ files
- Calculate comprehensive layer statistics

## Quick Start

### Basic Layer Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from lib.visualization.layer_visualizer import visualize_single_layer

# Create or load your array
layer = np.random.rand(100, 100)

# Visualize it
fig = visualize_single_layer(
    layer,
    layer_name="Signal Layer 3",
    show_stats=True,
    save_path="output/layer_viz.png"
)
plt.show()
```

### Comparing Crop Results

```python
from lib.array_operations.border_operations import center_crop_by_area
from lib.visualization.layer_visualizer import compare_crop_result

# Original array
original = np.random.rand(100, 100)

# Crop to 25% area
cropped = center_crop_by_area(original, pct_area=0.25)

# Compare them
fig = compare_crop_result(
    original,
    cropped,
    layer_name="Signal Layer 3",
    save_path="output/comparison.png"
)
plt.show()
```

### Visualizing from NPZ Files

```python
from lib.visualization.layer_visualizer import visualize_layer_from_npz

# Visualize a specific layer from NPZ
fig = visualize_layer_from_npz(
    "Assets/ProcessedPNGs/board1_dpi_600/board1_dpi_600.npz",
    "signal_layer_3_1oz",
    show_stats=True
)
plt.show()
```

## Function Reference

### `visualize_single_layer()`

Visualize a single 2D array layer with statistics and percentage information.

**Parameters:**
- `array` (np.ndarray): 2D numpy array to visualize
- `layer_name` (str): Name/label for the layer (default: "Layer")
- `show_stats` (bool): Whether to display statistics on the plot (default: True)
- `save_path` (str, optional): Path to save the figure
- `figsize` (tuple): Figure size (width, height) in inches (default: (12, 10))
- `cmap` (str): Colormap to use (default: 'viridis')
- `show_colorbar` (bool): Whether to show the colorbar (default: True)

**Returns:**
- `matplotlib.Figure`: The created figure object

**Statistics Displayed:**
- Min, Max, Mean, Median, Standard Deviation
- Percentage of non-zero values
- Percentage of valid (finite) values

### `compare_crop_result()`

Compare original and cropped arrays side-by-side with area percentage.

**Parameters:**
- `original` (np.ndarray): Original 2D array before cropping
- `cropped` (np.ndarray): Cropped 2D array
- `layer_name` (str): Name/label for the layer (default: "Layer")
- `save_path` (str, optional): Path to save the figure
- `figsize` (tuple): Figure size (width, height) in inches (default: (16, 6))
- `cmap` (str): Colormap to use (default: 'viridis')

**Returns:**
- `matplotlib.Figure`: The created figure object

### `visualize_layer_from_npz()`

Load and visualize a specific layer from an NPZ file.

**Parameters:**
- `npz_path` (str): Path to the NPZ file
- `layer_key` (str): Key name of the layer to visualize
- `show_stats` (bool): Whether to display statistics (default: True)
- `save_path` (str, optional): Path to save the figure
- `figsize` (tuple): Figure size (width, height) in inches (default: (12, 10))
- `cmap` (str): Colormap to use (default: 'viridis')

**Returns:**
- `matplotlib.Figure`: The created figure object

### `calculate_layer_statistics()`

Calculate comprehensive statistics for a 2D array layer.

**Parameters:**
- `array` (np.ndarray): 2D numpy array

**Returns:**
- `dict`: Dictionary containing:
  - `min`: Minimum value
  - `max`: Maximum value
  - `mean`: Mean value
  - `median`: Median value
  - `std`: Standard deviation
  - `non_zero_pct`: Percentage of non-zero values
  - `valid_pct`: Percentage of valid (finite) values

## Use Cases

### 1. Debugging Parameter Sweep

When running parameter sweeps, you can inspect what's happening at each step:

```python
from lib.array_operations.border_operations import center_crop_by_area
from lib.visualization.layer_visualizer import visualize_single_layer

# After processing in parameter_sweep.py
calculated_layers_blended_shrink_rescaled_cropped = center_crop_by_area(
    calculated_layers_blended_shrink_rescale, 
    pct_area=0.50
)

# Visualize the result
fig = visualize_single_layer(
    calculated_layers_blended_shrink_rescaled_cropped,
    layer_name="Processed Layer (50% crop)",
    show_stats=True
)
plt.show()
```

### 2. Understanding Crop Percentages

Test different crop percentages to understand their effect:

```python
import matplotlib.pyplot as plt
from lib.array_operations.border_operations import center_crop_by_area

original = np.random.rand(100, 100)
crop_percentages = [0.75, 0.50, 0.25, 0.10]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, pct in enumerate(crop_percentages):
    cropped = center_crop_by_area(original, pct_area=pct)
    axes[idx].imshow(cropped, cmap='viridis')
    axes[idx].set_title(f'{pct*100:.0f}% Area: {cropped.shape}')

plt.tight_layout()
plt.show()
```

### 3. Inspecting NPZ Layers

Quickly check what's in your processed NPZ files:

```python
import numpy as np
from lib.visualization.layer_visualizer import visualize_layer_from_npz

npz_path = "Assets/ProcessedPNGs/board1_dpi_600/board1_dpi_600.npz"

# List all layers
with np.load(npz_path) as data:
    print("Available layers:")
    for key in data.files:
        print(f"  - {key}: {data[key].shape}")

# Visualize each layer
with np.load(npz_path) as data:
    for key in data.files:
        fig = visualize_layer_from_npz(npz_path, key)
        plt.show()
```

## Examples

See `examples/visualize_layer_example.py` for complete working examples including:
- Basic layer visualization
- Crop comparison
- Multiple crop percentages
- NPZ file inspection
- Parameter sweep style inspection

Run the examples:
```bash
python examples/visualize_layer_example.py
```

## Tips

1. **Choosing Colormaps**: Use 'viridis' (default) for general data, 'RdBu' for diverging data, or 'gray' for grayscale
2. **Statistics Box**: The statistics box shows key metrics to help identify issues like NaN values or unexpected ranges
3. **Saving Figures**: Always specify `save_path` to keep a record of your visualizations
4. **Large Arrays**: For very large arrays, consider downsampling before visualization for faster rendering

## Integration with Parameter Sweep

To add visualization to your parameter sweep workflow:

```python
from lib.gerber_operations.parameter_sweep import parameter_sweep_analysis
from lib.visualization.layer_visualizer import visualize_single_layer

# Run parameter sweep
results = parameter_sweep_analysis(...)

# After processing, visualize key layers
# (Add this inside the parameter_sweep_analysis function or after it)
fig = visualize_single_layer(
    calculated_layers_blended_shrink_rescaled_cropped,
    layer_name=f"{name}_cropped",
    save_path=f"output/debug/{name}_visualization.png"
)
plt.close(fig)  # Close to free memory
```

## Troubleshooting

**Issue**: "Import matplotlib.pyplot could not be resolved"
- **Solution**: Install matplotlib: `pip install matplotlib`

**Issue**: "NPZ file not found"
- **Solution**: Check the path and ensure the NPZ file exists. Use absolute paths if needed.

**Issue**: "Layer key not found in NPZ"
- **Solution**: List available keys first using `np.load(npz_path).files`

**Issue**: Figures not displaying
- **Solution**: Make sure to call `plt.show()` after creating the figure

## Made with Bob