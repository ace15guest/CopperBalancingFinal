# Usage Examples

Practical examples demonstrating how to use the Copper Balancing library.

## Table of Contents

1. [Basic Blur Operations](#basic-blur-operations)
2. [Gerber to PNG Conversion](#gerber-to-png-conversion)
3. [Batch Processing](#batch-processing)
4. [Complete Pipeline](#complete-pipeline)
5. [Working with NPZ Files](#working-with-npz-files)
6. [Custom Processing](#custom-processing)

---

## Basic Blur Operations

### Example 1: Simple Box Blur

```python
from lib.conversion_operations import load_png
from lib.array_operations import box_blur
import matplotlib.pyplot as plt

# Load image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Apply box blur
blurred = box_blur(array, kernel_size=5)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(array, cmap='gray')
ax1.set_title('Original')
ax2.imshow(blurred, cmap='gray')
ax2.set_title('Box Blur (kernel=5)')
plt.show()
```

### Example 2: Gaussian Blur with Different Sigmas

```python
from lib.conversion_operations import load_png
from lib.array_operations import gaussian_blur
import matplotlib.pyplot as plt

# Load image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Apply different blur strengths
blur_weak = gaussian_blur(array, kernel_size=5, sigma=0.5)
blur_medium = gaussian_blur(array, kernel_size=5, sigma=1.5)
blur_strong = gaussian_blur(array, kernel_size=7, sigma=3.0)

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(array, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(blur_weak, cmap='gray')
axes[0, 1].set_title('Weak Blur (σ=0.5)')
axes[1, 0].imshow(blur_medium, cmap='gray')
axes[1, 0].set_title('Medium Blur (σ=1.5)')
axes[1, 1].imshow(blur_strong, cmap='gray')
axes[1, 1].set_title('Strong Blur (σ=3.0)')
plt.tight_layout()
plt.show()
```

### Example 3: Comparing Box vs Gaussian Blur

```python
from lib.conversion_operations import load_png
from lib.array_operations import box_blur, gaussian_blur

# Load image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Apply both blur types
box_result = box_blur(array, kernel_size=7)
gauss_result = gaussian_blur(array, kernel_size=7, sigma=1.5)

# Calculate difference
difference = abs(box_result - gauss_result)

print(f"Max difference: {difference.max():.4f}")
print(f"Mean difference: {difference.mean():.4f}")
```

---

## Gerber to PNG Conversion

### Example 4: Convert Single Gerber File

```python
from lib.conversion_operations import gerber_to_png

# Convert single file
png_path = gerber_to_png(
    gerber_path="Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr",
    output_path="output/pngs/layer2.png",
    dpi=300
)

print(f"PNG saved to: {png_path}")
```

### Example 5: Batch Convert Gerber Directory

```python
from lib.conversion_operations import batch_gerber_to_png
import os

# Convert all Gerber files in Q1 directory
png_files = batch_gerber_to_png(
    gerber_dir="Assets/gerbers/Cu_Balancing_Gerber/Q1",
    output_dir="output/Q1_pngs",
    dpi=300
)

print(f"Converted {len(png_files)} files:")
for png in png_files:
    print(f"  - {os.path.basename(png)}")
```

### Example 6: Multi-Resolution Conversion

```python
from lib.conversion_operations import gerber_to_png

gerber_file = "Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr"

# Generate multiple resolutions
for dpi in [100, 200, 300, 500]:
    output_path = f"output/layer2_dpi{dpi}.png"
    gerber_to_png(gerber_file, output_path, dpi=dpi)
    print(f"Generated {dpi} DPI version")
```

---

## Batch Processing

### Example 7: Process All Quadrants

```python
from lib.gerber_operations import get_quadrant_layers, process_layer_stack
from lib.array_operations import gaussian_blur
from lib.conversion_operations import save_npz

# Define processing operation
def blur_operation(array):
    return gaussian_blur(array, kernel_size=5, sigma=1.5)

# Get all quadrant layers
base_dir = "Assets/gerbers/Cu_Balancing_Gerber"
quadrants = get_quadrant_layers(base_dir)

# Process each quadrant
for quadrant, layers in quadrants.items():
    print(f"Processing {quadrant}...")
    
    # Process layer stack
    processed = process_layer_stack(layers, [blur_operation])
    
    # Save results
    output_file = f"output/{quadrant}_blurred.npz"
    save_npz(output_file, {
        f'layer_{i}': arr for i, arr in enumerate(processed)
    })
    
    print(f"  Saved {len(processed)} layers to {output_file}")
```

### Example 8: Parallel Processing with Custom Operations

```python
from lib.gerber_operations import parallel_process
from lib.conversion_operations import load_png
from lib.array_operations import gaussian_blur, normalize_array
import glob

def process_single_file(filepath):
    """Process a single PNG file"""
    # Load
    array = load_png(filepath)
    
    # Blur
    blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)
    
    # Normalize
    normalized = normalize_array(blurred, 0, 1)
    
    return normalized

# Get all PNG files
png_files = glob.glob("Assets/processed_pngs/Q1_dpi_300/*.png")

# Process in parallel with 4 workers
results = parallel_process(
    files=png_files,
    operation=process_single_file,
    workers=4
)

print(f"Processed {len(results)} files in parallel")
```

---

## Complete Pipeline

### Example 9: End-to-End Processing Pipeline

```python
from lib.conversion_operations import (
    gerber_to_png, 
    load_png, 
    save_npz
)
from lib.array_operations import (
    gaussian_blur, 
    normalize_array,
    calculate_statistics
)
from lib.gerber_operations import extract_layer_info
import os

def complete_pipeline(gerber_path, output_dir, dpi=300):
    """Complete processing pipeline for a Gerber file"""
    
    # Step 1: Extract layer info
    layer_info = extract_layer_info(gerber_path)
    print(f"Processing layer {layer_info['layer']}: {layer_info['type']}")
    
    # Step 2: Convert to PNG
    png_path = os.path.join(output_dir, "temp.png")
    gerber_to_png(gerber_path, png_path, dpi=dpi)
    
    # Step 3: Load as array
    array = load_png(png_path)
    
    # Step 4: Apply processing
    blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)
    normalized = normalize_array(blurred, 0, 1)
    
    # Step 5: Calculate statistics
    stats = calculate_statistics(normalized)
    
    # Step 6: Save results
    output_file = os.path.join(output_dir, f"layer_{layer_info['layer']}.npz")
    save_npz(output_file, {
        'original': array,
        'blurred': blurred,
        'normalized': normalized,
        'metadata': layer_info,
        'statistics': stats
    })
    
    # Cleanup temp file
    os.remove(png_path)
    
    return output_file, stats

# Run pipeline
gerber_file = "Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr"
output_file, stats = complete_pipeline(gerber_file, "output/processed")

print(f"Results saved to: {output_file}")
print(f"Statistics: {stats}")
```

### Example 10: Multi-Layer Analysis Pipeline

```python
from lib.gerber_operations import get_layer_stack
from lib.conversion_operations import gerber_to_png, load_png, save_array_stack
from lib.array_operations import gaussian_blur
import numpy as np

def analyze_layer_stack(quadrant, base_dir, output_dir):
    """Analyze complete layer stack for a quadrant"""
    
    # Get layers
    layers = get_layer_stack(quadrant, base_dir)
    print(f"Found {len(layers)} layers for {quadrant}")
    
    # Process each layer
    processed_arrays = []
    layer_stats = []
    
    for i, gerber_path in enumerate(layers):
        print(f"Processing layer {i+1}/{len(layers)}...")
        
        # Convert and load
        png_path = f"output/temp_layer_{i}.png"
        gerber_to_png(gerber_path, png_path, dpi=300)
        array = load_png(png_path)
        
        # Apply blur
        blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)
        processed_arrays.append(blurred)
        
        # Calculate coverage
        coverage = (blurred > 0.5).sum() / blurred.size
        layer_stats.append({
            'layer': i,
            'coverage': coverage,
            'mean': blurred.mean(),
            'std': blurred.std()
        })
    
    # Stack arrays
    stack = np.stack(processed_arrays, axis=0)
    
    # Calculate layer-wise statistics
    mean_per_layer = stack.mean(axis=(1, 2))
    total_coverage = (stack > 0.5).sum(axis=0)
    
    # Save results
    metadata = {
        'quadrant': quadrant,
        'num_layers': len(layers),
        'layer_stats': layer_stats,
        'mean_per_layer': mean_per_layer,
        'total_coverage': total_coverage
    }
    
    output_file = f"{output_dir}/{quadrant}_stack.npz"
    save_array_stack(output_file, processed_arrays, metadata)
    
    return output_file, metadata

# Analyze Q1
output_file, metadata = analyze_layer_stack(
    'Q1',
    'Assets/gerbers/Cu_Balancing_Gerber',
    'output/analysis'
)

print(f"Analysis complete: {output_file}")
print(f"Average coverage per layer: {metadata['mean_per_layer']}")
```

---

## Working with NPZ Files

### Example 11: Load and Visualize NPZ Data

```python
from lib.conversion_operations import load_npz
import matplotlib.pyplot as plt

# Load existing NPZ file
data = load_npz("Assets/processed_pngs/Q1_dpi_300/Q1_dpi_300.npz")

# Display available arrays
print("Available arrays:", list(data.keys()))

# Visualize first few layers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (key, array) in enumerate(list(data.items())[:6]):
    axes[i].imshow(array, cmap='gray')
    axes[i].set_title(key)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

### Example 12: Combine Multiple NPZ Files

```python
from lib.conversion_operations import load_npz, save_npz
import numpy as np

# Load multiple NPZ files
q1_data = load_npz("Assets/processed_pngs/Q1_dpi_300/Q1_dpi_300.npz")
q2_data = load_npz("Assets/processed_pngs/Q2_dpi_300/Q2_dpi_300.npz")

# Combine data
combined = {}
for key, array in q1_data.items():
    combined[f'Q1_{key}'] = array
for key, array in q2_data.items():
    combined[f'Q2_{key}'] = array

# Save combined
save_npz("output/combined_Q1_Q2.npz", combined, compressed=True)

print(f"Combined {len(combined)} arrays")
```

---

## Custom Processing

### Example 13: Custom Blur Kernel

```python
from lib.array_operations import separable_blur
from lib.conversion_operations import load_png
import numpy as np

# Load image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Create custom kernel (e.g., sharpening)
custom_kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# Apply custom blur
result = separable_blur(array, custom_kernel)

print(f"Applied custom kernel: {custom_kernel}")
```

### Example 14: Multi-Stage Processing

```python
from lib.conversion_operations import load_png, save_npz
from lib.array_operations import (
    gaussian_blur,
    normalize_array,
    threshold,
    calculate_statistics
)

# Load image
array = load_png("Assets/processed_pngs/Q1_dpi_300/layer2.png")

# Stage 1: Normalize
normalized = normalize_array(array, 0, 1)

# Stage 2: Light blur to reduce noise
denoised = gaussian_blur(normalized, kernel_size=3, sigma=0.5)

# Stage 3: Threshold to binary
binary = threshold(denoised, threshold_value=0.5)

# Stage 4: Blur binary for smooth edges
smoothed = gaussian_blur(binary.astype(float), kernel_size=5, sigma=1.0)

# Save all stages
save_npz("output/multi_stage.npz", {
    'stage1_normalized': normalized,
    'stage2_denoised': denoised,
    'stage3_binary': binary,
    'stage4_smoothed': smoothed
})

# Calculate statistics for each stage
for name, arr in [
    ('normalized', normalized),
    ('denoised', denoised),
    ('binary', binary),
    ('smoothed', smoothed)
]:
    stats = calculate_statistics(arr)
    print(f"{name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
```

### Example 15: Adaptive Processing Based on Layer Type

```python
from lib.gerber_operations import extract_layer_info
from lib.conversion_operations import gerber_to_png, load_png
from lib.array_operations import gaussian_blur, box_blur

def adaptive_blur(gerber_path, output_path):
    """Apply different blur based on layer type"""
    
    # Extract layer info
    info = extract_layer_info(gerber_path)
    layer_type = info['type']
    
    # Convert to PNG
    png_path = "temp.png"
    gerber_to_png(gerber_path, png_path, dpi=300)
    array = load_png(png_path)
    
    # Apply appropriate blur
    if layer_type == 'plane':
        # Planes need more smoothing
        result = gaussian_blur(array, kernel_size=7, sigma=2.0)
        print(f"Applied strong Gaussian blur for plane layer")
    elif layer_type == 'signal':
        # Signals need less smoothing to preserve traces
        result = gaussian_blur(array, kernel_size=3, sigma=0.8)
        print(f"Applied light Gaussian blur for signal layer")
    else:
        # Default: medium blur
        result = box_blur(array, kernel_size=5)
        print(f"Applied box blur for unknown layer type")
    
    return result

# Process with adaptive blur
result = adaptive_blur(
    "Assets/gerbers/Q1/l2_plane_1oz_Q1.gbr",
    "output/adaptive_result.png"
)
```

---

## Performance Tips

### Example 16: Efficient Batch Processing

```python
from lib.gerber_operations import parallel_process
from lib.array_operations import gaussian_blur
from lib.conversion_operations import load_png, save_npz
import time

def process_with_timing(filepath):
    """Process file and measure time"""
    start = time.time()
    
    array = load_png(filepath)
    blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)
    
    elapsed = time.time() - start
    return blurred, elapsed

# Get files
import glob
files = glob.glob("Assets/processed_pngs/Q1_dpi_300/*.png")

# Sequential processing
start = time.time()
sequential_results = [process_with_timing(f) for f in files]
sequential_time = time.time() - start

# Parallel processing
start = time.time()
parallel_results = parallel_process(
    files,
    lambda f: process_with_timing(f)[0],
    workers=4
)
parallel_time = time.time() - start

print(f"Sequential: {sequential_time:.2f}s")
print(f"Parallel: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time/parallel_time:.2f}x")
```

---

## Integration Examples

### Example 17: Integration with Existing Code

```python
# If you have existing DAT conversion code
def your_existing_dat_converter(dat_path):
    """Your existing DAT to array function"""
    # Your implementation here
    pass

# Integrate with new blur operations
from lib.array_operations import gaussian_blur
from lib.conversion_operations import save_npz

# Use your existing converter
array = your_existing_dat_converter("data/layer.dat")

# Apply new blur operations
blurred = gaussian_blur(array, kernel_size=5, sigma=1.5)

# Save with new NPZ handler
save_npz("output/result.npz", {
    'original': array,
    'blurred': blurred
})
```

### Example 18: Custom Configuration

```python
from src.config import Config

# Override default settings
Config.DEFAULT_DPI = 500
Config.DEFAULT_BLUR_KERNEL = 7
Config.DEFAULT_GAUSSIAN_SIGMA = 2.0

# Now all operations use new defaults
from lib.conversion_operations import gerber_to_png
from lib.array_operations import gaussian_blur

# Uses DPI=500
gerber_to_png("input.gbr", "output.png")

# Uses kernel=7, sigma=2.0
blurred = gaussian_blur(array)
```

---

## Troubleshooting Examples

### Example 19: Handling Large Files

```python
from lib.conversion_operations import load_png
from lib.array_operations import gaussian_blur
import numpy as np

# For very large files, process in chunks
def process_large_array(filepath, chunk_size=1000):
    """Process large array in chunks"""
    array = load_png(filepath)
    h, w = array.shape
    
    result = np.zeros_like(array)
    
    for i in range(0, h, chunk_size):
        for j in range(0, w, chunk_size):
            # Extract chunk with padding
            chunk = array[
                max(0, i-10):min(h, i+chunk_size+10),
                max(0, j-10):min(w, j+chunk_size+10)
            ]
            
            # Process chunk
            blurred_chunk = gaussian_blur(chunk, kernel_size=5, sigma=1.5)
            
            # Place back (remove padding)
            result[i:i+chunk_size, j:j+chunk_size] = blurred_chunk[10:-10, 10:-10]
    
    return result
```

### Example 20: Validation and Error Checking

```python
from lib.array_operations import validate_array, gaussian_blur
from lib.conversion_operations import load_png
import numpy as np

def safe_blur(filepath, kernel_size=5, sigma=1.5):
    """Blur with validation and error handling"""
    try:
        # Load
        array = load_png(filepath)
        
        # Validate
        if not validate_array(array):
            raise ValueError("Invalid array")
        
        # Check dimensions
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {array.ndim}D")
        
        # Check size
        if array.size > 10000 * 10000:
            print("Warning: Large array, processing may be slow")
        
        # Apply blur
        result = gaussian_blur(array, kernel_size, sigma)
        
        # Validate result
        assert result.shape == array.shape, "Shape mismatch"
        assert not np.isnan(result).any(), "NaN values in result"
        
        return result
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Use safe version
result = safe_blur("Assets/processed_pngs/Q1_dpi_300/layer2.png")
```

---

## Summary

These examples demonstrate:
- Basic blur operations (box and Gaussian)
- Gerber file conversion workflows
- Batch and parallel processing
- Complete end-to-end pipelines
- NPZ file handling
- Custom processing techniques
- Performance optimization
- Error handling and validation

For more details, see:
- [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation
- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) - Project structure and design
- [`REQUIREMENTS.md`](REQUIREMENTS.md) - Dependencies and installation