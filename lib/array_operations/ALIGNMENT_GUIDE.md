# Array Alignment Guide

## Overview

The `alignment.py` module provides robust image registration capabilities for aligning DAT measurement arrays to Gerber reference designs. This guide explains the alignment methodology and addresses common questions about the approach.

## Table of Contents

1. [Alignment Methodology](#alignment-methodology)
2. [Why Not SVD?](#why-not-svd)
3. [Key Functions](#key-functions)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## Alignment Methodology

The alignment process uses a **multi-stage registration pipeline** optimized for dense image-to-image alignment:

### Stage 1: Orientation Estimation
- **Method**: Moment-based principal component analysis
- **Function**: `dominant_angle_safe()`
- **Purpose**: Estimates the dominant orientation angle of features in the image
- **Robustness**: Safe on sparse data (requires minimum 100 pixels)

### Stage 2: Scale Estimation
- **Method**: Normalized Cross-Correlation (NCC) search
- **Function**: `auto_scale()`
- **Purpose**: Finds optimal scale factor by maximizing similarity
- **Alternative**: Direct computation from known pixel densities (`px_per_mm_gerber`, `px_per_mm_dat`)

### Stage 3: Translation Alignment
- **Method**: Phase cross-correlation
- **Function**: `phase_translate()`
- **Purpose**: Sub-pixel accurate translation estimation
- **Accuracy**: Up to 10x upsampling for sub-pixel precision

### Stage 4: Transformation Application
- **Method**: Similarity transform with warping
- **Implementation**: `skimage.transform.warp` with `SimilarityTransform`
- **Features**: Configurable fill modes, validity masking

---

## Why Not SVD?

### Question: Should I use Singular Value Decomposition (SVD) for alignment?

**Answer: No.** Here's why:

### SVD is for Point-Set Registration

SVD (via Procrustes analysis) is appropriate when you have:
- **Discrete corresponding point pairs** (e.g., landmarks, feature matches)
- Known point-to-point correspondences
- Need to compute optimal rigid transformation between point sets

### Your Use Case: Dense Image Registration

For aligning continuous image arrays (Gerber ↔ DAT), the current approach is superior:

| Aspect | Current Approach | SVD Approach |
|--------|-----------------|--------------|
| **Input Type** | Dense pixel arrays | Discrete point pairs |
| **Correspondence** | Automatic (phase correlation) | Requires manual/feature matching |
| **Sub-pixel Accuracy** | ✅ Yes (10x upsampling) | ❌ Limited by point precision |
| **Robustness** | ✅ Handles noise, blur | ❌ Sensitive to outliers |
| **Scale Estimation** | ✅ Automatic NCC search | ❌ Requires known scale |
| **Computational Cost** | ✅ Efficient FFT-based | ❌ Requires feature extraction |

### When Would SVD Be Appropriate?

Only if you:
1. Extract corresponding feature points (SIFT, ORB, manual landmarks)
2. Have reliable point-to-point matches
3. Need Procrustes alignment for point clouds

For your Gerber-DAT alignment, **stick with phase correlation**.

---

## Key Functions

### `align_dat_to_gerber()`

Primary alignment function that registers DAT arrays to Gerber references.

```python
aligned, params, valid_mask = align_dat_to_gerber(
    gerber_arr=gerber_image,
    dat_arr=dat_measurement,
    px_per_mm_gerber=100.0,  # Optional: known pixel density
    px_per_mm_dat=50.0,      # Optional: known pixel density
    scale_search=(0.7, 1.4, 21),  # Scale search range
    flip_x=False,            # Horizontal flip
    flip_y=False,            # Vertical flip
    fill_mode='edge',        # Fill mode: 'edge', 'reflect', 'constant'
    fill_cval=np.nan,        # Fill value for 'constant' mode
    return_valid_mask=True   # Return validity mask
)
```

**Returns:**
- `aligned`: Transformed DAT array on Gerber canvas
- `params`: Dictionary with transformation parameters
- `valid_mask`: Boolean mask of valid (non-extrapolated) pixels

### `apply_alignment()`

Apply previously computed alignment to other arrays.

```python
aligned_layer2 = apply_alignment(
    mov_arr=dat_layer2,
    params=params,           # From align_dat_to_gerber()
    out_shape=gerber_arr.shape,
    order=1,                 # Interpolation: 0=nearest, 1=bilinear, 3=cubic
    cval=0.0                 # Fill value
)
```

**Use Case:** Apply the same alignment to multiple DAT layers.

### Helper Functions

#### `sanitize(a)`
Ensures array is finite, float32, normalized to [0, 1].

#### `binarize_robust(a, sigma=1.0)`
Gaussian blur + Otsu thresholding with safety checks.

#### `dominant_angle_safe(binary)`
Estimates orientation from image moments (safe on sparse data).

#### `auto_scale(ref, mov, scales, preblur=1.0)`
Finds scale maximizing NCC between reference and moving images.

#### `phase_translate(ref, mov)`
Returns (dy, dx) shift using phase cross-correlation.

---

## Usage Examples

### Example 1: Basic Alignment

```python
import numpy as np
from lib.array_operations.alignment import align_dat_to_gerber

# Load your data
gerber = np.load('gerber_layer.npy')
dat = np.load('dat_measurement.npy')

# Align with automatic scale search
aligned, params, mask = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    return_valid_mask=True
)

print(f"Rotation: {params['rotation_deg']:.2f}°")
print(f"Scale: {params['scale']:.4f}")
print(f"Shift: {params['shift']}")
```

### Example 2: Known Pixel Densities

```python
# When you know the pixel densities, skip scale search
aligned, params = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    px_per_mm_gerber=100.0,  # 100 pixels per mm
    px_per_mm_dat=50.0,      # 50 pixels per mm
    return_valid_mask=False
)
```

### Example 3: Multiple Layers

```python
# Align first layer
aligned_layer1, params, mask = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat_layer1
)

# Apply same alignment to other layers
from lib.array_operations.alignment import apply_alignment

aligned_layer2 = apply_alignment(
    mov_arr=dat_layer2,
    params=params,
    out_shape=gerber.shape
)

aligned_layer3 = apply_alignment(
    mov_arr=dat_layer3,
    params=params,
    out_shape=gerber.shape
)
```

### Example 4: Handling Extrapolation

```python
# Use NaN for extrapolated regions
aligned, params, mask = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    fill_mode='constant',
    fill_cval=np.nan,
    return_valid_mask=True
)

# Mask out invalid regions
aligned_masked = np.where(mask, aligned, np.nan)

# Or use edge extension (no holes)
aligned_edge, params = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    fill_mode='edge',
    return_valid_mask=False
)
```

---

## Advanced Features

### Fill Modes

Control how regions outside the original DAT data are handled:

- **`'edge'`** (default): Extend nearest edge values
  - No zeros, no NaNs
  - Best for continuous data
  
- **`'reflect'`**: Mirror reflection about borders
  - Smooth transitions
  - Good for periodic patterns
  
- **`'constant'`**: Fill with specified value
  - Use `fill_cval=np.nan` to mark extrapolated regions
  - Use `fill_cval=0.0` for zero padding

### Validity Masking

When `return_valid_mask=True`, you get a boolean mask indicating which pixels come from actual DAT data vs. extrapolation:

```python
aligned, params, valid_mask = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    return_valid_mask=True
)

# Count valid pixels
valid_percentage = 100 * valid_mask.sum() / valid_mask.size
print(f"Valid coverage: {valid_percentage:.1f}%")

# Apply mask
aligned_clean = np.where(valid_mask, aligned, np.nan)
```

### Flip Operations

Handle mirrored or inverted data:

```python
# Horizontal flip
aligned, params = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    flip_x=True
)

# Vertical flip
aligned, params = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    flip_y=True
)

# Both
aligned, params = align_dat_to_gerber(
    gerber_arr=gerber,
    dat_arr=dat,
    flip_x=True,
    flip_y=True
)
```

---

## Troubleshooting

### Poor Alignment Quality

**Symptoms:** Misaligned features, incorrect rotation/scale

**Solutions:**
1. Check input data quality:
   ```python
   print(f"Gerber range: [{gerber.min()}, {gerber.max()}]")
   print(f"DAT range: [{dat.min()}, {dat.max()}]")
   print(f"Gerber non-zero: {(gerber > 0).sum()}")
   print(f"DAT non-zero: {(dat > 0).sum()}")
   ```

2. Adjust scale search range:
   ```python
   # Wider search
   aligned, params = align_dat_to_gerber(
       gerber_arr=gerber,
       dat_arr=dat,
       scale_search=(0.5, 2.0, 31)  # Wider range, more steps
   )
   ```

3. Try different flip combinations:
   ```python
   for flip_x in [False, True]:
       for flip_y in [False, True]:
           aligned, params = align_dat_to_gerber(
               gerber_arr=gerber,
               dat_arr=dat,
               flip_x=flip_x,
               flip_y=flip_y
           )
           # Evaluate quality...
   ```

### Holes or Artifacts

**Symptoms:** Black regions, NaN values, edge artifacts

**Solutions:**
1. Use edge fill mode:
   ```python
   aligned, params = align_dat_to_gerber(
       gerber_arr=gerber,
       dat_arr=dat,
       fill_mode='edge'  # No holes
   )
   ```

2. Check validity mask:
   ```python
   aligned, params, mask = align_dat_to_gerber(
       gerber_arr=gerber,
       dat_arr=dat,
       return_valid_mask=True
   )
   print(f"Invalid pixels: {(~mask).sum()}")
   ```

### Slow Performance

**Symptoms:** Long computation time

**Solutions:**
1. Reduce scale search steps:
   ```python
   aligned, params = align_dat_to_gerber(
       gerber_arr=gerber,
       dat_arr=dat,
       scale_search=(0.7, 1.4, 11)  # Fewer steps
   )
   ```

2. Provide known pixel densities:
   ```python
   aligned, params = align_dat_to_gerber(
       gerber_arr=gerber,
       dat_arr=dat,
       px_per_mm_gerber=100.0,
       px_per_mm_dat=50.0  # Skip scale search
   )
   ```

3. Downsample for initial alignment:
   ```python
   from skimage.transform import rescale
   
   # Coarse alignment
   gerber_small = rescale(gerber, 0.5, preserve_range=True)
   dat_small = rescale(dat, 0.5, preserve_range=True)
   _, params = align_dat_to_gerber(gerber_small, dat_small)
   
   # Adjust scale for full resolution
   params['scale'] *= 2.0
   aligned = apply_alignment(dat, params, gerber.shape)
   ```

---

## Technical Details

### Phase Cross-Correlation

The translation estimation uses FFT-based phase correlation:

1. Compute FFT of both images
2. Calculate cross-power spectrum
3. Inverse FFT to get correlation surface
4. Find peak location (with sub-pixel refinement)

**Advantages:**
- Sub-pixel accuracy (10x upsampling)
- Robust to noise and illumination changes
- Computationally efficient (O(n log n))

### Moment-Based Orientation

Rotation estimation uses second-order image moments:

1. Compute covariance matrix of pixel coordinates
2. Find eigenvector of largest eigenvalue
3. Calculate angle from eigenvector direction

**Advantages:**
- Fast computation
- Works on binary or grayscale images
- Robust to sparse features

### Normalized Cross-Correlation

Scale search maximizes NCC:

```
NCC = Σ(A - mean(A)) * (B - mean(B)) / (||A - mean(A)|| * ||B - mean(B)||)
```

**Advantages:**
- Invariant to linear intensity changes
- Range: [-1, 1] (1 = perfect match)
- Efficient for discrete scale search

---

## References

- **Phase Correlation**: Kuglin & Hines (1975), "The Phase Correlation Image Alignment Method"
- **Image Moments**: Hu (1962), "Visual Pattern Recognition by Moment Invariants"
- **scikit-image**: van der Walt et al. (2014), "scikit-image: image processing in Python"

---

## See Also

- [`array_utils.py`](array_utils.py) - Array manipulation utilities
- [`comparison.py`](comparison.py) - Alignment quality metrics
- [`image_processing.py`](image_processing.py) - Pre/post-processing operations

---

*Documentation generated for CopperBalancingFinal project*
*Last updated: 2026-02-17*