"""
Layer Visualization Module

Functions to visualize individual layers and their properties,
useful for debugging and understanding processing operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict
from pathlib import Path


def visualize_single_layer(
    array: np.ndarray,
    layer_name: str = "Layer",
    show_stats: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    show_colorbar: bool = True
) -> plt.Figure:
    """
    Visualize a single 2D array layer with statistics and percentage information.
    
    This function is useful for inspecting individual layers after operations
    like center_crop_by_area to see the actual cropped result and understand
    what percentage of the original area is retained.
    
    Parameters:
        array: 2D numpy array to visualize
        layer_name: Name/label for the layer
        show_stats: Whether to display statistics on the plot
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use for visualization
        show_colorbar: Whether to show the colorbar
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> import numpy as np
        >>> from lib.array_operations.border_operations import center_crop_by_area
        >>> 
        >>> # Create test array
        >>> original = np.random.rand(100, 100)
        >>> 
        >>> # Crop to 25% area
        >>> cropped = center_crop_by_area(original, pct_area=0.25)
        >>> 
        >>> # Visualize the cropped layer
        >>> fig = visualize_single_layer(
        ...     cropped,
        ...     layer_name="Signal Layer 3 (Cropped)",
        ...     show_stats=True
        ... )
        >>> plt.show()
    """
    if array.ndim != 2:
        raise ValueError(f"Array must be 2D, got shape {array.shape}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the array
    im = ax.imshow(array, cmap=cmap, aspect='auto', interpolation='nearest')
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Set title
    ax.set_title(f'{layer_name}\nShape: {array.shape[0]} × {array.shape[1]} pixels', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel('Column Index', fontsize=11)
    ax.set_ylabel('Row Index', fontsize=11)
    
    # Calculate and display statistics if requested
    if show_stats:
        stats = calculate_layer_statistics(array)
        
        # Create statistics text box
        stats_text = (
            f"Statistics:\n"
            f"  Min: {stats['min']:.4f}\n"
            f"  Max: {stats['max']:.4f}\n"
            f"  Mean: {stats['mean']:.4f}\n"
            f"  Median: {stats['median']:.4f}\n"
            f"  Std Dev: {stats['std']:.4f}\n"
            f"  Non-zero: {stats['non_zero_pct']:.2f}%\n"
            f"  Valid (finite): {stats['valid_pct']:.2f}%"
        )
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props,
                family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def calculate_layer_statistics(array: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a 2D array layer.
    
    Parameters:
        array: 2D numpy array
        
    Returns:
        Dictionary containing various statistics
    """
    # Handle NaN values
    valid_mask = np.isfinite(array)
    valid_data = array[valid_mask]
    
    if valid_data.size == 0:
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'non_zero_pct': 0.0,
            'valid_pct': 0.0
        }
    
    # Calculate statistics
    stats = {
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'mean': float(np.mean(valid_data)),
        'median': float(np.median(valid_data)),
        'std': float(np.std(valid_data)),
        'non_zero_pct': float(np.count_nonzero(valid_data) / valid_data.size * 100),
        'valid_pct': float(valid_mask.sum() / array.size * 100)
    }
    
    return stats


def compare_crop_result(
    original: np.ndarray,
    cropped: np.ndarray,
    layer_name: str = "Layer",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Compare original and cropped arrays side-by-side with area percentage.
    
    This function is specifically designed to visualize the result of
    center_crop_by_area and similar operations, showing both the original
    and cropped versions with their relative sizes.
    
    Parameters:
        original: Original 2D array before cropping
        cropped: Cropped 2D array
        layer_name: Name/label for the layer
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use for visualization
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> import numpy as np
        >>> from lib.array_operations.border_operations import center_crop_by_area
        >>> 
        >>> # Create and crop array
        >>> original = np.random.rand(100, 100)
        >>> cropped = center_crop_by_area(original, pct_area=0.25)
        >>> 
        >>> # Compare them
        >>> fig = compare_crop_result(original, cropped, "Signal Layer 3")
        >>> plt.show()
    """
    if original.ndim != 2 or cropped.ndim != 2:
        raise ValueError("Both arrays must be 2D")
    
    # Calculate area percentage
    original_area = original.shape[0] * original.shape[1]
    cropped_area = cropped.shape[0] * cropped.shape[1]
    area_pct = (cropped_area / original_area) * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original
    im1 = ax1.imshow(original, cmap=cmap, aspect='auto', interpolation='nearest')
    ax1.set_title(f'Original {layer_name}\n{original.shape[0]} × {original.shape[1]} pixels\n'
                  f'Area: {original_area:,} pixels²',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot cropped
    im2 = ax2.imshow(cropped, cmap=cmap, aspect='auto', interpolation='nearest')
    ax2.set_title(f'Cropped {layer_name}\n{cropped.shape[0]} × {cropped.shape[1]} pixels\n'
                  f'Area: {cropped_area:,} pixels² ({area_pct:.2f}% of original)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add overall title
    fig.suptitle(f'Crop Comparison: {layer_name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def visualize_layer_from_npz(
    npz_path: str,
    layer_key: str,
    show_stats: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Load and visualize a specific layer from an NPZ file.
    
    This is a convenience function for quickly visualizing layers
    stored in NPZ format without manually loading them first.
    
    Parameters:
        npz_path: Path to the NPZ file
        layer_key: Key name of the layer to visualize
        show_stats: Whether to display statistics
        save_path: Optional path to save the figure
        figsize: Figure size (width, height) in inches
        cmap: Colormap to use
        
    Returns:
        matplotlib Figure object
        
    Examples:
        >>> # Visualize a specific layer from NPZ
        >>> fig = visualize_layer_from_npz(
        ...     "Assets/ProcessedPNGs/board1_dpi_600/board1_dpi_600.npz",
        ...     "signal_layer_3_1oz",
        ...     show_stats=True
        ... )
        >>> plt.show()
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load the NPZ file
    with np.load(npz_path) as data:
        if layer_key not in data.files:
            available_keys = ', '.join(data.files)
            raise KeyError(f"Layer '{layer_key}' not found in NPZ. "
                          f"Available keys: {available_keys}")
        
        layer_array = data[layer_key]
    
    # Visualize the layer
    fig = visualize_single_layer(
        layer_array,
        layer_name=f"{layer_key} (from {npz_path.name})",
        show_stats=show_stats,
        save_path=save_path,
        figsize=figsize,
        cmap=cmap
    )
    
    return fig


# Made with Bob