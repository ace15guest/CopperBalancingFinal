"""
Example: Visualizing Individual Layers After Cropping

This example demonstrates how to use the layer visualization functions
to inspect individual layers and see the results of operations like
center_crop_by_area.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.array_operations.border_operations import center_crop_by_area
from lib.visualization.layer_visualizer import (
    visualize_single_layer,
    compare_crop_result,
    visualize_layer_from_npz
)


def example_1_basic_visualization():
    """Example 1: Basic single layer visualization"""
    print("\n=== Example 1: Basic Layer Visualization ===")
    
    # Create a test array with some pattern
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    test_array = np.sin(np.sqrt(X**2 + Y**2))
    
    # Visualize the layer
    fig = visualize_single_layer(
        test_array,
        layer_name="Test Signal Layer",
        show_stats=True,
        save_path="output/test_layer_visualization.png"
    )
    plt.show()
    print("✓ Basic visualization complete")


def example_2_crop_comparison():
    """Example 2: Compare original and cropped arrays"""
    print("\n=== Example 2: Crop Comparison ===")
    
    # Create a test array
    original = np.random.rand(100, 100) * 100
    
    # Apply center crop to 25% area (50x50 pixels)
    cropped = center_crop_by_area(original, pct_area=0.25)
    
    print(f"Original shape: {original.shape}")
    print(f"Cropped shape: {cropped.shape}")
    print(f"Area retained: {(cropped.size / original.size) * 100:.2f}%")
    
    # Compare them side-by-side
    fig = compare_crop_result(
        original,
        cropped,
        layer_name="Signal Layer 3",
        save_path="output/crop_comparison.png"
    )
    plt.show()
    print("✓ Crop comparison complete")


def example_3_multiple_crop_percentages():
    """Example 3: Visualize multiple crop percentages"""
    print("\n=== Example 3: Multiple Crop Percentages ===")
    
    # Create a test array with gradient
    original = np.linspace(0, 100, 10000).reshape(100, 100)
    
    # Test different crop percentages
    crop_percentages = [0.75, 0.50, 0.25, 0.10]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, pct in enumerate(crop_percentages):
        cropped = center_crop_by_area(original, pct_area=pct)
        
        im = axes[idx].imshow(cropped, cmap='viridis', aspect='auto')
        axes[idx].set_title(
            f'Crop: {pct*100:.0f}% Area\n'
            f'Shape: {cropped.shape[0]} × {cropped.shape[1]} pixels',
            fontsize=11, fontweight='bold'
        )
        axes[idx].set_xlabel('Column Index')
        axes[idx].set_ylabel('Row Index')
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        print(f"  {pct*100:.0f}% crop: {cropped.shape}")
    
    fig.suptitle('Center Crop by Area - Multiple Percentages', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/multiple_crop_percentages.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Multiple crop percentages visualization complete")


def example_4_visualize_from_npz():
    """Example 4: Visualize a layer from an NPZ file"""
    print("\n=== Example 4: Visualize from NPZ File ===")
    
    # Example NPZ path (adjust to your actual file)
    npz_path = "Assets/ProcessedPNGs/board1_dpi_600/board1_dpi_600.npz"
    
    # Check if file exists
    if not Path(npz_path).exists():
        print(f"⚠ NPZ file not found at: {npz_path}")
        print("  This example requires an existing NPZ file.")
        print("  Skipping this example.")
        return
    
    # List available layers
    with np.load(npz_path) as data:
        print(f"Available layers in {Path(npz_path).name}:")
        for key in data.files:
            print(f"  - {key}: shape {data[key].shape}")
    
    # Visualize the first layer
    with np.load(npz_path) as data:
        first_key = data.files[0]
    
    fig = visualize_layer_from_npz(
        npz_path,
        first_key,
        show_stats=True,
        save_path=f"output/npz_layer_{first_key}.png"
    )
    plt.show()
    print(f"✓ Visualized layer '{first_key}' from NPZ file")


def example_5_inspect_cropped_layer():
    """Example 5: Inspect a specific cropped layer (like in parameter_sweep)"""
    print("\n=== Example 5: Inspect Cropped Layer (Parameter Sweep Style) ===")
    
    # Simulate what happens in parameter_sweep.py
    # Create a mock layer
    layer = np.random.rand(200, 200) * 50 + 25
    
    print("Simulating parameter_sweep processing:")
    print(f"1. Original layer shape: {layer.shape}")
    
    # Apply center crop (as done in parameter_sweep.py line 178-183)
    percent_from_center = 0.50  # 50% area
    cropped_layer = center_crop_by_area(layer, pct_area=percent_from_center)
    
    print(f"2. After center_crop_by_area({percent_from_center}): {cropped_layer.shape}")
    print(f"3. Area retained: {(cropped_layer.size / layer.size) * 100:.2f}%")
    
    # Visualize just the cropped result
    fig = visualize_single_layer(
        cropped_layer,
        layer_name=f"Signal Layer 3 (Cropped to {percent_from_center*100:.0f}% area)",
        show_stats=True,
        save_path="output/parameter_sweep_style_crop.png"
    )
    plt.show()
    
    print("✓ Parameter sweep style inspection complete")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Layer Visualization Examples")
    print("=" * 60)
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    # Run examples
    example_1_basic_visualization()
    example_2_crop_comparison()
    example_3_multiple_crop_percentages()
    example_4_visualize_from_npz()
    example_5_inspect_cropped_layer()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("Check the 'output' folder for saved visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Made with Bob
