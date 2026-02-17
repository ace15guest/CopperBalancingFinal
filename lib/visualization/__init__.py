"""
Visualization Module for Parameter Sweep Analysis

This module provides interactive visualization tools for analyzing parameter sweep
results from Gerber processing experiments.

Main Components:
- data_loader: Load and preprocess CSV data
- plot_generators: Create individual plot components
- dashboard: Main Dash application
- utils: Helper functions
- layer_visualizer: Visualize individual layers and cropping results

Usage:
    # Dashboard
    from lib.visualization.dashboard import run_dashboard
    
    run_dashboard(
        data_path="Assets/DataOutput/data_out.csv",
        host="127.0.0.1",
        port=8050
    )
    
    # Layer visualization
    from lib.visualization.layer_visualizer import visualize_single_layer
    import matplotlib.pyplot as plt
    
    fig = visualize_single_layer(array, layer_name="Signal Layer 3")
    plt.show()
"""

from .dashboard import run_dashboard
from .layer_visualizer import (
    visualize_single_layer,
    compare_crop_result,
    visualize_layer_from_npz,
    calculate_layer_statistics
)

__all__ = [
    'run_dashboard',
    'visualize_single_layer',
    'compare_crop_result',
    'visualize_layer_from_npz',
    'calculate_layer_statistics'
]
__version__ = '1.0.0'

# Made with Bob
