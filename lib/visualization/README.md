# Parameter Sweep Visualization Dashboard

An interactive web-based dashboard for analyzing and visualizing parameter sweep results from Gerber processing experiments.

## Features

### 🎯 Main Visualizations
- **Scatter Plots**: Visualize relationships between any parameter and metric
- **Line Plots**: Track metric changes across parameter ranges
- **Box Plots**: Compare distributions across categories

### 🔍 Advanced Analysis
- **Correlation Heatmap**: Identify relationships between all parameters and metrics
- **Multi-Metric Comparison**: Compare multiple metrics simultaneously (with normalization)
- **Best Configuration Finder**: Identify optimal parameter combinations
- **Statistical Summaries**: Comprehensive statistics for all metrics

### 🎛️ Interactive Controls
- **Dynamic Filtering**: Filter by Material, Location, Side, Edge Fill, Blur Type, and more
- **Color Coding**: Color points by categorical variables
- **Real-time Updates**: All visualizations update instantly based on selections
- **Export Functionality**: Export best configurations to CSV

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `dash>=2.14.0` - Web framework
- `dash-bootstrap-components>=1.5.0` - UI components
- `plotly>=5.18.0` - Interactive plotting
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations

### 2. Verify Data File

Ensure your parameter sweep results are in CSV format at:
```
Assets/DataOutput/data_out.csv
```

## Usage

### Quick Start

```python
from lib.visualization import run_dashboard

# Run with default settings
run_dashboard()
```

The dashboard will start at `http://127.0.0.1:8050`

### Custom Configuration

```python
from lib.visualization import run_dashboard

# Custom data path and port
run_dashboard(
    data_path="path/to/your/data.csv",
    host="0.0.0.0",  # Allow external access
    port=8080,
    debug=False  # Production mode
)
```

### Command Line

Create a simple script `run_dashboard.py`:

```python
#!/usr/bin/env python
"""Run the parameter sweep dashboard."""

from lib.visualization import run_dashboard

if __name__ == '__main__':
    run_dashboard(
        data_path="Assets/DataOutput/data_out.csv",
        port=8050,
        debug=True
    )
```

Then run:
```bash
python run_dashboard.py
```

## Dashboard Layout

### 1. Filters Section
- Filter data by categorical parameters
- Multiple selections supported
- Filters apply to all visualizations

### 2. Main Visualization
- **X-Axis**: Select any parameter (DPI, Radius, Sigma, etc.)
- **Y-Axis**: Select any metric (RMSE, Pearson R, etc.)
- **Color By**: Color points by categorical variables
- **Plot Type**: Choose between Scatter, Line, or Box plots

### 3. Tabbed Analysis Views

#### 🔗 Correlation Analysis
- Interactive correlation heatmap
- Select which parameters/metrics to include
- Identify strong relationships

#### 📊 Multi-Metric Comparison
- Compare multiple metrics on one plot
- Optional normalization to 0-1 range
- Useful for identifying trade-offs

#### 🏆 Best Configurations
- Find optimal parameter combinations
- Choose metric to optimize (minimize or maximize)
- Adjust number of top configurations shown
- Export results to CSV

#### 📉 Statistics
- Summary statistics for selected metric
- Count, mean, std dev, quartiles, etc.
- Updates based on active filters

## Data Format

The dashboard expects CSV data with the following structure:

### Input Parameters (Columns)
- `DPI`: Resolution (e.g., 200, 250, 300, 500)
- `Percent from Center`: Crop percentage (e.g., 0.25, 0.4)
- `Material`: Material type (e.g., 890K)
- `Location`: Board location (Q1, Q2, Q3, Q4, Global)
- `Side`: Top or Bottom
- `Edge Fill`: Fill method (idw, nearest, percent_max)
- `Percent Max Fill`: Fill percentage (when using percent_max)
- `Blur Type`: box_blur or gauss
- `Radius`: Box blur radius
- `Sigma`: Gaussian blur sigma
- `Gradient Method`: finite or plane
- `Dx`, `Dy`, `Window`: Gradient calculation parameters

### Output Metrics (Columns)
- **3D Error Metrics**: `rmse_3d`, `mae_3d`, `p95_3d`, `max_3d`
- **Z-Axis Metrics**: `rmse_z`, `mae_z`, `p95_z`, `max_z`
- **Correlation**: `pearson_r`, `slope`, `intercept`, `r2`
- **Gradient Analysis**: `angle_mean_deg`, `angle_median_deg`, `angle_p95_deg`
- **Magnitude Ratios**: `mag_ratio_mean`, `mag_ratio_median`, `mag_ratio_p05`, `mag_ratio_p95`

## Tips for Analysis

### Finding Optimal Parameters
1. Go to **Best Configurations** tab
2. Select metric to optimize (e.g., `rmse_3d` to minimize error)
3. Choose "Minimize" or "Maximize"
4. Review top configurations
5. Export results for further analysis

### Identifying Relationships
1. Go to **Correlation Analysis** tab
2. Select parameters and metrics of interest
3. Look for strong correlations (close to ±1)
4. Use insights to guide parameter selection

### Comparing Trade-offs
1. Go to **Multi-Metric Comparison** tab
2. Select a parameter for X-axis (e.g., DPI)
3. Choose multiple metrics to compare
4. Enable normalization to see relative changes
5. Identify parameter values that balance multiple objectives

### Filtering for Specific Conditions
1. Use filter dropdowns at the top
2. Select specific materials, locations, or methods
3. All visualizations update automatically
4. Compare filtered vs. unfiltered results

## Troubleshooting

### Dashboard won't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify data file exists at specified path
- Check port is not already in use

### No data showing
- Verify CSV file has correct format
- Check that filters aren't excluding all data
- Look for error messages in console

### Slow performance
- Large datasets (>100k rows) may be slow
- Consider filtering data before loading
- Reduce number of points in correlation heatmap

### Plot not updating
- Check that valid columns are selected
- Ensure filtered data is not empty
- Refresh browser if needed

## Module Structure

```
lib/visualization/
├── __init__.py              # Package initialization
├── data_loader.py           # CSV loading and preprocessing
├── plot_generators.py       # Plot creation functions
├── dashboard.py             # Main Dash application
├── utils.py                 # Helper functions
└── README.md               # This file
```

## API Reference

### DataLoader Class

```python
from lib.visualization.data_loader import DataLoader

loader = DataLoader("path/to/data.csv")

# Get data
df = loader.get_data()

# Get column lists
params = loader.get_parameter_columns()
metrics = loader.get_metric_columns()

# Filter data
filtered = loader.filter_data({'DPI': 300, 'Side': 'Top'})

# Find best configs
best = loader.find_best_configs('rmse_3d', top_n=10, minimize=True)

# Get statistics
stats = loader.get_summary_statistics('pearson_r')
```

### Plot Generators

```python
from lib.visualization.plot_generators import (
    create_scatter_plot,
    create_correlation_heatmap,
    create_multi_metric_comparison
)

# Create scatter plot
fig = create_scatter_plot(df, 'DPI', 'rmse_3d', color_by='Edge Fill')

# Create correlation heatmap
fig = create_correlation_heatmap(df, columns=['DPI', 'Radius', 'rmse_3d'])

# Create multi-metric comparison
fig = create_multi_metric_comparison(
    df, 'DPI', ['rmse_3d', 'mae_3d', 'pearson_r'], normalize=True
)
```

## Contributing

To add new visualizations:

1. Add plot function to `plot_generators.py`
2. Add UI controls to `dashboard.py` layout
3. Create callback in `dashboard.py` to update plot
4. Update this README with usage instructions

## License

Part of the Copper Balancing Image Processing project.

## Support

For issues or questions, please refer to the main project documentation.