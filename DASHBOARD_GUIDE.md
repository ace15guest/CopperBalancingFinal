# 📊 Parameter Sweep Dashboard - Quick Start Guide

## Overview

This interactive dashboard allows you to visualize and analyze the results from your parameter sweep experiments. It plots how different processing parameters (DPI, blur radius, edge fill methods, etc.) affect output metrics (RMSE, correlation, gradient analysis, etc.).

## Installation

### Step 1: Install Dependencies

```bash
pip install plotly dash dash-bootstrap-components pandas numpy
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Verify Data File

Make sure your parameter sweep results are at:
```
Assets/DataOutput/data_out.csv
```

## Running the Dashboard

### Method 1: Using the Launcher Script (Recommended)

```bash
python run_dashboard.py
```

### Method 2: Direct Python Import

```python
from lib.visualization import run_dashboard

run_dashboard()
```

### Method 3: Custom Configuration

```python
from lib.visualization import run_dashboard

run_dashboard(
    data_path="path/to/your/data.csv",
    host="127.0.0.1",
    port=8050,
    debug=True
)
```

## Accessing the Dashboard

Once started, open your web browser and navigate to:
```
http://127.0.0.1:8050
```

Press `Ctrl+C` in the terminal to stop the server.

## Dashboard Features

### 🔍 1. Filters (Top Section)

Filter your data by:
- **Material**: e.g., 890K
- **Location**: Q1, Q2, Q3, Q4, Global
- **Side**: Top or Bottom
- **Edge Fill**: idw, nearest, percent_max
- **Blur Type**: box_blur, gauss
- **Gradient Method**: finite, plane

**Tip**: Select multiple values in each filter to compare different conditions.

### 📈 2. Main Visualization

**Controls:**
- **X-Axis**: Choose a parameter (e.g., DPI, Radius, Sigma)
- **Y-Axis**: Choose a metric (e.g., rmse_3d, pearson_r)
- **Color By**: Color points by a categorical variable
- **Plot Type**: 
  - **Scatter**: See individual data points
  - **Line**: Track trends across parameter values
  - **Box**: Compare distributions across categories

**Interactive Features:**
- Hover over points to see details
- Click and drag to zoom
- Double-click to reset zoom
- Click legend items to show/hide traces

### 🔗 3. Correlation Analysis Tab

**Purpose**: Identify which parameters most strongly affect which metrics.

**How to Use:**
1. Click the "Correlation Analysis" tab
2. Select parameters and metrics to analyze
3. Look for strong correlations (red = positive, blue = negative)
4. Values close to ±1 indicate strong relationships

**Example Insights:**
- Does higher DPI always reduce error?
- Which blur radius gives best correlation?
- Are certain edge fill methods consistently better?

### 📊 4. Multi-Metric Comparison Tab

**Purpose**: Compare how multiple metrics respond to parameter changes.

**How to Use:**
1. Click "Multi-Metric Comparison" tab
2. Select X-axis parameter (e.g., DPI)
3. Select multiple metrics to compare
4. Enable "Normalize" to see relative changes

**Use Cases:**
- Find parameter values that balance multiple objectives
- Identify trade-offs (e.g., lower error but worse correlation)
- See which metrics are most sensitive to parameter changes

### 🏆 5. Best Configurations Tab

**Purpose**: Find the optimal parameter combinations for your goals.

**How to Use:**
1. Click "Best Configurations" tab
2. Select metric to optimize (e.g., rmse_3d)
3. Choose objective:
   - **Minimize**: For error metrics (RMSE, MAE)
   - **Maximize**: For correlation metrics (Pearson R, R²)
4. Adjust "Top N" slider to see more/fewer results
5. Click "Export Best Configs" to save results

**Example Workflow:**
```
Goal: Minimize 3D error
1. Select "rmse_3d" as metric
2. Choose "Minimize"
3. Set Top N = 10
4. Review parameter combinations
5. Export for documentation
```

### 📉 6. Statistics Tab

**Purpose**: View detailed statistics for selected metrics.

**Shows:**
- Count of data points
- Mean and standard deviation
- Minimum and maximum values
- Quartiles (25th, 50th, 75th percentile)

## Common Analysis Workflows

### Workflow 1: Find Best DPI Setting

1. **Filter** by your specific material and location
2. **Main Plot**: Set X-axis = "DPI", Y-axis = "rmse_3d"
3. **Observe** which DPI gives lowest error
4. **Verify** in Best Configurations tab
5. **Check** if other metrics also improve at that DPI

### Workflow 2: Compare Edge Fill Methods

1. **Main Plot**: Set Color By = "Edge Fill"
2. **Try different** Y-axis metrics
3. **Look for** consistent patterns across metrics
4. **Use Box Plot** to see distribution differences
5. **Check Statistics** tab for quantitative comparison

### Workflow 3: Optimize Blur Parameters

1. **Filter** to specific edge fill method
2. **Main Plot**: X-axis = "Radius" (or "Sigma"), Y-axis = error metric
3. **Multi-Metric**: Compare multiple error metrics
4. **Find** radius/sigma that minimizes errors
5. **Verify** correlation metrics don't degrade

### Workflow 4: Understand Parameter Interactions

1. **Correlation Tab**: Select all parameters and key metrics
2. **Identify** strong correlations
3. **Main Plot**: Explore identified relationships
4. **Filter** by different conditions to see if relationships hold
5. **Document** findings for parameter selection

## Understanding the Metrics

### Error Metrics (Lower is Better)
- **rmse_3d**: Root Mean Square Error in 3D space
- **mae_3d**: Mean Absolute Error in 3D space
- **p95_3d**: 95th percentile error (captures outliers)
- **max_3d**: Maximum error (worst case)

### Correlation Metrics (Higher is Better)
- **pearson_r**: Correlation coefficient (-1 to 1, closer to 1 is better)
- **r2**: R-squared (0 to 1, closer to 1 is better)
- **slope**: Should be close to 1 for good agreement

### Gradient Metrics
- **angle_mean_deg**: Mean angle difference (lower is better)
- **mag_ratio_mean**: Magnitude ratio (closer to 1 is better)

## Tips and Tricks

### Performance Tips
- **Large datasets**: Use filters to reduce data before plotting
- **Correlation heatmap**: Limit to 10-15 columns for readability
- **Export data**: Save filtered results for offline analysis

### Analysis Tips
- **Start broad**: Look at all data first, then filter
- **Multiple views**: Use different plot types for same data
- **Normalize**: Use normalization when comparing metrics with different scales
- **Document**: Export best configs and take screenshots of key findings

### Troubleshooting
- **No data showing**: Check filters aren't too restrictive
- **Plot not updating**: Ensure valid columns are selected
- **Slow performance**: Reduce number of data points or columns
- **Can't see legend**: Resize browser window or zoom out

## Example Analysis Session

```
Goal: Find optimal parameters for Q1 Top side processing

1. Set Filters:
   - Location: Q1
   - Side: Top
   
2. Main Visualization:
   - X-axis: DPI
   - Y-axis: rmse_3d
   - Color By: Edge Fill
   - Result: DPI 300 with IDW fill shows lowest error

3. Verify with Multi-Metric:
   - X-axis: DPI
   - Metrics: rmse_3d, mae_3d, pearson_r
   - Result: DPI 300 good for all metrics

4. Optimize Blur:
   - Filter: DPI = 300, Edge Fill = idw
   - X-axis: Radius
   - Y-axis: rmse_3d
   - Result: Radius 500 optimal

5. Best Configurations:
   - Metric: rmse_3d
   - Objective: Minimize
   - Result: DPI=300, Edge Fill=idw, Radius=500, Gradient=finite
   
6. Export results and document findings
```

## Keyboard Shortcuts

- **Ctrl+C**: Stop dashboard server
- **F5**: Refresh browser (if dashboard seems stuck)
- **Ctrl+Scroll**: Zoom in/out on plots
- **Shift+Click**: Select multiple items in dropdowns

## Getting Help

### Common Issues

**"Module not found" error**
```bash
pip install -r requirements.txt
```

**"Data file not found" error**
- Check file path in run_dashboard.py
- Verify data_out.csv exists in Assets/DataOutput/

**Dashboard won't start**
- Check port 8050 is not in use
- Try different port: `run_dashboard(port=8051)`

**Plots are empty**
- Check data file has valid data
- Verify column names match expected format
- Check filters aren't excluding all data

### Additional Resources

- Full API documentation: `lib/visualization/README.md`
- Project documentation: `README.md`
- Example scripts: `examples/` directory

## Next Steps

After finding optimal parameters:
1. Document your findings
2. Update configuration files with optimal values
3. Run batch processing with selected parameters
4. Validate results with test data
5. Share insights with team

---

**Happy Analyzing! 📊✨**