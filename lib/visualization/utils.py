"""
Utility Functions for Visualization

Helper functions for formatting, calculations, and data export.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def format_column_name(col: str) -> str:
    """
    Format column name for display.
    
    Args:
        col: Column name
        
    Returns:
        Formatted column name
    """
    # Replace underscores with spaces
    formatted = col.replace('_', ' ')
    
    # Capitalize words
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Special cases
    replacements = {
        'Rmse': 'RMSE',
        'Mae': 'MAE',
        'P95': 'P95',
        'Dpi': 'DPI',
        'Dx': 'dx',
        'Dy': 'dy',
        'R2': 'R²',
        'Deg': '(deg)',
        'Idw': 'IDW',
        '3d': '3D',
        'Mag': 'Magnitude'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted


def format_value(value: Any, precision: int = 4) -> str:
    """
    Format value for display.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if isinstance(value, (int, np.integer)):
        return str(value)
    
    if isinstance(value, (float, np.floating)):
        # Use scientific notation for very small or large numbers
        if abs(value) < 0.001 or abs(value) > 10000:
            return f'{value:.{precision}e}'
        else:
            return f'{value:.{precision}f}'
    
    return str(value)


def calculate_statistics(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a column.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        Dictionary of statistics
    """
    if column not in df.columns:
        return {}
    
    data = df[column].dropna()
    
    if len(data) == 0:
        return {'count': 0}
    
    stats = {
        'count': len(data),
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'q25': data.quantile(0.25),
        'median': data.median(),
        'q75': data.quantile(0.75),
        'max': data.max(),
        'range': data.max() - data.min(),
        'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else np.nan  # Coefficient of variation
    }
    
    return stats


def find_optimal_configs(df: pd.DataFrame, metric: str, 
                        top_n: int = 10, minimize: bool = True) -> pd.DataFrame:
    """
    Find optimal configurations based on a metric.
    
    Args:
        df: DataFrame with results
        metric: Metric to optimize
        top_n: Number of configurations to return
        minimize: If True, minimize metric; if False, maximize
        
    Returns:
        DataFrame with top configurations
    """
    if metric not in df.columns:
        return pd.DataFrame()
    
    # Remove rows with NaN in the metric
    df_clean = df.dropna(subset=[metric])
    
    # Sort
    df_sorted = df_clean.sort_values(by=metric, ascending=minimize)
    
    # Return top N
    return df_sorted.head(top_n)


def export_to_csv(df: pd.DataFrame, filename: str) -> bool:
    """
    Export DataFrame to CSV.
    
    Args:
        df: DataFrame to export
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        df.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False


def create_filter_options(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    """
    Create options list for dropdown from unique column values.
    
    Args:
        df: DataFrame
        column: Column name
        
    Returns:
        List of {'label': str, 'value': Any} dictionaries
    """
    if column not in df.columns:
        return []
    
    unique_vals = df[column].dropna().unique()
    
    # Sort if possible
    try:
        unique_vals = sorted(unique_vals)
    except TypeError:
        # Can't sort mixed types
        unique_vals = sorted([str(v) for v in unique_vals])
    
    options = [{'label': str(val), 'value': val} for val in unique_vals]
    
    return options


def get_color_scale(n_colors: int = 10) -> List[str]:
    """
    Get a color scale for plotting.
    
    Args:
        n_colors: Number of colors needed
        
    Returns:
        List of color hex codes
    """
    # Use Plotly's default color sequence
    plotly_colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    
    # Repeat if needed
    colors = (plotly_colors * ((n_colors // len(plotly_colors)) + 1))[:n_colors]
    
    return colors


def normalize_data(data: pd.Series) -> pd.Series:
    """
    Normalize data to 0-1 range.
    
    Args:
        data: Series to normalize
        
    Returns:
        Normalized series
    """
    min_val = data.min()
    max_val = data.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(data), index=data.index)
    
    return (data - min_val) / (max_val - min_val)


def get_metric_description(metric: str) -> str:
    """
    Get description for a metric.
    
    Args:
        metric: Metric name
        
    Returns:
        Description string
    """
    descriptions = {
        'rmse_3d': 'Root Mean Square Error in 3D space',
        'mae_3d': 'Mean Absolute Error in 3D space',
        'p95_3d': '95th percentile error in 3D space',
        'max_3d': 'Maximum error in 3D space',
        'rmse_z': 'Root Mean Square Error in Z direction',
        'mae_z': 'Mean Absolute Error in Z direction',
        'p95_z': '95th percentile error in Z direction',
        'max_z': 'Maximum error in Z direction',
        'pearson_r': 'Pearson correlation coefficient',
        'slope': 'Linear regression slope',
        'intercept': 'Linear regression intercept',
        'r2': 'R-squared (coefficient of determination)',
        'angle_mean_deg': 'Mean gradient angle difference (degrees)',
        'angle_median_deg': 'Median gradient angle difference (degrees)',
        'angle_p95_deg': '95th percentile gradient angle difference (degrees)',
        'mag_ratio_mean': 'Mean gradient magnitude ratio',
        'mag_ratio_median': 'Median gradient magnitude ratio',
        'mag_ratio_p05': '5th percentile gradient magnitude ratio',
        'mag_ratio_p95': '95th percentile gradient magnitude ratio'
    }
    
    return descriptions.get(metric, 'No description available')


def create_summary_table(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Create a summary statistics table for multiple columns.
    
    Args:
        df: DataFrame
        columns: List of columns to summarize
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        stats = calculate_statistics(df, col)
        stats['Column'] = format_column_name(col)
        summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Reorder columns
    col_order = ['Column', 'count', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max']
    col_order = [c for c in col_order if c in summary_df.columns]
    
    return summary_df[col_order]

# Made with Bob
