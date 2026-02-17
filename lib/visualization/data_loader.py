"""
Data Loader Module

Handles loading and preprocessing of parameter sweep CSV data.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class DataLoader:
    """Load and preprocess parameter sweep data."""
    
    # Define parameter columns (inputs)
    PARAMETER_COLUMNS = [
        'DPI', 'Percent from Center', 'Material', 'Location', 'Side',
        'Edge Fill', 'Percent Max Fill', 'Blur Type', 'Radius', 'Sigma',
        'Gradient Method', 'Dx', 'Dy', 'Window'
    ]
    
    # Define metric columns (outputs)
    METRIC_COLUMNS = [
        'rmse_3d', 'mae_3d', 'p95_3d', 'max_3d',
        'rmse_z', 'mae_z', 'p95_z', 'max_z',
        'pearson_r', 'slope', 'intercept', 'r2',
        'angle_mean_deg', 'angle_median_deg', 'angle_p95_deg',
        'mag_ratio_mean', 'mag_ratio_median', 'mag_ratio_p05', 'mag_ratio_p95'
    ]
    
    # Transformation matrix columns (usually not plotted directly)
    TRANSFORM_COLUMNS = [
        'scale', 'R00', 'R01', 'R02', 'R10', 'R11', 'R12',
        'R20', 'R21', 'R22', 't_x', 't_y', 't_z'
    ]
    
    def __init__(self, file_path: str):
        """
        Initialize data loader.
        
        Args:
            file_path: Path to CSV file
        """
        self.file_path = Path(file_path)
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load CSV data with proper data types."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        
        # Load CSV
        self.df = pd.read_csv(self.file_path)
        
        # Convert numeric columns
        numeric_cols = self.METRIC_COLUMNS + self.TRANSFORM_COLUMNS + ['DPI', 'Dx', 'Dy', 'Window']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Handle empty strings in Radius and Sigma
        if 'Radius' in self.df.columns:
            self.df['Radius'] = pd.to_numeric(self.df['Radius'], errors='coerce')
        if 'Sigma' in self.df.columns:
            self.df['Sigma'] = pd.to_numeric(self.df['Sigma'], errors='coerce')
        
        print(f"Loaded {len(self.df)} rows from {self.file_path}")
    
    def get_data(self) -> pd.DataFrame:
        """Return the full dataframe."""
        return self.df.copy()
    
    def get_parameter_columns(self) -> List[str]:
        """Return list of parameter columns that exist in the data."""
        return [col for col in self.PARAMETER_COLUMNS if col in self.df.columns]
    
    def get_metric_columns(self) -> List[str]:
        """Return list of metric columns that exist in the data."""
        return [col for col in self.METRIC_COLUMNS if col in self.df.columns]
    
    def get_numeric_parameters(self) -> List[str]:
        """Return list of numeric parameter columns."""
        numeric_params = ['DPI', 'Percent from Center', 'Percent Max Fill', 
                         'Radius', 'Sigma', 'Dx', 'Dy', 'Window']
        return [col for col in numeric_params if col in self.df.columns]
    
    def get_categorical_parameters(self) -> List[str]:
        """Return list of categorical parameter columns."""
        categorical_params = ['Material', 'Location', 'Side', 'Edge Fill', 
                             'Blur Type', 'Gradient Method']
        return [col for col in categorical_params if col in self.df.columns]
    
    def get_unique_values(self, column: str) -> List[Any]:
        """
        Get unique values for a column.
        
        Args:
            column: Column name
            
        Returns:
            List of unique values (sorted if numeric)
        """
        if column not in self.df.columns:
            return []
        
        unique_vals = self.df[column].dropna().unique()
        
        # Sort if numeric
        if pd.api.types.is_numeric_dtype(self.df[column]):
            unique_vals = sorted(unique_vals)
        else:
            unique_vals = sorted([str(v) for v in unique_vals])
        
        return list(unique_vals)
    
    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter dataframe based on provided filters.
        
        Args:
            filters: Dictionary of {column: value} or {column: [values]}
            
        Returns:
            Filtered dataframe
        """
        df_filtered = self.df.copy()
        
        for column, value in filters.items():
            if column not in df_filtered.columns:
                continue
            
            if value is None or (isinstance(value, list) and len(value) == 0):
                continue
            
            # Handle list of values (multi-select)
            if isinstance(value, list):
                df_filtered = df_filtered[df_filtered[column].isin(value)]
            # Handle single value
            else:
                df_filtered = df_filtered[df_filtered[column] == value]
        
        return df_filtered
    
    def get_summary_statistics(self, column: str, filters: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate summary statistics for a column.
        
        Args:
            column: Column name
            filters: Optional filters to apply
            
        Returns:
            Dictionary of statistics
        """
        df = self.filter_data(filters) if filters else self.df
        
        if column not in df.columns:
            return {}
        
        data = df[column].dropna()
        
        if len(data) == 0:
            return {}
        
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'q25': data.quantile(0.25),
            'median': data.median(),
            'q75': data.quantile(0.75),
            'max': data.max()
        }
        
        return stats
    
    def find_best_configs(self, metric: str, top_n: int = 10, 
                         minimize: bool = True, filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Find best configurations based on a metric.
        
        Args:
            metric: Metric column to optimize
            top_n: Number of top configurations to return
            minimize: If True, find minimum values; if False, find maximum
            filters: Optional filters to apply
            
        Returns:
            DataFrame with top configurations
        """
        df = self.filter_data(filters) if filters else self.df
        
        if metric not in df.columns:
            return pd.DataFrame()
        
        # Sort by metric
        df_sorted = df.sort_values(by=metric, ascending=minimize)
        
        # Get top N
        top_configs = df_sorted.head(top_n)
        
        # Select relevant columns
        display_cols = ['Name'] + self.get_parameter_columns() + [metric]
        display_cols = [col for col in display_cols if col in top_configs.columns]
        
        return top_configs[display_cols]
    
    def get_correlation_matrix(self, columns: Optional[List[str]] = None, 
                               filters: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for specified columns.
        
        Args:
            columns: List of columns to include (default: all numeric)
            filters: Optional filters to apply
            
        Returns:
            Correlation matrix as DataFrame
        """
        df = self.filter_data(filters) if filters else self.df
        
        if columns is None:
            # Use all numeric columns
            columns = self.get_numeric_parameters() + self.get_metric_columns()
        
        # Filter to existing columns
        columns = [col for col in columns if col in df.columns]
        
        # Calculate correlation
        corr_matrix = df[columns].corr()
        
        return corr_matrix


def load_data(file_path: str) -> DataLoader:
    """
    Convenience function to load data.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataLoader instance
    """
    return DataLoader(file_path)

# Made with Bob
