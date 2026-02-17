"""
Plot Generators Module

Functions to create individual plot components using Plotly.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .utils import format_column_name, get_color_scale


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                       color_by: Optional[str] = None,
                       title: Optional[str] = None,
                       height: int = 600) -> go.Figure:
    """
    Create scatter plot of parameter vs metric.
    
    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_by: Optional column to color points by
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if x_col not in df.columns or y_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Create title if not provided
    if title is None:
        title = f"{format_column_name(y_col)} vs {format_column_name(x_col)}"
    
    # Create scatter plot
    if color_by and color_by in df.columns:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_by,
            title=title,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col),
                color_by: format_column_name(color_by)
            },
            hover_data=['Name'] if 'Name' in df.columns else None
        )
    else:
        fig = px.scatter(
            df, x=x_col, y=y_col,
            title=title,
            labels={
                x_col: format_column_name(x_col),
                y_col: format_column_name(y_col)
            },
            hover_data=['Name'] if 'Name' in df.columns else None
        )
    
    fig.update_layout(height=height, hovermode='closest')
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    
    return fig


def create_line_plot(df: pd.DataFrame, x_col: str, y_cols: List[str],
                    group_by: Optional[str] = None,
                    title: Optional[str] = None,
                    height: int = 600) -> go.Figure:
    """
    Create line plot with multiple metrics.
    
    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_cols: List of columns for y-axis
        group_by: Optional column to group lines by
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if x_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid x-axis column",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    if title is None:
        title = f"Metrics vs {format_column_name(x_col)}"
    
    # Sort by x column
    df_sorted = df.sort_values(by=x_col)
    
    if group_by and group_by in df.columns:
        # Create separate lines for each group
        groups = df_sorted[group_by].unique()
        colors = get_color_scale(len(groups) * len(y_cols))
        color_idx = 0
        
        for group in groups:
            df_group = df_sorted[df_sorted[group_by] == group]
            for y_col in y_cols:
                if y_col in df_group.columns:
                    fig.add_trace(go.Scatter(
                        x=df_group[x_col],
                        y=df_group[y_col],
                        mode='lines+markers',
                        name=f"{format_column_name(y_col)} ({group})",
                        line=dict(color=colors[color_idx]),
                        marker=dict(size=6)
                    ))
                    color_idx += 1
    else:
        # Simple multi-line plot
        colors = get_color_scale(len(y_cols))
        for idx, y_col in enumerate(y_cols):
            if y_col in df_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=df_sorted[x_col],
                    y=df_sorted[y_col],
                    mode='lines+markers',
                    name=format_column_name(y_col),
                    line=dict(color=colors[idx]),
                    marker=dict(size=6)
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title=format_column_name(x_col),
        yaxis_title="Value",
        height=height,
        hovermode='x unified'
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None,
                               title: str = "Correlation Matrix",
                               height: int = 700) -> go.Figure:
    """
    Create correlation heatmap.
    
    Args:
        df: DataFrame with data
        columns: List of columns to include (default: all numeric)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if columns is None:
        # Use all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to existing columns
    columns = [col for col in columns if col in df.columns]
    
    if len(columns) < 2:
        return go.Figure().add_annotation(
            text="Need at least 2 numeric columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Calculate correlation
    corr_matrix = df[columns].corr()
    
    # Format labels
    labels = [format_column_name(col) for col in columns]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickangle=0)
    )
    
    return fig


def create_box_plot(df: pd.DataFrame, category_col: str, metric_col: str,
                   title: Optional[str] = None,
                   height: int = 600) -> go.Figure:
    """
    Create box plot for comparing distributions.
    
    Args:
        df: DataFrame with data
        category_col: Column for categories (x-axis)
        metric_col: Column for metric values (y-axis)
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if category_col not in df.columns or metric_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid column selection",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    if title is None:
        title = f"{format_column_name(metric_col)} by {format_column_name(category_col)}"
    
    fig = px.box(
        df, x=category_col, y=metric_col,
        title=title,
        labels={
            category_col: format_column_name(category_col),
            metric_col: format_column_name(metric_col)
        },
        color=category_col
    )
    
    fig.update_layout(height=height, showlegend=False)
    
    return fig


def create_parallel_coordinates(df: pd.DataFrame, columns: List[str],
                                color_col: Optional[str] = None,
                                title: str = "Parallel Coordinates Plot",
                                height: int = 600) -> go.Figure:
    """
    Create parallel coordinates plot for multi-dimensional visualization.
    
    Args:
        df: DataFrame with data
        columns: List of columns to include
        color_col: Column to use for coloring
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Filter to existing columns
    columns = [col for col in columns if col in df.columns]
    
    if len(columns) < 2:
        return go.Figure().add_annotation(
            text="Need at least 2 columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Prepare dimensions
    dimensions = []
    for col in columns:
        dimensions.append(dict(
            label=format_column_name(col),
            values=df[col]
        ))
    
    # Create color scale
    if color_col and color_col in df.columns:
        color_values = df[color_col]
    else:
        color_values = df[columns[0]]
    
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=color_values,
            colorscale='Viridis',
            showscale=True,
            cmin=color_values.min(),
            cmax=color_values.max()
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title=title,
        height=height
    )
    
    return fig


def create_best_config_table(df: pd.DataFrame, display_cols: List[str],
                             title: str = "Best Configurations") -> go.Figure:
    """
    Create table showing best configurations.
    
    Args:
        df: DataFrame with configurations
        display_cols: Columns to display
        title: Table title
        
    Returns:
        Plotly Figure object
    """
    # Filter to existing columns
    display_cols = [col for col in display_cols if col in df.columns]
    
    if len(display_cols) == 0:
        return go.Figure().add_annotation(
            text="No valid columns to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Format headers
    headers = [format_column_name(col) for col in display_cols]
    
    # Get values
    values = [df[col].tolist() for col in display_cols]
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=values,
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title=title,
        height=min(400, 50 + len(df) * 30)
    )
    
    return fig


def create_multi_metric_comparison(df: pd.DataFrame, x_col: str, 
                                   metric_cols: List[str],
                                   normalize: bool = True,
                                   title: Optional[str] = None,
                                   height: int = 600) -> go.Figure:
    """
    Create normalized multi-metric comparison plot.
    
    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        metric_cols: List of metric columns to compare
        normalize: Whether to normalize metrics to 0-1 range
        title: Plot title
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    if x_col not in df.columns:
        return go.Figure().add_annotation(
            text="Invalid x-axis column",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Filter to existing columns
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    if len(metric_cols) == 0:
        return go.Figure().add_annotation(
            text="No valid metric columns",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    if title is None:
        title = f"Multi-Metric Comparison vs {format_column_name(x_col)}"
        if normalize:
            title += " (Normalized)"
    
    fig = go.Figure()
    
    # Sort by x column
    df_sorted = df.sort_values(by=x_col)
    
    colors = get_color_scale(len(metric_cols))
    
    for idx, metric in enumerate(metric_cols):
        y_values = df_sorted[metric].values
        
        if normalize:
            # Normalize to 0-1
            min_val = np.nanmin(y_values)
            max_val = np.nanmax(y_values)
            if max_val > min_val:
                y_values = (y_values - min_val) / (max_val - min_val)
        
        fig.add_trace(go.Scatter(
            x=df_sorted[x_col],
            y=y_values,
            mode='lines+markers',
            name=format_column_name(metric),
            line=dict(color=colors[idx]),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=format_column_name(x_col),
        yaxis_title="Normalized Value" if normalize else "Value",
        height=height,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

# Made with Bob
