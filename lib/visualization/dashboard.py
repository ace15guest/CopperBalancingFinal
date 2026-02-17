"""
Interactive Dashboard for Parameter Sweep Analysis

Main Dash application for visualizing and analyzing parameter sweep results.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

from .data_loader import DataLoader
from .plot_generators import (
    create_scatter_plot, create_line_plot, create_correlation_heatmap,
    create_box_plot, create_parallel_coordinates, create_best_config_table,
    create_multi_metric_comparison
)
from .utils import format_column_name, create_filter_options, get_metric_description


# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)


def create_layout(data_loader: DataLoader) -> html.Div:
    """
    Create the dashboard layout.
    
    Args:
        data_loader: DataLoader instance with loaded data
        
    Returns:
        Dash HTML layout
    """
    # Get column lists
    param_cols = data_loader.get_parameter_columns()
    metric_cols = data_loader.get_metric_columns()
    numeric_params = data_loader.get_numeric_parameters()
    categorical_params = data_loader.get_categorical_parameters()
    
    # Create filter options
    filter_controls = []
    for col in categorical_params[:6]:  # Limit to first 6 for space
        options = create_filter_options(data_loader.df, col)
        filter_controls.append(
            dbc.Col([
                html.Label(format_column_name(col), className="fw-bold"),
                dcc.Dropdown(
                    id=f'filter-{col}',
                    options=options,
                    multi=True,
                    placeholder=f"All {format_column_name(col)}",
                    className="mb-2"
                )
            ], md=2)
        )
    
    layout = html.Div([
        # Header
        dbc.Container([
            html.H1("📊 Parameter Sweep Analysis Dashboard", 
                   className="text-center my-4 text-primary"),
            html.Hr(),
        ], fluid=True),
        
        # Filters Section
        dbc.Container([
            html.H4("🔍 Filters", className="mb-3"),
            dbc.Row(filter_controls),
            html.Hr(),
        ], fluid=True, className="mb-4"),
        
        # Main Visualization Controls
        dbc.Container([
            html.H4("📈 Main Visualization", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("X-Axis (Parameter)", className="fw-bold"),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': format_column_name(col), 'value': col} 
                                for col in param_cols],
                        value=param_cols[0] if param_cols else None,
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Y-Axis (Metric)", className="fw-bold"),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': format_column_name(col), 'value': col} 
                                for col in metric_cols],
                        value=metric_cols[0] if metric_cols else None,
                        clearable=False
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Color By", className="fw-bold"),
                    dcc.Dropdown(
                        id='color-by-dropdown',
                        options=[{'label': format_column_name(col), 'value': col} 
                                for col in categorical_params],
                        value=categorical_params[0] if categorical_params else None,
                        placeholder="None"
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Plot Type", className="fw-bold"),
                    dcc.RadioItems(
                        id='plot-type-radio',
                        options=[
                            {'label': ' Scatter', 'value': 'scatter'},
                            {'label': ' Line', 'value': 'line'},
                            {'label': ' Box', 'value': 'box'}
                        ],
                        value='scatter',
                        inline=True,
                        className="mt-2"
                    )
                ], md=3),
            ], className="mb-3"),
            
            # Metric description
            html.Div(id='metric-description', className="text-muted mb-3"),
            
            # Main plot
            dcc.Loading(
                dcc.Graph(id='main-plot', style={'height': '600px'}),
                type="default"
            ),
            
            # Data count
            html.Div(id='data-count', className="text-center text-muted mt-2"),
            
            html.Hr(),
        ], fluid=True, className="mb-4"),
        
        # Tabs for additional visualizations
        dbc.Container([
            dbc.Tabs([
                # Correlation Tab
                dbc.Tab(label="🔗 Correlation Analysis", children=[
                    html.Div([
                        html.H5("Correlation Heatmap", className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Columns", className="fw-bold"),
                                dcc.Dropdown(
                                    id='correlation-columns',
                                    options=[{'label': format_column_name(col), 'value': col} 
                                            for col in numeric_params + metric_cols],
                                    value=(numeric_params + metric_cols)[:10],
                                    multi=True
                                )
                            ], md=12)
                        ], className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(id='correlation-heatmap', style={'height': '700px'}),
                            type="default"
                        )
                    ])
                ]),
                
                # Multi-Metric Comparison Tab
                dbc.Tab(label="📊 Multi-Metric Comparison", children=[
                    html.Div([
                        html.H5("Compare Multiple Metrics", className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("X-Axis Parameter", className="fw-bold"),
                                dcc.Dropdown(
                                    id='multi-x-axis',
                                    options=[{'label': format_column_name(col), 'value': col} 
                                            for col in numeric_params],
                                    value=numeric_params[0] if numeric_params else None
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Metrics to Compare", className="fw-bold"),
                                dcc.Dropdown(
                                    id='multi-metrics',
                                    options=[{'label': format_column_name(col), 'value': col} 
                                            for col in metric_cols],
                                    value=metric_cols[:3] if len(metric_cols) >= 3 else metric_cols,
                                    multi=True
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Normalize", className="fw-bold"),
                                dcc.Checklist(
                                    id='normalize-checkbox',
                                    options=[{'label': ' Normalize to 0-1', 'value': 'normalize'}],
                                    value=['normalize'],
                                    className="mt-2"
                                )
                            ], md=2)
                        ], className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(id='multi-metric-plot', style={'height': '600px'}),
                            type="default"
                        )
                    ])
                ]),
                
                # Best Configurations Tab
                dbc.Tab(label="🏆 Best Configurations", children=[
                    html.Div([
                        html.H5("Find Optimal Parameter Combinations", className="mt-3 mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Optimize Metric", className="fw-bold"),
                                dcc.Dropdown(
                                    id='optimize-metric',
                                    options=[{'label': format_column_name(col), 'value': col} 
                                            for col in metric_cols],
                                    value=metric_cols[0] if metric_cols else None
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Objective", className="fw-bold"),
                                dcc.RadioItems(
                                    id='optimize-objective',
                                    options=[
                                        {'label': ' Minimize', 'value': 'minimize'},
                                        {'label': ' Maximize', 'value': 'maximize'}
                                    ],
                                    value='minimize',
                                    inline=True,
                                    className="mt-2"
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Top N", className="fw-bold"),
                                dcc.Slider(
                                    id='top-n-slider',
                                    min=5,
                                    max=50,
                                    step=5,
                                    value=10,
                                    marks={i: str(i) for i in range(5, 51, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], md=4)
                        ], className="mb-3"),
                        dcc.Loading(
                            dcc.Graph(id='best-config-table', style={'height': '500px'}),
                            type="default"
                        ),
                        html.Div([
                            dbc.Button("📥 Export Best Configs", id="export-button", 
                                      color="primary", className="mt-3"),
                            html.Div(id='export-status', className="mt-2")
                        ])
                    ])
                ]),
                
                # Statistics Tab
                dbc.Tab(label="📉 Statistics", children=[
                    html.Div([
                        html.H5("Summary Statistics", className="mt-3 mb-3"),
                        html.Div(id='statistics-content')
                    ])
                ])
            ])
        ], fluid=True, className="mb-4"),
        
        # Footer
        dbc.Container([
            html.Hr(),
            html.P("Parameter Sweep Analysis Dashboard | Built with Plotly Dash", 
                  className="text-center text-muted")
        ], fluid=True)
    ])
    
    return layout


def register_callbacks(data_loader: DataLoader):
    """
    Register all dashboard callbacks.
    
    Args:
        data_loader: DataLoader instance
    """
    
    @app.callback(
        Output('main-plot', 'figure'),
        Output('data-count', 'children'),
        [Input('x-axis-dropdown', 'value'),
         Input('y-axis-dropdown', 'value'),
         Input('color-by-dropdown', 'value'),
         Input('plot-type-radio', 'value')] +
        [Input(f'filter-{col}', 'value') for col in data_loader.get_categorical_parameters()[:6]]
    )
    def update_main_plot(x_col, y_col, color_by, plot_type, *filter_values):
        """Update main visualization based on selections."""
        # Build filters dictionary
        filters = {}
        categorical_params = data_loader.get_categorical_parameters()[:6]
        for col, value in zip(categorical_params, filter_values):
            if value:
                filters[col] = value
        
        # Get filtered data
        df_filtered = data_loader.filter_data(filters)
        
        # Create appropriate plot
        if plot_type == 'scatter':
            fig = create_scatter_plot(df_filtered, x_col, y_col, color_by)
        elif plot_type == 'line':
            fig = create_line_plot(df_filtered, x_col, [y_col], color_by)
        elif plot_type == 'box':
            if color_by:
                fig = create_box_plot(df_filtered, color_by, y_col)
            else:
                fig = create_scatter_plot(df_filtered, x_col, y_col)
        else:
            fig = create_scatter_plot(df_filtered, x_col, y_col, color_by)
        
        # Data count message
        count_msg = f"Showing {len(df_filtered)} of {len(data_loader.df)} data points"
        
        return fig, count_msg
    
    @app.callback(
        Output('metric-description', 'children'),
        Input('y-axis-dropdown', 'value')
    )
    def update_metric_description(metric):
        """Update metric description."""
        if metric:
            desc = get_metric_description(metric)
            return html.P([html.Strong("Metric: "), desc])
        return ""
    
    @app.callback(
        Output('correlation-heatmap', 'figure'),
        [Input('correlation-columns', 'value')] +
        [Input(f'filter-{col}', 'value') for col in data_loader.get_categorical_parameters()[:6]]
    )
    def update_correlation_heatmap(columns, *filter_values):
        """Update correlation heatmap."""
        # Build filters
        filters = {}
        categorical_params = data_loader.get_categorical_parameters()[:6]
        for col, value in zip(categorical_params, filter_values):
            if value:
                filters[col] = value
        
        df_filtered = data_loader.filter_data(filters)
        
        return create_correlation_heatmap(df_filtered, columns)
    
    @app.callback(
        Output('multi-metric-plot', 'figure'),
        [Input('multi-x-axis', 'value'),
         Input('multi-metrics', 'value'),
         Input('normalize-checkbox', 'value')] +
        [Input(f'filter-{col}', 'value') for col in data_loader.get_categorical_parameters()[:6]]
    )
    def update_multi_metric_plot(x_col, metrics, normalize, *filter_values):
        """Update multi-metric comparison plot."""
        # Build filters
        filters = {}
        categorical_params = data_loader.get_categorical_parameters()[:6]
        for col, value in zip(categorical_params, filter_values):
            if value:
                filters[col] = value
        
        df_filtered = data_loader.filter_data(filters)
        
        normalize_flag = 'normalize' in (normalize or [])
        
        return create_multi_metric_comparison(
            df_filtered, x_col, metrics, normalize=normalize_flag
        )
    
    @app.callback(
        Output('best-config-table', 'figure'),
        [Input('optimize-metric', 'value'),
         Input('optimize-objective', 'value'),
         Input('top-n-slider', 'value')] +
        [Input(f'filter-{col}', 'value') for col in data_loader.get_categorical_parameters()[:6]]
    )
    def update_best_configs(metric, objective, top_n, *filter_values):
        """Update best configurations table."""
        # Build filters
        filters = {}
        categorical_params = data_loader.get_categorical_parameters()[:6]
        for col, value in zip(categorical_params, filter_values):
            if value:
                filters[col] = value
        
        minimize = (objective == 'minimize')
        
        best_configs = data_loader.find_best_configs(
            metric, top_n=top_n, minimize=minimize, filters=filters
        )
        
        display_cols = data_loader.get_parameter_columns() + [metric]
        
        return create_best_config_table(best_configs, display_cols)
    
    @app.callback(
        Output('statistics-content', 'children'),
        [Input('y-axis-dropdown', 'value')] +
        [Input(f'filter-{col}', 'value') for col in data_loader.get_categorical_parameters()[:6]]
    )
    def update_statistics(metric, *filter_values):
        """Update statistics display."""
        # Build filters
        filters = {}
        categorical_params = data_loader.get_categorical_parameters()[:6]
        for col, value in zip(categorical_params, filter_values):
            if value:
                filters[col] = value
        
        stats = data_loader.get_summary_statistics(metric, filters)
        
        if not stats:
            return html.P("No data available")
        
        # Create statistics cards
        cards = []
        stat_names = {
            'count': 'Count',
            'mean': 'Mean',
            'std': 'Std Dev',
            'min': 'Minimum',
            'q25': '25th Percentile',
            'median': 'Median',
            'q75': '75th Percentile',
            'max': 'Maximum'
        }
        
        for key, label in stat_names.items():
            if key in stats:
                cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(label, className="text-muted"),
                                html.H4(f"{stats[key]:.4f}" if key != 'count' else str(stats[key]))
                            ])
                        ])
                    ], md=3, className="mb-3")
                )
        
        return dbc.Row(cards)


def run_dashboard(data_path: str = "Assets/DataOutput/data_out.csv",
                 host: str = "127.0.0.1",
                 port: int = 8050,
                 debug: bool = True):
    """
    Run the dashboard application.
    
    Args:
        data_path: Path to CSV data file
        host: Host address
        port: Port number
        debug: Enable debug mode
    """
    # Load data
    print(f"Loading data from {data_path}...")
    data_loader = DataLoader(data_path)
    
    # Create layout
    app.layout = create_layout(data_loader)
    
    # Register callbacks
    register_callbacks(data_loader)
    
    # Run server
    print(f"\n🚀 Starting dashboard at http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_dashboard()

# Made with Bob
