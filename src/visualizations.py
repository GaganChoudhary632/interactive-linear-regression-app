import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import streamlit as st
from typing import Tuple, Dict

class RegressionVisualizer:
    """
    Handle all visualization components for the linear regression app
    """
    
    def __init__(self):
        self.colors = {
            'data_points': '#3498db',
            'regression_line': '#e74c3c',
            'best_fit_line': '#2ecc71',
            'residuals': '#f39c12',
            'background': '#ecf0f1'
        }
    
    def create_main_plot(self, x_data: np.ndarray, y_data: np.ndarray,
                        x_line: np.ndarray, y_pred: np.ndarray,
                        slope: float, intercept: float,
                        show_best_fit: bool = False,
                        best_fit_line: Tuple[np.ndarray, np.ndarray] = None) -> go.Figure:
        """
        Create the main regression plot with data points and regression line
        """
        fig = go.Figure()
        
        # Add scatter plot for data points
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Data Points',
            marker=dict(
                color=self.colors['data_points'],
                size=8,
                opacity=0.7
            ),
            hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_pred,
            mode='lines',
            name=f'Your Line (y = {slope:.2f}x + {intercept:.2f})',
            line=dict(
                color=self.colors['regression_line'],
                width=3
            ),
            hovertemplate='<b>Predicted Y:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Add best fit line if requested
        if show_best_fit and best_fit_line is not None:
            fig.add_trace(go.Scatter(
                x=best_fit_line[0],
                y=best_fit_line[1],
                mode='lines',
                name='Best Fit Line',
                line=dict(
                    color=self.colors['best_fit_line'],
                    width=2,
                    dash='dash'
                ),
                hovertemplate='<b>Best Fit Y:</b> %{y:.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Linear Regression Visualization',
            xaxis_title='X Values',
            yaxis_title='Y Values',
            hovermode='closest',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_residuals_plot(self, x_data: np.ndarray, residuals: np.ndarray) -> go.Figure:
        """
        Create residuals plot to show prediction errors
        """
        fig = go.Figure()
        
        # Add residuals scatter plot
        fig.add_trace(go.Scatter(
            x=x_data,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.colors['residuals'],
                size=6,
                opacity=0.7
            ),
            hovertemplate='<b>X:</b> %{x:.2f}<br><b>Residual:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Add horizontal line at y=0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Perfect Prediction Line"
        )
        
        # Update layout
        fig.update_layout(
            title='Residuals Plot (Actual - Predicted)',
            xaxis_title='X Values',
            yaxis_title='Residuals',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_metrics_plot(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create a bar chart showing regression metrics
        """
        metric_names = ['MSE', 'MAE', 'RMSE', 'R²']
        metric_values = [metrics['mse'], metrics['mae'], metrics['rmse'], abs(metrics['r2'])]
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#e74c3c', '#f39c12', '#9b59b6', '#2ecc71'],
                text=[f'{val:.3f}' for val in metric_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Regression Metrics',
            yaxis_title='Value',
            template='plotly_white',
            height=300,
            showlegend=False
        )
        
        return fig
    
    def create_prediction_plot(self, x_range: Tuple[float, float], 
                             slope: float, intercept: float,
                             prediction_x: float = None,
                             prediction_y: float = None) -> go.Figure:
        """
        Create a plot showing the regression line over a wider range for predictions
        """
        x_extended = np.linspace(x_range[0] - 5, x_range[1] + 5, 100)
        y_extended = slope * x_extended + intercept
        
        fig = go.Figure()
        
        # Add extended regression line
        fig.add_trace(go.Scatter(
            x=x_extended,
            y=y_extended,
            mode='lines',
            name=f'Regression Line (y = {slope:.2f}x + {intercept:.2f})',
            line=dict(color=self.colors['regression_line'], width=2)
        ))
        
        # Add prediction point if provided
        if prediction_x is not None and prediction_y is not None:
            fig.add_trace(go.Scatter(
                x=[prediction_x],
                y=[prediction_y],
                mode='markers',
                name=f'Prediction ({prediction_x:.1f}, {prediction_y:.2f})',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='star'
                )
            ))
        
        fig.update_layout(
            title='Extended Regression Line for Predictions',
            xaxis_title='X Values',
            yaxis_title='Predicted Y Values',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_comparison_plot(self, x_data: np.ndarray, y_data: np.ndarray,
                             lines_data: list) -> go.Figure:
        """
        Create a plot comparing multiple regression lines
        """
        fig = go.Figure()
        
        # Add data points
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Data Points',
            marker=dict(color=self.colors['data_points'], size=6, opacity=0.7)
        ))
        
        # Add multiple regression lines
        colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, (x_line, y_line, slope, intercept, label) in enumerate(lines_data):
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name=f'{label} (m={slope:.2f}, b={intercept:.2f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Multiple Regression Lines Comparison',
            xaxis_title='X Values',
            yaxis_title='Y Values',
            template='plotly_white',
            height=500
        )
        
        return fig
