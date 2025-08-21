import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, List

class LinearRegressionEngine:
    """
    Core engine for linear regression calculations and predictions
    """
    
    def __init__(self):
        self.slope = 1.0
        self.intercept = 0.0
        
    def set_parameters(self, slope: float, intercept: float):
        """Set the slope and intercept parameters"""
        self.slope = slope
        self.intercept = intercept
    
    def predict(self, x_values: np.ndarray) -> np.ndarray:
        """
        Calculate y values using linear equation: y = mx + b
        """
        return self.slope * x_values + self.intercept
    
    def calculate_metrics(self, x_true: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Calculate regression metrics (MSE, MAE, R²)
        """
        y_pred = self.predict(x_true)
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def calculate_residuals(self, x_values: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculate residuals (difference between actual and predicted values)
        """
        y_pred = self.predict(x_values)
        return y_true - y_pred
    
    def fit_best_line(self, x_values: np.ndarray, y_values: np.ndarray):
        """
        Calculate the best fit line using least squares method
        """
        # Calculate slope and intercept using least squares
        n = len(x_values)
        sum_x = np.sum(x_values)
        sum_y = np.sum(y_values)
        sum_xy = np.sum(x_values * y_values)
        sum_x2 = np.sum(x_values ** 2)
        
        # Calculate slope (m) and intercept (b)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        self.slope = slope
        self.intercept = intercept
        
        return slope, intercept
