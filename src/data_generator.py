import numpy as np
import pandas as pd
from typing import Tuple

class DataGenerator:
    """
    Generate sample datasets for linear regression visualization
    """
    
    @staticmethod
    def generate_linear_data(n_points: int = 50, slope: float = 2.0, 
                           intercept: float = 1.0, noise_level: float = 0.5,
                           x_range: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate linear data with optional noise
        """
        np.random.seed(42)  # For reproducible results
        x = np.linspace(x_range[0], x_range[1], n_points)
        y_true = slope * x + intercept
        noise = np.random.normal(0, noise_level, n_points)
        y = y_true + noise
        
        return x, y
    
    @staticmethod
    def generate_quadratic_data(n_points: int = 50, a: float = 0.5, 
                              b: float = 2.0, c: float = 1.0,
                              noise_level: float = 0.5,
                              x_range: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate quadratic data for testing linear regression on non-linear data
        """
        np.random.seed(42)
        x = np.linspace(x_range[0], x_range[1], n_points)
        y_true = a * x**2 + b * x + c
        noise = np.random.normal(0, noise_level, n_points)
        y = y_true + noise
        
        return x, y
    
    @staticmethod
    def generate_exponential_data(n_points: int = 50, a: float = 1.0,
                                b: float = 0.1, noise_level: float = 0.5,
                                x_range: Tuple[float, float] = (0, 10)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate exponential data
        """
        np.random.seed(42)
        x = np.linspace(x_range[0], x_range[1], n_points)
        y_true = a * np.exp(b * x)
        noise = np.random.normal(0, noise_level, len(x))
        y = y_true + noise
        
        return x, y
    
    @staticmethod
    def generate_custom_data(n_points: int = 50, 
                           x_range: Tuple[float, float] = (-10, 10)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate custom scattered data points
        """
        np.random.seed(42)
        x = np.random.uniform(x_range[0], x_range[1], n_points)
        y = np.random.uniform(-20, 20, n_points)
        
        return x, y
    
    @staticmethod
    def get_dataset_options() -> dict:
        """
        Return available dataset options
        """
        return {
            "Linear Data": "linear",
            "Quadratic Data": "quadratic", 
            "Exponential Data": "exponential",
            "Random Scatter": "custom"
        }
    
    @staticmethod
    def generate_dataset(dataset_type: str, n_points: int = 50, 
                        noise_level: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset based on type
        """
        if dataset_type == "linear":
            return DataGenerator.generate_linear_data(n_points, noise_level=noise_level)
        elif dataset_type == "quadratic":
            return DataGenerator.generate_quadratic_data(n_points, noise_level=noise_level)
        elif dataset_type == "exponential":
            return DataGenerator.generate_exponential_data(n_points, noise_level=noise_level)
        elif dataset_type == "custom":
            return DataGenerator.generate_custom_data(n_points)
        else:
            return DataGenerator.generate_linear_data(n_points, noise_level=noise_level)
