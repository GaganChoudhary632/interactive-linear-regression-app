import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, Tuple
import io
import base64

class AppUtils:
    """
    Utility functions for the linear regression app
    """
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """
        Format metrics for display
        """
        formatted = []
        formatted.append(f"**Mean Squared Error (MSE):** {metrics['mse']:.4f}")
        formatted.append(f"**Mean Absolute Error (MAE):** {metrics['mae']:.4f}")
        formatted.append(f"**Root Mean Squared Error (RMSE):** {metrics['rmse']:.4f}")
        formatted.append(f"**R-squared (R²):** {metrics['r2']:.4f}")
        
        return "\n".join(formatted)
    
    @staticmethod
    def interpret_r_squared(r2: float) -> str:
        """
        Provide interpretation of R-squared value
        """
        if r2 >= 0.9:
            return "🟢 Excellent fit - The model explains most of the variance"
        elif r2 >= 0.7:
            return "🟡 Good fit - The model explains a substantial portion of the variance"
        elif r2 >= 0.5:
            return "🟠 Moderate fit - The model has some predictive power"
        elif r2 >= 0.3:
            return "🔴 Poor fit - The model has limited predictive power"
        else:
            return "❌ Very poor fit - The model explains very little variance"
    
    @staticmethod
    def create_download_data(x_data: np.ndarray, y_data: np.ndarray, 
                           y_pred: np.ndarray, residuals: np.ndarray) -> pd.DataFrame:
        """
        Create DataFrame for download
        """
        df = pd.DataFrame({
            'X': x_data,
            'Y_Actual': y_data,
            'Y_Predicted': y_pred,
            'Residuals': residuals
        })
        return df
    
    @staticmethod
    def get_csv_download_link(df: pd.DataFrame, filename: str = "regression_data.csv") -> str:
        """
        Generate download link for CSV data
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
        return href
    
    @staticmethod
    def validate_input_range(value: float, min_val: float, max_val: float, 
                           param_name: str) -> bool:
        """
        Validate if input value is within acceptable range
        """
        if min_val <= value <= max_val:
            return True
        else:
            st.error(f"{param_name} must be between {min_val} and {max_val}")
            return False
    
    @staticmethod
    def calculate_line_equation(slope: float, intercept: float) -> str:
        """
        Format the linear equation as a string
        """
        if intercept >= 0:
            return f"y = {slope:.2f}x + {intercept:.2f}"
        else:
            return f"y = {slope:.2f}x - {abs(intercept):.2f}"
    
    @staticmethod
    def get_app_info() -> Dict[str, Any]:
        """
        Return app information and instructions
        """
        return {
            "title": "Interactive Linear Regression Model",
            "description": """
            This application allows you to explore linear regression interactively. 
            You can adjust the slope and intercept parameters to see how they affect 
            the regression line and model performance.
            """,
            "instructions": [
                "1. Choose a dataset from the dropdown menu",
                "2. Adjust the slope and intercept using the sliders",
                "3. Observe how the regression line changes in real-time",
                "4. Check the metrics to evaluate model performance",
                "5. Use the residuals plot to analyze prediction errors",
                "6. Try the prediction feature to forecast new values"
            ],
            "features": [
                "Real-time parameter adjustment",
                "Multiple dataset options",
                "Interactive visualizations",
                "Statistical metrics calculation",
                "Residuals analysis",
                "Data export functionality",
                "Best fit line comparison"
            ]
        }
    
    @staticmethod
    def create_sidebar_info():
        """
        Create informational sidebar content
        """
        st.sidebar.markdown("### 📊 About Linear Regression")
        st.sidebar.markdown("""
        **Linear Regression** models the relationship between variables using the equation:
        
        **y = mx + b**
        
        Where:
        - **m** = slope (rate of change)
        - **b** = intercept (y-value when x=0)
        - **x** = independent variable
        - **y** = dependent variable
        """)
        
        st.sidebar.markdown("### 📈 Key Metrics")
        st.sidebar.markdown("""
        - **MSE**: Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)  
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **R²**: Coefficient of determination (higher is better, max = 1.0)
        """)
        
        st.sidebar.markdown("### 🎯 Tips")
        st.sidebar.markdown("""
        - Try different datasets to see how linear regression performs
        - Compare your line with the best fit line
        - Watch the residuals plot for patterns
        - R² close to 1.0 indicates a good fit
        """)

class DataProcessor:
    """
    Handle data processing and validation
    """
    
    @staticmethod
    def validate_uploaded_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate uploaded CSV data
        """
        if df.empty:
            return False, "The uploaded file is empty"
        
        if df.shape[1] < 2:
            return False, "The file must contain at least 2 columns (X and Y)"
        
        # Check for numeric data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return False, "The file must contain at least 2 numeric columns"
        
        return True, "Data validation successful"
    
    @staticmethod
    def process_uploaded_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process uploaded data and return X, Y arrays
        """
        # Take first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
        x_data = df[numeric_cols[0]].values
        y_data = df[numeric_cols[1]].values
        
        # Remove any NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        return x_data, y_data
