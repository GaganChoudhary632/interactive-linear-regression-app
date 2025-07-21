import streamlit as st
import numpy as np
import pandas as pd
from src.regression_engine import LinearRegressionEngine
from src.data_generator import DataGenerator
from src.visualizations import RegressionVisualizer
from src.utils import AppUtils, DataProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="Interactive Linear Regression",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'regression_engine' not in st.session_state:
    st.session_state.regression_engine = LinearRegressionEngine()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = RegressionVisualizer()
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

def main():
    """Main application function"""
    
    # App header
    st.title("📈 Interactive Linear Regression Model")
    st.markdown("Explore how slope and intercept parameters affect linear regression performance")
    
    # Sidebar for controls
    AppUtils.create_sidebar_info()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("🎛️ Controls")
        
        # Dataset selection
        st.subheader("Dataset Selection")
        dataset_options = DataGenerator.get_dataset_options()
        selected_dataset = st.selectbox(
            "Choose a dataset:",
            options=list(dataset_options.keys()),
            index=0
        )
        
        # Dataset parameters
        n_points = st.slider("Number of data points:", 20, 200, 50, 10)
        noise_level = st.slider("Noise level:", 0.0, 2.0, 0.5, 0.1)
        
        # Generate data button
        if st.button("🔄 Generate New Data"):
            dataset_type = dataset_options[selected_dataset]
            x_data, y_data = DataGenerator.generate_dataset(
                dataset_type, n_points, noise_level
            )
            st.session_state.current_data = (x_data, y_data)
        
        # Initialize data if not exists
        if st.session_state.current_data is None:
            dataset_type = dataset_options[selected_dataset]
            x_data, y_data = DataGenerator.generate_dataset(
                dataset_type, n_points, noise_level
            )
            st.session_state.current_data = (x_data, y_data)
        
        x_data, y_data = st.session_state.current_data
        
        st.divider()
        
        # Regression parameters
        st.subheader("Regression Parameters")
        
        # Slope and intercept controls
        slope = st.slider(
            "Slope (m):",
            min_value=-10.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Rate of change of Y with respect to X"
        )
        
        intercept = st.slider(
            "Intercept (b):",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.5,
            help="Y-value when X equals zero"
        )
        
        # Update regression engine
        st.session_state.regression_engine.set_parameters(slope, intercept)
        
        # Display current equation
        equation = AppUtils.calculate_line_equation(slope, intercept)
        st.markdown(f"**Current Equation:** `{equation}`")
        
        st.divider()
        
        # Best fit line option
        st.subheader("Analysis Options")
        show_best_fit = st.checkbox("Show Best Fit Line", value=False)
        
        if show_best_fit:
            # Calculate best fit line
            best_fit_engine = LinearRegressionEngine()
            best_slope, best_intercept = best_fit_engine.fit_best_line(x_data, y_data)
            best_equation = AppUtils.calculate_line_equation(best_slope, best_intercept)
            st.markdown(f"**Best Fit:** `{best_equation}`")
        
        # Reset button
        if st.button("🔄 Reset Parameters"):
            st.rerun()
    
    with col1:
        st.header("📊 Visualizations")
        
        # Calculate predictions and metrics
        x_line = np.linspace(x_data.min() - 2, x_data.max() + 2, 100)
        y_pred_line = st.session_state.regression_engine.predict(x_line)
        y_pred_points = st.session_state.regression_engine.predict(x_data)
        
        # Calculate metrics
        metrics = st.session_state.regression_engine.calculate_metrics(x_data, y_data)
        residuals = st.session_state.regression_engine.calculate_residuals(x_data, y_data)
        
        # Main regression plot
        best_fit_data = None
        if show_best_fit:
            best_fit_y = best_fit_engine.predict(x_line)
            best_fit_data = (x_line, best_fit_y)
        
        main_fig = st.session_state.visualizer.create_main_plot(
            x_data, y_data, x_line, y_pred_line, slope, intercept,
            show_best_fit, best_fit_data
        )
        st.plotly_chart(main_fig, use_container_width=True)
        
        # Metrics and residuals in tabs
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📈 Residuals", "🎯 Prediction"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Performance Metrics")
                st.markdown(AppUtils.format_metrics(metrics))
                
                # R-squared interpretation
                st.markdown("**Model Quality:**")
                st.markdown(AppUtils.interpret_r_squared(metrics['r2']))
            
            with col_b:
                # Metrics visualization
                metrics_fig = st.session_state.visualizer.create_metrics_plot(metrics)
                st.plotly_chart(metrics_fig, use_container_width=True)
        
        with tab2:
            # Residuals plot
            residuals_fig = st.session_state.visualizer.create_residuals_plot(x_data, residuals)
            st.plotly_chart(residuals_fig, use_container_width=True)
            
            st.markdown("**Residuals Analysis:**")
            st.markdown(f"- Mean residual: {np.mean(residuals):.4f}")
            st.markdown(f"- Std deviation: {np.std(residuals):.4f}")
            st.markdown(f"- Min residual: {np.min(residuals):.4f}")
            st.markdown(f"- Max residual: {np.max(residuals):.4f}")
        
        with tab3:
            st.subheader("Make Predictions")
            
            # Prediction input
            pred_x = st.number_input(
                "Enter X value for prediction:",
                value=0.0,
                step=0.1
            )
            
            if st.button("🎯 Predict"):
                pred_y = st.session_state.regression_engine.predict(np.array([pred_x]))[0]
                st.success(f"Predicted Y value: **{pred_y:.2f}**")
                
                # Show prediction plot
                pred_fig = st.session_state.visualizer.create_prediction_plot(
                    (x_data.min(), x_data.max()), slope, intercept, pred_x, pred_y
                )
                st.plotly_chart(pred_fig, use_container_width=True)
    
    # Data export section
    st.divider()
    st.header("💾 Data Export")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📥 Download Current Data"):
            # Create download data
            download_df = AppUtils.create_download_data(
                x_data, y_data, y_pred_points, residuals
            )
            
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="regression_analysis.csv",
                mime="text/csv"
            )
    
    with col_export2:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with X and Y columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                is_valid, message = DataProcessor.validate_uploaded_data(df)
                
                if is_valid:
                    x_upload, y_upload = DataProcessor.process_uploaded_data(df)
                    st.session_state.current_data = (x_upload, y_upload)
                    st.success("Data uploaded successfully!")
                    st.rerun()
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()
