# app.py
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Wind Energy Feasibility Dashboard",
    page_icon="💨",
    layout="wide"
)

# Title and description
st.title("🌬️ Wind Energy Feasibility Dashboard")
st.markdown("""
This tool helps assess the viability of wind energy projects by analyzing historical wind data 
and forecasting future wind patterns. Upload your data or use the sample dataset to get started.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("📊 Data Input")
    
    # Option to use sample data or upload
    data_source = st.radio("Choose data source:", 
                          ["Use Sample Data", "Upload Your CSV"])
    
    if data_source == "Upload Your CSV":
        uploaded_file = st.file_uploader("Upload wind data CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file or use sample data.")
            st.stop()
    else:
        # Generate sample data
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        wind_speeds = np.random.normal(7.5, 2.5, len(dates))  # Mean 7.5 m/s
        seasonal_effect = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        wind_speeds += seasonal_effect + np.random.normal(0, 1, len(dates))
        
        df = pd.DataFrame({
            'ds': dates,
            'y': np.maximum(wind_speeds, 0)  # Wind speed can't be negative
        })
        st.success("✅ Sample data loaded successfully!")

    # Turbine selection
    st.header("⚙️ Turbine Specifications")
    turbine_model = st.selectbox(
        "Select Turbine Model:",
        ["Generic 1.5 MW", "Vestas V90-2.0 MW", "GE 2.5-120", "Custom"]
    )
    
    if turbine_model == "Custom":
        cut_in_speed = st.slider("Cut-in Speed (m/s)", 2.0, 5.0, 3.0)
        rated_speed = st.slider("Rated Speed (m/s)", 10.0, 15.0, 12.0)
        cut_out_speed = st.slider("Cut-out Speed (m/s)", 20.0, 25.0, 25.0)
        rated_power = st.number_input("Rated Power (kW)", 1000, 5000, 1500)
    else:
        # Pre-defined turbine specs [cut-in, rated, cut-out, rated power kW]
        turbine_specs = {
            "Generic 1.5 MW": [3.0, 12.0, 25.0, 1500],
            "Vestas V90-2.0 MW": [4.0, 13.0, 25.0, 2000],
            "GE 2.5-120": [3.5, 12.5, 25.0, 2500]
        }
        cut_in_speed, rated_speed, cut_out_speed, rated_power = turbine_specs[turbine_model]

    # Forecast parameters
    st.header("🔮 Forecast Settings")
    forecast_days = st.slider("Days to forecast:", 30, 365, 90)

# Main content area
tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🔍 Forecast Analysis", "💡 Energy Estimation"])

with tab1:
    st.header("Historical Wind Data")
    
    if 'ds' in df.columns and 'y' in df.columns:
        # Convert to datetime if needed
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(df))
        with col2:
            st.metric("Average Wind Speed", f"{df['y'].mean():.1f} m/s")
        with col3:
            st.metric("Maximum Wind Speed", f"{df['y'].max():.1f} m/s")
        with col4:
            st.metric("Data Period", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
        
        # Plot historical data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Wind Speed'))
        fig.update_layout(
            title='Historical Wind Speed Data',
            xaxis_title='Date',
            yaxis_title='Wind Speed (m/s)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show raw data
        if st.checkbox("Show raw data"):
            st.dataframe(df)
    else:
        st.error("CSV must contain 'ds' (date) and 'y' (wind speed) columns")

with tab2:
    st.header("Wind Speed Forecast")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training forecasting model..."):
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Plot forecast
            fig = model.plot(forecast)
            plt.title('Wind Speed Forecast')
            st.pyplot(fig)
            
            # Plot components
            st.subheader("Forecast Components")
            comp_fig = model.plot_components(forecast)
            st.pyplot(comp_fig)
            
            # Store forecast in session state
            st.session_state.forecast = forecast

with tab3:
    st.header("Energy Production Estimate")
    
    if 'forecast' not in st.session_state:
        st.warning("Please generate a forecast first in the 'Forecast Analysis' tab.")
    else:
        forecast = st.session_state.forecast
        
        # Calculate energy production
        def calculate_power(wind_speed, cut_in, rated, cut_out, rated_power):
            if wind_speed < cut_in or wind_speed > cut_out:
                return 0
            elif wind_speed < rated:
                # Cubic relationship between wind speed and power
                return rated_power * ((wind_speed - cut_in) / (rated - cut_in)) ** 3
            else:
                return rated_power
        
        # Vectorize the power calculation function
        vectorized_power = np.vectorize(calculate_power)
        
        # Calculate power for each forecasted wind speed
        forecast['power_kW'] = vectorized_power(
            forecast['yhat'], 
            cut_in_speed, 
            rated_speed, 
            cut_out_speed, 
            rated_power
        )
        
        # Display energy metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_power = forecast['power_kW'].mean()
            st.metric("Average Power Output", f"{avg_power:.0f} kW")
        with col2:
            capacity_factor = (avg_power / rated_power) * 100
            st.metric("Capacity Factor", f"{capacity_factor:.1f}%")
        with col3:
            annual_energy = avg_power * 24 * 365 / 1000  # MWh
            st.metric("Estimated Annual Energy", f"{annual_energy:,.0f} MWh")
        
        # Plot power output forecast
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add wind speed trace
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Wind Speed", line=dict(color='blue')),
            secondary_y=False,
        )
        
        # Add power output trace
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['power_kW'], name="Power Output", line=dict(color='green')),
            secondary_y=True,
        )
        
        # Add turbine operating range
        fig.add_hline(y=rated_power, line_dash="dash", line_color="red", annotation_text="Rated Power", secondary_y=True)
        
        fig.update_layout(
            title="Wind Speed vs. Power Output Forecast",
            xaxis_title="Date",
        )
        
        fig.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
        fig.update_yaxes(title_text="Power Output (kW)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Power curve visualization
        st.subheader("Turbine Power Curve")
        wind_speeds = np.linspace(0, 30, 100)
        power_output = vectorized_power(wind_speeds, cut_in_speed, rated_speed, cut_out_speed, rated_power)
        
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(x=wind_speeds, y=power_output, mode='lines', name='Power Curve'))
        fig_pc.add_vline(x=cut_in_speed, line_dash="dash", line_color="green", annotation_text="Cut-in")
        fig_pc.add_vline(x=rated_speed, line_dash="dash", line_color="blue", annotation_text="Rated")
        fig_pc.add_vline(x=cut_out_speed, line_dash="dash", line_color="red", annotation_text="Cut-out")
        fig_pc.update_layout(
            title="Turbine Power Curve",
            xaxis_title="Wind Speed (m/s)",
            yaxis_title="Power Output (kW)"
        )
        st.plotly_chart(fig_pc, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Wind Energy Feasibility Dashboard Prototype | Built with Streamlit")
