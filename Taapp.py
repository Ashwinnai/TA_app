import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
from numpy import NaN as npNaN


# Streamlit application
def main():
    # Set up the page configuration
    st.set_page_config(
        page_title="Advanced Technical Indicators Analysis",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .main {background-color: #1f1f1f;}
        h1 {color: #ffd700 !important;}
        .stSidebar {background-color: #333;}
        .stButton>button {background-color: #ff4b4b; color: #fff;}
        .stDownloadButton>button {background-color: #00b300; color: #fff;}
        .st-expander-content {background-color: #2c2c2c; color: #fff;}
        .sidebar-content {color: #fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Application Title
    st.markdown(
        "<h1 style='text-align: center; color: #ffd700;'>📊 Advanced Technical Indicators Analysis</h1>",
        unsafe_allow_html=True,
    )

    # Sidebar for user input and configuration
    with st.sidebar:
        st.header("⚙️ Configuration Panel")

        # User Guide
        with st.expander("📖 User Guide", expanded=False):
            st.markdown(
                """
                **Welcome to the Advanced Technical Indicators Analysis Tool!**

                - **Upload Your Data**: Upload a CSV file containing your financial data.
                - **Select Columns**: Choose relevant columns (Open, High, Low, Close, Volume) for analysis.
                - **Choose Indicators**: Pick from different categories (Overlap, Momentum, Volatility, Trend, Others).
                - **Customize Parameters**: Adjust the parameters for each indicator.
                - **View and Analyze**: Visualize the indicators on interactive charts.
                - **Export Results**: Download the processed data in CSV or Excel format.

                **Created by Ashwin Nair**
                """
            )

        # File uploader
        uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

        # Check if a file is uploaded
        if uploaded_file is not None:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)

            # Display the first few rows of the dataframe
            st.subheader("🔍 Data Overview")
            st.write(df.head())

            # Ensure correct column selection for dates
            st.subheader("🗓 Select Date Column")
            date_col = st.selectbox("Select the Date column", df.columns.tolist())
            df[date_col] = pd.to_datetime(df[date_col])

            # Set the date column as index
            df.set_index(date_col, inplace=True)

            # Display the columns available for analysis
            columns = df.columns.tolist()
            st.subheader("📈 Available Columns for Analysis")
            st.write(columns)

            # Option to select columns
            st.subheader("🧮 Select Columns for Indicators")
            col1, col2 = st.columns(2)
            with col1:
                open_col = st.selectbox("Select the Open column", columns, index=columns.index('Open') if 'Open' in columns else 0)
                high_col = st.selectbox("Select the High column", columns, index=columns.index('High') if 'High' in columns else 0)
                low_col = st.selectbox("Select the Low column", columns, index=columns.index('Low') if 'Low' in columns else 0)
            with col2:
                close_col = st.selectbox("Select the Close column", columns, index=columns.index('Close') if 'Close' in columns else 0)
                volume_col = st.selectbox("Select the Volume column", columns, index=columns.index('Volume') if 'Volume' in columns else 0)

            # Technical indicators selection
            indicators = {
                "Overlap": ["SMA", "EMA", "VWAP", "BBANDS", "Ichimoku"],
                "Momentum": ["RSI", "MACD", "Stochastic Oscillator", "CCI", "ROC"],
                "Volatility": ["ATR", "Donchian Channels", "Bollinger Bands"],
                "Trend": ["ADX", "Parabolic SAR"],
                "Others": ["Volume", "MFI", "OBV"]
            }

            st.subheader("📊 Indicator Configuration")

            # Tabs for different indicator categories
            tabs = st.tabs(["Overlap", "Momentum", "Volatility", "Trend", "Others"])

            # Overlap Indicators
            with tabs[0]:
                st.markdown("### Select Overlap Indicators")
                overlap_params = {}
                selected_overlap_indicators = st.multiselect("Select Overlap Indicators", indicators["Overlap"])
                for indicator in selected_overlap_indicators:
                    st.markdown(f"**Parameters for {indicator}:**")
                    if indicator in ["SMA", "EMA"]:
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        overlap_params[indicator] = {"length": period}
                    elif indicator == "VWAP":
                        st.write("VWAP selected: No additional parameters needed.")
                    elif indicator == "BBANDS":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=20)
                        std_dev = st.slider(f"Standard Deviation for {indicator}", min_value=0.1, max_value=5.0, value=2.0)
                        overlap_params[indicator] = {"length": period, "stddev": std_dev}
                    elif indicator == "Ichimoku":
                        st.write("Ichimoku selected: No additional parameters needed.")

            # Momentum Indicators
            with tabs[1]:
                st.markdown("### Select Momentum Indicators")
                momentum_params = {}
                selected_momentum_indicators = st.multiselect("Select Momentum Indicators", indicators["Momentum"])
                for indicator in selected_momentum_indicators:
                    st.markdown(f"**Parameters for {indicator}:**")
                    if indicator == "RSI":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        momentum_params[indicator] = {"length": period}
                    elif indicator == "MACD":
                        fast = st.slider("Fast Period", min_value=1, max_value=100, value=12)
                        slow = st.slider("Slow Period", min_value=1, max_value=100, value=26)
                        signal = st.slider("Signal Period", min_value=1, max_value=100, value=9)
                        momentum_params[indicator] = {"fast": fast, "slow": slow, "signal": signal}
                    elif indicator == "Stochastic Oscillator":
                        k_period = st.slider(f"Period for %K", min_value=1, max_value=100, value=14)
                        d_period = st.slider(f"Period for %D", min_value=1, max_value=100, value=3)
                        momentum_params[indicator] = {"k": k_period, "d": d_period}
                    elif indicator == "CCI":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=20)
                        momentum_params[indicator] = {"length": period}
                    elif indicator == "ROC":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        momentum_params[indicator] = {"length": period}

            # Volatility Indicators
            with tabs[2]:
                st.markdown("### Select Volatility Indicators")
                volatility_params = {}
                selected_volatility_indicators = st.multiselect("Select Volatility Indicators", indicators["Volatility"])
                for indicator in selected_volatility_indicators:
                    st.markdown(f"**Parameters for {indicator}:**")
                    if indicator == "ATR":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        volatility_params[indicator] = {"length": period}
                    elif indicator == "Donchian Channels":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=20)
                        volatility_params[indicator] = {"length": period}
                    elif indicator == "Bollinger Bands":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=20)
                        std_dev = st.slider(f"Standard Deviation for {indicator}", min_value=0.1, max_value=5.0, value=2.0)
                        volatility_params[indicator] = {"length": period, "stddev": std_dev}

            # Trend Indicators
            with tabs[3]:
                st.markdown("### Select Trend Indicators")
                trend_params = {}
                selected_trend_indicators = st.multiselect("Select Trend Indicators", indicators["Trend"])
                for indicator in selected_trend_indicators:
                    st.markdown(f"**Parameters for {indicator}:**")
                    if indicator == "ADX":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        trend_params[indicator] = {"length": period}
                    elif indicator == "Parabolic SAR":
                        step = st.slider(f"Step for {indicator}", min_value=0.001, max_value=0.2, value=0.02)
                        max_step = st.slider(f"Maximum Step for {indicator}", min_value=0.1, max_value=1.0, value=0.2)
                        trend_params[indicator] = {"step": step, "max_step": max_step}

            # Others Indicators
            with tabs[4]:
                st.markdown("### Select Other Indicators")
                other_params = {}
                selected_other_indicators = st.multiselect("Select Other Indicators", indicators["Others"])
                for indicator in selected_other_indicators:
                    st.markdown(f"**Parameters for {indicator}:**")
                    if indicator == "Volume":
                        st.write("Volume selected: No additional parameters needed.")
                    elif indicator == "MFI":
                        period = st.slider(f"Period for {indicator}", min_value=1, max_value=100, value=14)
                        other_params[indicator] = {"length": period}
                    elif indicator == "OBV":
                        st.write("OBV selected: No additional parameters needed.")

            # Plot Configuration
            st.subheader("📅 Plot Configuration")
            plot_choice = st.radio(
                "Choose where to display the indicators",
                ("Overlay on Price Plot", "Separate Plot")
            )

            # Date range selection
            st.subheader("📅 Select Date Range")
            date_range = st.date_input(
                "Select the date range",
                [df.index.min(), df.index.max()]
            )

    # Main area for plotting and displaying results
    if uploaded_file is not None:
        # Filter data based on the selected date range
        df_filtered = df.loc[(df.index >= pd.to_datetime(date_range[0])) & (df.index <= pd.to_datetime(date_range[1]))]

        # Apply selected indicators to the data
        with st.spinner("🔄 Processing data..."):
            for indicator in selected_overlap_indicators:
                df_filtered = apply_indicator(df_filtered, indicator, high_col, low_col, close_col, volume_col, overlap_params.get(indicator, {}))

            for indicator in selected_momentum_indicators:
                df_filtered = apply_indicator(df_filtered, indicator, high_col, low_col, close_col, volume_col, momentum_params.get(indicator, {}))

            for indicator in selected_volatility_indicators:
                df_filtered = apply_indicator(df_filtered, indicator, high_col, low_col, close_col, volume_col, volatility_params.get(indicator, {}))

            for indicator in selected_trend_indicators:
                df_filtered = apply_indicator(df_filtered, indicator, high_col, low_col, close_col, volume_col, trend_params.get(indicator, {}))

            for indicator in selected_other_indicators:
                df_filtered = apply_indicator(df_filtered, indicator, high_col, low_col, close_col, volume_col, other_params.get(indicator, {}))

        # Plot the data with indicators
        plot_data(
            df_filtered,
            selected_overlap_indicators,
            selected_momentum_indicators,
            selected_volatility_indicators,
            selected_trend_indicators,
            selected_other_indicators,
            open_col,
            high_col,
            low_col,
            close_col,
            volume_col,
            plot_choice,
            overlap_params,
            momentum_params,
            volatility_params,
            trend_params,
            other_params
        )

        # Export data
        st.subheader("⬇️ Export Data")
        csv = df_filtered.to_csv().encode('utf-8')
        excel_file = BytesIO()
        df_filtered.to_excel(excel_file, index=True, engine='xlsxwriter')
        excel_file.seek(0)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='indicator_data.csv',
                mime='text/csv',
            )
        with col2:
            st.download_button(
                label="Download data as Excel",
                data=excel_file,
                file_name='indicator_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

# Function to apply technical indicators
def apply_indicator(df, indicator, high_col, low_col, close_col, volume_col, params):
    try:
        if indicator == "SMA":
            df[f'SMA_{params.get("length", 14)}'] = ta.sma(df[close_col], length=params.get("length", 14))
        elif indicator == "EMA":
            df[f'EMA_{params.get("length", 14)}'] = ta.ema(df[close_col], length=params.get("length", 14))
        elif indicator == "VWAP":
            df['VWAP'] = ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col])
        elif indicator == "BBANDS":
            bbands = ta.bbands(df[close_col], length=params.get("length", 20), std=params.get("stddev", 2.0))
            df = pd.concat([df, bbands], axis=1)
        elif indicator == "Ichimoku":
            ichimoku = ta.ichimoku(df[high_col], df[low_col], df[close_col])
            df = pd.concat([df, ichimoku], axis=1)
        elif indicator == "RSI":
            df[f'RSI_{params.get("length", 14)}'] = ta.rsi(df[close_col], length=params.get("length", 14))
        elif indicator == "MACD":
            macd = ta.macd(df[close_col], fast=params.get("fast", 12), slow=params.get("slow", 26), signal=params.get("signal", 9))
            df = pd.concat([df, macd], axis=1)
        elif indicator == "Stochastic Oscillator":
            df['%K'], df['%D'] = ta.stoch(df[high_col], df[low_col], df[close_col], k=params.get("k", 14), d=params.get("d", 3))
        elif indicator == "CCI":
            df[f'CCI_{params.get("length", 20)}'] = ta.cci(df[high_col], df[low_col], df[close_col], length=params.get("length", 20))
        elif indicator == "ROC":
            df[f'ROC_{params.get("length", 14)}'] = ta.roc(df[close_col], length=params.get("length", 14))
        elif indicator == "ATR":
            df[f'ATR_{params.get("length", 14)}'] = ta.atr(df[high_col], df[low_col], df[close_col], length=params.get("length", 14))
        elif indicator == "Donchian Channels":
            donchian = ta.donchian(df[high_col], df[low_col], length=params.get("length", 20))
            df = pd.concat([df, donchian], axis=1)
        elif indicator == "ADX":
            df[f'ADX_{params.get("length", 14)}'] = ta.adx(df[high_col], df[low_col], df[close_col], length=params.get("length", 14))
        elif indicator == "Parabolic SAR":
            df['PSAR'] = ta.psar(df[high_col], df[low_col], step=params.get("step", 0.02), max_step=params.get("max_step", 0.2))
        elif indicator == "Volume":
            df['Volume'] = df[volume_col]
        elif indicator == "MFI":
            df[f'MFI_{params.get("length", 14)}'] = ta.mfi(df[high_col], df[low_col], df[close_col], df[volume_col], length=params.get("length", 14))
        elif indicator == "OBV":
            df['OBV'] = ta.obv(df[close_col], df[volume_col])
    except Exception as e:
        st.error(f"Error applying indicator {indicator}: {str(e)}")
    return df

# Function to plot data with selected indicators
def plot_data(df, selected_overlap_indicators, selected_momentum_indicators, selected_volatility_indicators, selected_trend_indicators, selected_other_indicators,
              open_col, high_col, low_col, close_col, volume_col, plot_choice,
              overlap_params, momentum_params, volatility_params, trend_params, other_params):
    # Main Price Plot with Candlesticks
    price_fig = go.Figure()

    # Add candlestick trace
    price_fig.add_trace(go.Candlestick(
        x=df.index,
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col],
        name='Market Data'
    ))

    # Overlay the selected overlap indicators on the price chart
    for indicator in selected_overlap_indicators:
        if indicator in ["SMA", "EMA"]:
            column_name = f'{indicator}_{overlap_params[indicator]["length"]}'
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df[column_name],
                mode='lines',
                name=indicator
            ))
        elif indicator == "VWAP":
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['VWAP'],
                mode='lines',
                name='VWAP'
            ))
        elif indicator == "BBANDS":
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BBL_20_2.0'],
                line=dict(color='rgba(173,216,230,0.5)'),
                name='Lower Band'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BBM_20_2.0'],
                line=dict(color='rgba(173,216,230,0.8)'),
                name='Middle Band'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BBU_20_2.0'],
                line=dict(color='rgba(173,216,230,0.5)'),
                name='Upper Band'
            ))
        elif indicator == "Ichimoku":
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ITS_9'],
                mode='lines',
                name='Tenkan-sen'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['IKS_26'],
                mode='lines',
                name='Kijun-sen'
            ))
            price_fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ICS_52'],
                mode='lines',
                name='Senkou Span A'
            ))

    # Update layout for price chart
    price_fig.update_layout(
        title="Price Chart with Overlaid Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700
    )

    # Display price plot
    st.plotly_chart(price_fig, use_container_width=True)

    # Create separate plots for momentum and volatility indicators if selected
    if plot_choice == "Separate Plot":
        # Momentum Indicators Plot
        if selected_momentum_indicators:
            momentum_fig = go.Figure()
            for indicator in selected_momentum_indicators:
                if indicator == "RSI":
                    column_name = f'RSI_{momentum_params[indicator]["length"]}'
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[column_name],
                        mode='lines',
                        name=indicator
                    ))
                elif indicator == "MACD":
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACD_12_26_9'],
                        mode='lines',
                        name='MACD'
                    ))
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACDs_12_26_9'],
                        mode='lines',
                        name='Signal Line'
                    ))
                    momentum_fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['MACDh_12_26_9'],
                        name='MACD Histogram'
                    ))
                elif indicator == "Stochastic Oscillator":
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['%K'],
                        mode='lines',
                        name='%K'
                    ))
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['%D'],
                        mode='lines',
                        name='%D'
                    ))
                elif indicator == "CCI":
                    column_name = f'CCI_{momentum_params[indicator]["length"]}'
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[column_name],
                        mode='lines',
                        name=indicator
                    ))
                elif indicator == "ROC":
                    column_name = f'ROC_{momentum_params[indicator]["length"]}'
                    momentum_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[column_name],
                        mode='lines',
                        name=indicator
                    ))
            momentum_fig.update_layout(
                title="Momentum Indicators",
                xaxis_title="Date",
                yaxis_title="Value",
                legend_title="Legend",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(momentum_fig, use_container_width=True)

        # Volatility Indicators Plot
        if selected_volatility_indicators:
            volatility_fig = go.Figure()
            for indicator in selected_volatility_indicators:
                if indicator == "ATR":
                    column_name = f'ATR_{volatility_params[indicator]["length"]}'
                    volatility_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[column_name],
                        mode='lines',
                        name=indicator
                    ))
                elif indicator == "Donchian Channels":
                    volatility_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['DCL_20_20'],
                        line=dict(color='rgba(255,192,203,0.5)'),
                        name='Lower Donchian'
                    ))
                    volatility_fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['DCU_20_20'],
                        line=dict(color='rgba(255,192,203,0.5)'),
                        name='Upper Donchian'
                    ))
            volatility_fig.update_layout(
                title="Volatility Indicators",
                xaxis_title="Date",
                yaxis_title="Value",
                legend_title="Legend",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(volatility_fig, use_container_width=True)

# Run the application
if __name__ == "__main__":
    main()
# Footer
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #1f1f1f;">
        <p>|Developed with ❤️ by Ashwin Nair | 
    </div>
    """, unsafe_allow_html=True)
