import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI Trader Dashboard", layout="wide")

# --- Title and Header ---
st.title("🤖 AI Trader Forecast")

# --- Load and Display Data ---
PREDICTIONS_FILE = 'predictions.csv'

if os.path.exists(PREDICTIONS_FILE):
    # Get the last modification time of the file
    last_updated_timestamp = os.path.getmtime(PREDICTIONS_FILE)
    last_updated_time = datetime.fromtimestamp(last_updated_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"**Last prediction run:** {last_updated_time}")

    # Load the predictions from the CSV file
    df = pd.read_csv(PREDICTIONS_FILE)

    # --- Interactive Filters in the Sidebar ---
    st.sidebar.header("Filter Options")
    
    # Confidence slider
    confidence_threshold = st.sidebar.slider(
        'Filter by Confidence (%)', 0, 100, 50
    )
    
    # Ticker multi-select
    all_tickers = df['Ticker'].unique()
    selected_tickers = st.sidebar.multiselect(
        'Select Tickers', all_tickers, default=all_tickers
    )

    # --- Apply Filters ---
    # Create a numeric confidence column for filtering
    df['ConfidenceValue'] = pd.to_numeric(df['Confidence'].str.replace('%', ''))
    
    filtered_df = df[
        (df['ConfidenceValue'] >= confidence_threshold) &
        (df['Ticker'].isin(selected_tickers))
    ].drop(columns=['ConfidenceValue']) # Drop the helper column before displaying

    # --- Display the Filtered Table ---
    st.header("Model Predictions")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

else:
    st.error("The 'predictions.csv' file was not found. Please run the 'automated_trader.py' script first to generate the predictions.")