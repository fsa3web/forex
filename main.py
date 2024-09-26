import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
import json
import os
import asyncio
from data_fetcher import fetch_forex_data, validate_forex_data, get_forex_data
from technical_analysis import calculate_indicators
from signal_generator import generate_signals
from chart_visualizer import create_chart
from utils import format_number
from backtester import run_backtest
from datetime import datetime, timedelta
from telegram_notifier import TelegramNotifier
from management_panel import get_currency_pairs

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Forex Day Trading Signal", layout="wide")

# Title and introduction
st.title("Forex Day Trading Signal")
st.write("Real-time analysis and trading sihhgnals for currency pairs")

# Initialize Telegram Notifier
telegram_token = st.secrets["TELEGRAM_BOT_TOKEN"]
telegram_chat_id = st.secrets["TELEGRAM_CHAT_ID"]
telegram_notifier = TelegramNotifier(telegram_token)

# Function to load signals from JSON file
def load_signals():
    try:
        with open('signals.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Function to save signals to JSON file
def save_signals(signals):
    with open('signals.json', 'w') as f:
        json.dump(signals, f)

# Function to load last sent signals from JSON file
def load_last_sent_signals():
    try:
        with open('last_sent_signals.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Function to save last sent signals to JSON file
def save_last_sent_signals(signals):
    with open('last_sent_signals.json', 'w') as f:
        json.dump(signals, f)

# Initialize signals and last sent signals
if 'signals' not in st.session_state:
    st.session_state.signals = load_signals()
if 'last_sent_signals' not in st.session_state:
    st.session_state.last_sent_signals = load_last_sent_signals()

# Function to send Telegram notification
async def send_telegram_notification(pair, signal):
    try:
        logger.debug(f"Checking if signal for {pair} is new")
        last_sent = st.session_state.last_sent_signals.get(pair, {})
        if last_sent != signal:
            logger.debug(f"Sending new Telegram notification for {pair}")
            await telegram_notifier.send_signal_notification(telegram_chat_id, pair, signal)
            st.session_state.last_sent_signals[pair] = signal
            save_last_sent_signals(st.session_state.last_sent_signals)
            logger.debug(f"Telegram notification sent successfully for {pair}")
        else:
            logger.debug(f"Skipping duplicate notification for {pair}")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification for {pair}: {str(e)}")

# Function to update data and signals
def update_data_and_signals(pair):
    try:
        # Fetch data
        logger.info(f"Fetching data for {pair}")
        data = get_forex_data(pair)
        
        if data is None:
            logger.error(f"No data available for {pair}")
            return None, []
        
        if not validate_forex_data(data, pair):
            logger.error(f"Invalid data for {pair}")
            return None, []
        
        logger.debug(f"Data for {pair}. Shape: {data.shape}")
        
        # Calculate indicators
        logger.info(f"Calculating indicators for {pair}")
        df = calculate_indicators(data)
        logger.debug(f"Indicators calculated for {pair}. Shape: {df.shape}")
        
        # Generate signals
        logger.info(f"Generating signals for {pair}")
        signals = generate_signals(df)
        logger.debug(f"Signals generated for {pair}: {signals}")
        
        # Send Telegram notifications for new signals
        for signal in signals:
            asyncio.run(send_telegram_notification(pair, signal))
        
        return df, signals
    except Exception as e:
        logger.exception(f"An error occurred while processing {pair}")
        return None, []

# Main function to display data and signals
def main():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"

    if st.session_state.current_page == "detailed_analysis":
        try:
            logger.info("Loading detailed analysis page")
            from pages.detailed_analysis import show_analysis
            show_analysis()
        except Exception as e:
            logger.error(f"Error loading detailed analysis page: {str(e)}")
            st.error("An error occurred while loading the detailed analysis page. Please try again later.")
        return

    for pair in get_currency_pairs():
        st.header(pair)
        
        df, signals = update_data_and_signals(pair)
        
        if df is not None and not df.empty:
            # Create and display chart
            logger.info(f"Creating chart for {pair}")
            fig = create_chart(df, pair)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current price
            current_price = df['Close'].iloc[-1]
            st.metric("Current Price", format_number(current_price))
            
            # Add "Detailed Analysis" button
            if st.button(f"Detailed Analysis for {pair}"):
                st.session_state.selected_pair = pair
                st.session_state.current_page = "detailed_analysis"
                st.experimental_rerun()
            
            # Display signals (if any)
            if signals:
                st.subheader("Trading Signals")
                for signal in signals:
                    st.write(f"**{signal['action']}** at {format_number(signal['price'])}")
                    st.write(f"Stop Loss: {format_number(signal['stop_loss'])}")
                    st.write(f"Take Profit 1: {format_number(signal['tp1'])}")
                    st.write(f"Take Profit 2: {format_number(signal['tp2'])}")
                    st.write(f"Take Profit 3: {format_number(signal['tp3'])}")
            else:
                st.info("No trading signals at the moment.")
            
            # Add a button for on-demand backtesting
            if st.button(f"Run Backtest for {pair}"):
                st.subheader("Backtesting Results")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                try:
                    logger.info(f"Running backtest for {pair} from {start_date} to {end_date}")
                    backtest_report, equity_curve = run_backtest([pair], start_date, end_date)
                    if backtest_report and equity_curve:
                        st.text(backtest_report)
                        st.plotly_chart(equity_curve, use_container_width=True)
                    else:
                        st.error("Failed to run backtest. Please check the logs for more information.")
                        logger.error(f"Backtest failed for {pair}")
                except Exception as backtest_error:
                    logger.exception(f"Error running backtest for {pair}: {str(backtest_error)}")
                    st.error(f"Error running backtest for {pair}: {str(backtest_error)}")
        else:
            st.error(f"No data available for {pair}. Please check the logs for more information.")
        
        # Add a separator between currency pairs
        st.markdown("---")

    # Add a footer
    st.write("Disclaimer: This application is for educational purposes only. Always do your own research before making any investment decisions.")

if __name__ == "__main__":
    main()
