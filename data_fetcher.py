import yfinance as yf
import pandas as pd
import logging
import time
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def fetch_forex_data(symbol, period="1d", interval="5m", max_retries=3, initial_delay=1):
    """
    Fetch forex data from Yahoo Finance with retry mechanism and error handling.

    :param symbol: The forex pair symbol (e.g., "EURUSD=X")
    :param period: The time period to fetch data for (default: "1d")
    :param interval: The interval between data points (default: "5m")
    :param max_retries: Maximum number of retry attempts (default: 3)
    :param initial_delay: Initial delay between retries in seconds (default: 1)
    :return: A pandas DataFrame with the forex data, or None if there's an error
    """
    logger.info(f"Attempting to fetch data for {symbol} with period={period} and interval={interval}")
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1}/{max_retries} to fetch data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            logger.debug(f"Raw data received for {symbol}. Shape: {data.shape}, Columns: {data.columns}")
            
            if validate_forex_data(data, symbol):
                logger.info(f"Successfully fetched and validated data for {symbol}. Rows: {len(data)}")
                return data
            else:
                logger.warning(f"Invalid data received for {symbol}. Retrying...")
        
        except RequestException as e:
            logger.error(f"Network error while fetching data for {symbol}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching data for {symbol}: {str(e)}", exc_info=True)
        
        if attempt < max_retries - 1:
            delay = initial_delay * (2 ** attempt)
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return None

def validate_forex_data(data, symbol):
    """
    Validate the fetched forex data.

    :param data: pandas DataFrame with the forex data
    :param symbol: The forex pair symbol
    :return: True if the data is valid, False otherwise
    """
    if data is None or data.empty:
        logger.error(f"Empty DataFrame received for {symbol}")
        return False
    
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Missing required OHLC columns for {symbol}")
        return False
    
    if data[required_columns].isna().any().any():
        logger.error(f"NaN values found in OHLC data for {symbol}")
        return False
    
    return True

def get_forex_data(symbol, period="1d", interval="5m"):
    """
    Get forex data in real-time without caching.

    :param symbol: The forex pair symbol (e.g., "EURUSD=X")
    :param period: The time period to fetch data for (default: "1d")
    :param interval: The interval between data points (default: "5m")
    :return: A pandas DataFrame with the forex data, or None if there's an error
    """
    logger.info(f"Getting forex data for {symbol}")
    data = fetch_forex_data(symbol, period, interval)
    
    if data is None:
        logger.error(f"Failed to fetch data for {symbol}")
        return None
    
    return data
