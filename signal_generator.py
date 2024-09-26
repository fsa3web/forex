import pandas as pd
import numpy as np
import logging
from technical_analysis import calculate_composite_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calculate_adx(df, period=14):
    """Calculate the Average Directional Index (ADX)"""
    df = df.copy()
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                   abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1))
                         > (df['Low'].shift(1) - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low'])
                         > (df['High'] - df['High'].shift(1)),
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    df['TR14'] = df['TR'].rolling(window=period).sum()
    df['+DI14'] = df['+DM'].rolling(window=period).sum() / df['TR14'] * 100
    df['-DI14'] = df['-DM'].rolling(window=period).sum() / df['TR14'] * 100
    df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] +
                                                 df['-DI14']) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()
    return df['ADX']


def generate_signals(df):
    """
    Generate trading signals based on technical analysis.
    
    :param df: pandas DataFrame with OHLC and indicator data
    :return: list of dictionaries containing signal information
    """
    logger.info("Starting signal generation process")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input DataFrame columns: {df.columns}")

    signals = []
    last_row = df.iloc[-1]

    # Calculate ADX
    df['ADX'] = calculate_adx(df)

    # Calculate composite score
    composite_score = calculate_composite_score(df)

    # Debug logging for key variables
    logger.debug(f"Last close price: {last_row['Close']:.5f}")
    logger.debug(f"EMA20: {last_row['EMA20']:.5f}")
    logger.debug(f"EMA50: {last_row['EMA50']:.5f}")
    logger.debug(f"RSI: {last_row['RSI']:.2f}")
    logger.debug(f"MACD: {last_row['MACD']:.5f}")
    logger.debug(f"Signal: {last_row['Signal']:.5f}")
    logger.debug(f"ADX: {last_row['ADX']:.2f}")
    logger.debug(f"Composite score: {composite_score:.2f}")

    # Check for buying conditions
    logger.debug("Checking buying conditions")
    if (last_row['Close'] > last_row['EMA20'] > last_row['EMA50']
            and last_row['RSI'] > 50 and last_row['RSI'] < 70
            and last_row['MACD'] > last_row['Signal'] and last_row['ADX'] > 25
            and  # Strong trend
            composite_score > 0.6):  # High composite score for buy signal

        logger.info("Buy signal generated")
        entry = last_row['Close']
        stop_loss = last_row['Low']  # Using the most recent low as stop loss
        take_profit1 = entry + (entry - stop_loss)
        take_profit2 = entry + 2 * (entry - stop_loss)
        take_profit3 = entry + 3 * (entry - stop_loss)

        signals.append({
            'action': 'BUY',
            'price': entry,
            'stop_loss': stop_loss,
            'tp1': take_profit1,
            'tp2': take_profit2,
            'tp3': take_profit3,
            'probability': composite_score
        })
    else:
        logger.debug("Buy conditions not met")

    # Check for selling conditions
    logger.debug("Checking selling conditions")
    if (last_row['Close'] < last_row['EMA20'] < last_row['EMA50']
            and last_row['RSI'] < 55 and last_row['RSI'] > 25  # Genişletilmiş RSI aralığı
            and last_row['MACD'] < last_row['Signal'] 
            and last_row['ADX'] > 20  # ADX koşulunu düşürdük
            and  # Daha zayıf trendler için de sinyal verebilir
            composite_score < -0.55):  # Daha esnek bir bileşik puan eşiği

        logger.info("Sell signal generated")
        entry = last_row['Close']
        stop_loss = last_row['High']  # Using the most recent high as stop loss
        take_profit1 = entry - (stop_loss - entry)
        take_profit2 = entry - 2 * (stop_loss - entry)
        take_profit3 = entry - 3 * (stop_loss - entry)

        signals.append({
            'action': 'SELL',
            'price': entry,
            'stop_loss': stop_loss,
            'tp1': take_profit1,
            'tp2': take_profit2,
            'tp3': take_profit3,
            'probability': abs(composite_score)
        })
    else:
        logger.debug("Sell conditions not met")

    if not signals:
        logger.info("No signals generated")
    else:
        logger.info(f"Generated signals: {signals}")

    return signals
