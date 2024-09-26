import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calculate_sma(data, window):
    logger.debug(f"Calculating SMA with window {window}")
    sma = data.rolling(window=window).mean()
    logger.debug(f"SMA calculation complete. First few values: {sma.head()}")
    return sma

def calculate_ema(data, span):
    logger.debug(f"Calculating EMA with span {span}")
    ema = data.ewm(span=span, adjust=False).mean()
    logger.debug(f"EMA calculation complete. First few values: {ema.head()}")
    return ema

def calculate_rsi(data, window):
    logger.debug(f"Calculating RSI with window {window}")
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    logger.debug(f"RSI calculation complete. First few values: {rsi.head()}")
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    logger.debug(f"Calculating MACD with fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    logger.debug(f"MACD calculation complete. First few values: MACD={macd_line.head()}, Signal={signal_line.head()}, Histogram={histogram.head()}")
    return macd_line, signal_line, histogram

def calculate_adx(df, period=14):
    logger.debug(f"Calculating ADX with period {period}")
    try:
        df = df.copy()
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                             np.maximum(df['High'] - df['High'].shift(1), 0), 0)
        df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                             np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
        
        df['TR14'] = df['TR'].rolling(window=period).sum()
        df['+DI14'] = df['+DM'].rolling(window=period).sum() / df['TR14'] * 100
        df['-DI14'] = df['-DM'].rolling(window=period).sum() / df['TR14'] * 100
        df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']) * 100
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        logger.debug(f"ADX calculation complete. First few values: {df['ADX'].head()}")
        return df['ADX']
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        return pd.Series(np.nan, index=df.index)

def calculate_composite_score(df):
    logger.debug("Calculating composite score")
    
    # Trend score
    trend_score = 1 if df['Close'].iloc[-1] > df['SMA50'].iloc[-1] else -1
    
    # RSI score
    rsi = df['RSI'].iloc[-1]
    rsi_score = (rsi - 50) / 50  # Normalized between -1 and 1
    
    # MACD score
    macd = df['MACD'].iloc[-1]
    signal = df['Signal'].iloc[-1]
    macd_score = 1 if macd > signal else -1
    
    # Moving average score
    ma_score = 1 if df['Close'].iloc[-1] > df['SMA50'].iloc[-1] else -1
    
    # ADX score
    adx = df['ADX'].iloc[-1]
    adx_score = 1 if adx > 25 else 0  # Consider strong trend if ADX > 25
    
    # Calculate composite score
    composite_score = (trend_score + rsi_score + macd_score + ma_score + adx_score) / 5
    
    logger.debug(f"Composite score: {composite_score}")
    return composite_score

def calculate_indicators(df):
    logger.info("Calculating indicators")
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Input DataFrame columns: {df.columns}")
    logger.debug(f"First few rows of input DataFrame:\n{df.head()}")
    
    # Calculate SMA and EMA
    df['SMA20'] = calculate_sma(df['Close'], 20)
    df['SMA50'] = calculate_sma(df['Close'], 50)
    df['EMA20'] = calculate_ema(df['Close'], 20)
    df['EMA50'] = calculate_ema(df['Close'], 50)
    logger.debug(f"DataFrame shape after SMA and EMA: {df.shape}")
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)
    logger.debug(f"DataFrame shape after RSI: {df.shape}")
    
    # Calculate MACD
    df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df['Close'])
    logger.debug(f"DataFrame shape after MACD: {df.shape}")
    
    # Calculate ADX
    df['ADX'] = calculate_adx(df)
    logger.debug(f"DataFrame shape after ADX: {df.shape}")
    
    # Calculate Composite Score
    df['CompositeScore'] = df.apply(lambda row: calculate_composite_score(df.loc[:row.name]), axis=1)
    logger.debug(f"DataFrame shape after Composite Score: {df.shape}")
    
    logger.debug(f"Output DataFrame shape: {df.shape}")
    logger.debug(f"Output DataFrame columns: {df.columns}")
    logger.debug(f"First few rows of output DataFrame:\n{df.head()}")
    
    # Check for NaN values
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        logger.warning(f"NaN values found in columns: {nan_columns}")
        for col in nan_columns:
            logger.warning(f"Number of NaN values in {col}: {df[col].isna().sum()}")
    
    return df
