import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def create_chart(df, pair):
    """
    Create an interactive chart using plotly.
    
    :param df: pandas DataFrame with OHLC and indicator data
    :param pair: string representing the currency pair
    :return: plotly Figure object
    """
    try:
        logger.info(f"Creating chart for {pair}")
        logger.debug(f"Input DataFrame for {pair}:\n{df.head()}")
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns}")
        
        if df.empty:
            logger.warning(f"Empty DataFrame for {pair}")
            return create_error_chart(pair, "No data available")
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required OHLC columns for {pair}")
            return create_error_chart(pair, "Missing OHLC data")
        
        if df[required_columns].isna().all().any():
            logger.error(f"All NaN values in one or more required columns for {pair}")
            return create_error_chart(pair, "Invalid OHLC data (all NaN)")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{pair} Price', 'Indicators'),
                            row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Price'),
                      row=1, col=1)

        # Add moving averages
        for ma in ['SMA20', 'SMA50', 'EMA20', 'EMA50']:
            if ma in df.columns and not df[ma].isna().all():
                fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1)), row=1, col=1)
            else:
                logger.warning(f"{ma} not available or all NaN for {pair}")

        # RSI
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=1)), row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        else:
            logger.warning(f"RSI not available or all NaN for {pair}")

        # MACD
        if all(indicator in df.columns for indicator in ['MACD', 'Signal']) and not (df['MACD'].isna().all() or df['Signal'].isna().all()):
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='red', width=1)), row=2, col=1)
        else:
            logger.warning(f"MACD or Signal not available or all NaN for {pair}")
        
        fig.update_layout(height=800, title_text=f"{pair} Analysis")
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
        
        logger.info(f"Chart creation completed for {pair}")
        return fig
    except Exception as e:
        logger.error(f"Error creating chart for {pair}: {str(e)}", exc_info=True)
        return create_error_chart(pair, str(e))

def create_error_chart(pair, error_message):
    """
    Create a placeholder chart with an error message.
    
    :param pair: string representing the currency pair
    :param error_message: string containing the error message
    :return: plotly Figure object
    """
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"Error: {error_message}",
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(height=800, title_text=f"{pair} - Chart Unavailable")
    return fig
