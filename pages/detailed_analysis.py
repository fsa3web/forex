import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import traceback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import pandas_ta as ta  # pandas-ta kütüphanesi kullanılıyor
from data_fetcher import get_forex_data, validate_forex_data
from technical_analysis import calculate_indicators
from signal_generator import generate_signals
from chart_visualizer import create_chart
from utils import format_number

logger = logging.getLogger(__name__)

# Keras modeli oluştur ve tanımla
def create_keras_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input katmanı
    model.add(LSTM(64, activation='relu', return_sequences=True))  # LSTM katmanı
    model.add(Dropout(0.2))  # Dropout ekleyerek overfitting'i önlüyoruz
    model.add(LSTM(64, activation='relu'))  # İkinci LSTM katmanı
    model.add(Dropout(0.2))  # Dropout
    model.add(Dense(50, activation='relu'))  # Dense katmanları ekleniyor
    model.add(Dense(1))  # Çıkış katmanı
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Ichimoku göstergesini manuel hesapla
def calculate_ichimoku(df):
    # Tenkan-sen (Conversion Line)
    nine_period_high = df['High'].rolling(window=9).max()
    nine_period_low = df['Low'].rolling(window=9).min()
    df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    twenty_six_period_high = df['High'].rolling(window=26).max()
    twenty_six_period_low = df['Low'].rolling(window=26).min()
    df['Kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2

    # Senkou Span A (Leading Span A)
    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    fifty_two_period_high = df['High'].rolling(window=52).max()
    fifty_two_period_low = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)

    # Chikou Span (Lagging Span)
    df['Chikou_Span'] = df['Close'].shift(-26)

    return df

# Daha gelişmiş teknik göstergeler ekleniyor
def generate_advanced_indicators(df):
    # Bollinger Bantları (Upper, Middle, Lower)
    bbands = df.ta.bbands(close=df['Close'], length=20, std=2)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = bbands['BBU_20_2.0'], bbands['BBM_20_2.0'], bbands['BBL_20_2.0']

    # Parabolic SAR
    df['SAR'] = df.ta.psar(high=df['High'], low=df['Low'], close=df['Close'], af=0.02, max_af=0.2)

    # Stochastic Oscillator (Stoch)
    stoch = df.ta.stoch(high=df['High'], low=df['Low'], close=df['Close'], k=14, d=3)
    df['Stoch_k'], df['Stoch_d'] = stoch['STOCHk_14_3_3'], stoch['STOCHd_14_3_3']

    # Commodity Channel Index (CCI)
    df['CCI'] = df.ta.cci(high=df['High'], low=df['Low'], close=df['Close'], length=20)

    # Momentum
    df['Momentum'] = df.ta.mom(close=df['Close'], length=10)

    # ATR (Volatilite)
    df['ATR'] = df.ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=14)

    # Ichimoku Bileşenleri
    df = calculate_ichimoku(df)

    return df

# Tahmin fonksiyonu
def generate_keras_prediction(model, X):
    return model.predict(X[-1].reshape(1, X.shape[1], 1))[0][0]

# Pandas hareketli ortalamaya dayalı tahmin fonksiyonu
def generate_pandas_prediction(df):
    return df['Close'].rolling(window=20).mean().iloc[-1]

# 60 günlük 15 dakikalık ve 2 yıllık 1 saatlik verilerle tahmin fonksiyonu
def generate_prediction(df, pair, timeframe, keras_model, scaler):
    try:
        logger.info(f"Starting prediction generation for {pair} on {timeframe} timeframe")

        if timeframe == '15m':
            resampled_df = df.resample('15min').last().dropna()  # 15 dakika dilimi
            min_data_points = 50
        elif timeframe == '1h':
            resampled_df = df.resample('1h').last().dropna()  # 1 saatlik dilim
            min_data_points = 100
        else:
            return {"error": f"Invalid timeframe: {timeframe}"}

        resampled_df = resampled_df.reset_index()

        # Eksik sütunları kontrol et
        required_columns = ['Close', 'MACD', 'RSI', 'SMA20', 'SMA50', 'EMA20', 'EMA50', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower', 'SAR', 'Stoch_k', 'Stoch_d', 'CCI', 'Momentum']
        missing_columns = [col for col in required_columns if col not in resampled_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns for {pair}: {', '.join(missing_columns)}"
            logger.error(error_msg)
            return {"error": error_msg}

        if len(resampled_df) < min_data_points:
            logger.warning(f"Insufficient data for {pair} on {timeframe} timeframe.")
            return {"insufficient_data": True}

        features = ['Close', 'MACD', 'RSI', 'SMA20', 'SMA50', 'EMA20', 'EMA50', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower', 'SAR', 'Stoch_k', 'Stoch_d', 'CCI', 'Momentum']
        X = resampled_df[features].values[:-1]
        y = resampled_df['Close'].shift(-1).dropna().values

        if len(y) == 0 or len(X) == 0:
            return {"error": "Not enough data for training Keras model."}

        # Veriyi ölçeklendirme
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))

        X_scaled = X_scaled.reshape(-1, len(features), 1)

        # Early Stopping kullanarak model eğitimi
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        keras_model.fit(X_scaled, y_scaled, epochs=100, batch_size=32, verbose=0, callbacks=[early_stopping], validation_split=0.2)

        keras_prediction_scaled = generate_keras_prediction(keras_model, X_scaled)
        keras_prediction = scaler.inverse_transform([[keras_prediction_scaled]])[0][0]

        # Pandas hareketli ortalamaya dayalı tahmin
        pandas_prediction = generate_pandas_prediction(resampled_df)

        last_close = resampled_df['Close'].iloc[-1]
        sentiment = "Bullish" if keras_prediction > last_close else "Bearish"
        pandas_sentiment = "Bullish" if pandas_prediction > last_close else "Bearish"

        # Tahmin doğruluk yüzdesi
        accuracy = (abs(keras_prediction - last_close) / last_close) * 100

        # TP1, TP2, TP3 ve Stop Loss hesaplama sentiment'e göre değişiyor
        if sentiment == "Bullish":
            tp1 = keras_prediction + 0.005 * keras_prediction  # TP1 %0.5 üst
            tp2 = keras_prediction + 0.010 * keras_prediction  # TP2 %1.0 üst
            tp3 = keras_prediction + 0.015 * keras_prediction  # TP3 %1.5 üst
            stop_loss = keras_prediction - 0.005 * keras_prediction  # Stop Loss %0.5 alt
        else:  # Bearish senaryosu için
            tp1 = keras_prediction - 0.005 * keras_prediction  # TP1 %0.5 alt
            tp2 = keras_prediction - 0.010 * keras_prediction  # TP2 %1.0 alt
            tp3 = keras_prediction - 0.015 * keras_prediction  # TP3 %1.5 alt
            stop_loss = keras_prediction + 0.005 * keras_prediction  # Stop Loss %0.5 üst

        # Ortalama tahmin süresi (örneğin 30 bar sonra tahminin gerçekleşeceği kabul edilir)
        avg_time_to_hit = 30

        logger.info(f"Prediction generated successfully for {pair} on {timeframe} timeframe")
        return {
            "keras_prediction": keras_prediction,
            "pandas_prediction": pandas_prediction,
            "sentiment": sentiment,
            "pandas_sentiment": pandas_sentiment,
            "accuracy": accuracy,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "stop_loss": stop_loss,
            "last_close": last_close,  # Anlık fiyat
            "avg_time_to_hit": avg_time_to_hit
        }
    except Exception as e:
        logger.error(f"Error in generate_prediction for {pair} on {timeframe}: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def show_analysis():
    try:
        logger.info("Starting detailed analysis")
        if 'selected_pair' not in st.session_state:
            logger.error("No currency pair selected for detailed analysis")
            st.error("No currency pair selected. Please select a pair.")
            return

        pair = st.session_state.selected_pair
        st.title(f"Detailed Forex Analysis for {pair}")

        with st.spinner(f"Loading data for {pair}..."):
            try:
                # 60 günlük 15 dakikalık veri
                data_15m = get_forex_data(pair, period="60d", interval="15m")
                # 2 yıllık 1 saatlik veri
                data_1h = get_forex_data(pair, period="2y", interval="1h")

                if validate_forex_data(data_15m, pair) and validate_forex_data(data_1h, pair):
                    df_15m = calculate_indicators(data_15m)
                    df_15m = generate_advanced_indicators(df_15m)  # Gelişmiş göstergeler
                    df_1h = calculate_indicators(data_1h)
                    df_1h = generate_advanced_indicators(df_1h)  # Gelişmiş göstergeler

                    scaler = MinMaxScaler(feature_range=(0, 1))

                    # Modeli oluştur
                    keras_model = create_keras_model((len(df_15m.columns), 1))

                    # 15 dakikalık tahmin
                    st.subheader("60 Günlük 15 Dakikalık Veriler")
                    prediction_15m = generate_prediction(df_15m, pair, '15m', keras_model, scaler)
                    if "error" not in prediction_15m:
                        st.write(f"Anlık Fiyat: {format_number(prediction_15m['last_close'])}")
                        st.write(f"Keras Prediction (15m): {format_number(prediction_15m['keras_prediction'])}")
                        st.write(f"Pandas Prediction (15m): {format_number(prediction_15m['pandas_prediction'])}")
                        st.write(f"Sentiment (Keras): {prediction_15m['sentiment']}")
                        st.write(f"Sentiment (Pandas): {prediction_15m['pandas_sentiment']}")
                        st.write(f"Accuracy: {prediction_15m['accuracy']:.2f}%")
                        st.write(f"TP1: {format_number(prediction_15m['tp1'])}")
                        st.write(f"TP2: {format_number(prediction_15m['tp2'])}")
                        st.write(f"TP3: {format_number(prediction_15m['tp3'])}")
                        st.write(f"Stop Loss: {format_number(prediction_15m['stop_loss'])}")
                        st.write(f"Avg Time to Hit: {prediction_15m['avg_time_to_hit']} bars")

                    # 1 saatlik tahmin
                    st.subheader("2 Yıllık 1 Saatlik Veriler")
                    prediction_1h = generate_prediction(df_1h, pair, '1h', keras_model, scaler)
                    if "error" not in prediction_1h:
                        st.write(f"Anlık Fiyat: {format_number(prediction_1h['last_close'])}")
                        st.write(f"Keras Prediction (1h): {format_number(prediction_1h['keras_prediction'])}")
                        st.write(f"Pandas Prediction (1h): {format_number(prediction_1h['pandas_prediction'])}")
                        st.write(f"Sentiment (Keras): {prediction_1h['sentiment']}")
                        st.write(f"Sentiment (Pandas): {prediction_1h['pandas_sentiment']}")
                        st.write(f"Accuracy: {prediction_1h['accuracy']:.2f}%")
                        st.write(f"TP1: {format_number(prediction_1h['tp1'])}")
                        st.write(f"TP2: {format_number(prediction_1h['tp2'])}")
                        st.write(f"TP3: {format_number(prediction_1h['tp3'])}")
                        st.write(f"Stop Loss: {format_number(prediction_1h['stop_loss'])}")
                        st.write(f"Avg Time to Hit: {prediction_1h['avg_time_to_hit']} bars")

                else:
                    st.error("Failed to fetch valid data for analysis.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())

        if st.button("Back to Main Page"):
            logger.info("Back to Main Page button clicked")
            st.session_state.current_page = "main"
            st.experimental_rerun()
    except Exception as e:
        logger.error(f"Error in show_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred while analyzing {pair}. Please try again later.")

if __name__ == "__main__":
    show_analysis()
