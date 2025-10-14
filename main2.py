import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# --- Streamlit Setup ---
st.set_page_config(page_title="Stock Trend Predictor", layout="wide", page_icon="📈")

st.markdown("""
<div style="background-color:#f0f2f6;padding:15px;border-radius:10px">
    <h1 style="color:#0f4c81;text-align:center;">📈 Stock Trend Predictor</h1>
    <p style="text-align:center;">Predict stock trends using Logistic Regression and Technical Indicators</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Stock & Prediction Settings")
symbol = st.sidebar.text_input("Stock Symbol:", "RELIANCE.NS")

# Limit prediction date to avoid selecting future
today = pd.Timestamp.today().normalize()
max_date = today - pd.Timedelta(days=2)
predict_date = st.sidebar.date_input("Prediction Date:", value=max_date, max_value=max_date)

st.sidebar.markdown("### 📊 Indicators")
use_ema = st.sidebar.checkbox("Use EMA (5, 10)", value=True)
use_rsi = st.sidebar.checkbox("Use RSI (14)")
use_macd = st.sidebar.checkbox("Use MACD")

predict_button = st.sidebar.button("🔮 Predict Trend")

# --- Indicator Functions ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# --- Main Logic ---
if predict_button:
    end_date = today
    start_date = end_date - pd.Timedelta(days=365)

    with st.spinner(f"📥 Fetching data for **{symbol}**..."):
        data = yf.download(symbol, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ No data found for this symbol.")
        st.stop()

    # Save raw OHLC for candlestick
    raw_ohlc = data[['Open', 'High', 'Low', 'Close']].dropna() if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']) else None

    # Prepare df for indicators and prediction
    df = data.copy()

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Detect 'Close'
    close_col = None
    for col in df.columns:
        if 'close' in col.lower():
            close_col = col
            break
    if close_col is None:
        st.error("❌ No 'Close' column found.")
        st.stop()

    df.rename(columns={close_col: 'Close'}, inplace=True)

    # --- Indicators ---
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()

    if use_ema:
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    if use_rsi:
        df['RSI_14'] = compute_rsi(df['Close'])
    if use_macd:
        df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])

    # --- Target: Up (1) or Down (0) next day
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    if len(df) < 20:
        st.error("❌ Not enough data after applying indicators.")
        st.stop()

    # --- Features and Model ---
    features = ['SMA_5', 'SMA_10']
    if use_ema:
        features += ['EMA_5', 'EMA_10']
    if use_rsi:
        features.append('RSI_14')
    if use_macd:
        features += ['MACD', 'MACD_Signal']

    X = df[features]
    y = df['Target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # --- Prediction Date Handling ---
    input_date = pd.Timestamp(predict_date)

    if input_date > df.index[-2]:
        st.warning("⚠️ Not enough future data to predict from this date. Using latest valid date instead.")
        input_date = df.index[-2]

    while input_date not in df.index:
        input_date -= pd.Timedelta(days=1)
        if input_date < df.index[0]:
            st.error("❌ No valid trading data before this date.")
            st.stop()

    future_dates = df.index[df.index > input_date]
    if len(future_dates) == 0:
        st.error("❌ No future trading data available for prediction.")
        st.stop()

    next_day = future_dates[0]
    input_features = df.loc[input_date, features].to_numpy().reshape(1, -1)
    pred = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0]

    # --- Display Prediction ---
    st.subheader("📊 Prediction Result")
    col1, col2, col3 = st.columns(3)
    col3.metric("Latest Close", f"₹{df['Close'].iloc[-1]:.2f}")

    if pred == 1:
        col1.success(f"UP 📈\nConfidence: {proba[1]*100:.2f}%")
    else:
        col2.error(f"DOWN 📉\nConfidence: {proba[0]*100:.2f}%")

    st.markdown(f"**Prediction Date:** {next_day.date()} | Based on selected indicators")

    # --- Charts Tabs ---
    tabs = st.tabs(["📈 SMA/EMA Chart", "📉 Candlestick Chart", "📋 Raw Data"])

    # --- Tab 1: Indicator Chart ---
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close', color='blue', linewidth=2)
        ax.plot(df.index, df['SMA_5'], label='SMA 5', color='orange', linestyle='--')
        ax.plot(df.index, df['SMA_10'], label='SMA 10', color='green', linestyle='--')
        if use_ema:
            ax.plot(df.index, df['EMA_5'], label='EMA 5', color='purple', linestyle='-.')
            ax.plot(df.index, df['EMA_10'], label='EMA 10', color='brown', linestyle='-.')
        ax.axvline(input_date, color='red', linestyle=':', label='Prediction Date')
        ax.set_title(f"{symbol} - Price & Indicators")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Tab 2: Candlestick Chart ---
    with tabs[1]:
        if raw_ohlc is not None and not raw_ohlc.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=raw_ohlc.index,
                open=raw_ohlc['Open'],
                high=raw_ohlc['High'],
                low=raw_ohlc['Low'],
                close=raw_ohlc['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.update_layout(title=f'{symbol} - Candlestick Chart',
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Candlestick chart not available (missing Open/High/Low/Close data).")

    # --- Tab 3: Raw Data ---
    with tabs[2]:
        st.dataframe(df.tail(50))

    st.success("✅ Prediction complete!")
