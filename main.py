import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interactive Stock Predictor", layout="wide", page_icon="📈")

# --- Header ---
st.markdown("""
    <div style="background-color:#f0f2f6;padding:15px;border-radius:10px">
    <h1 style="color:#0f4c81;text-align:center;">📈 Interactive Stock Trend Predictor</h1>
    <p style="text-align:center;">Predict stock trends with Logistic Regression + SMA and explore historical data interactively</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("📌 Stock & Prediction Date")
symbol = st.sidebar.text_input("Stock Symbol:", "RELIANCE.NS")
predict_date = st.sidebar.date_input("Prediction Date:")

st.sidebar.markdown("### ℹ️ Instructions")
st.sidebar.markdown("""
- Enter a valid stock symbol (e.g., RELIANCE.NS, AAPL, TSLA).  
- Pick a prediction date.  
- Click **Predict Trend**.
""")

predict_button = st.sidebar.button("Predict Trend")

if predict_button:
    # --- Fetch Data ---
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=365)
    st.info(f"Fetching data for **{symbol}** from {start_date.date()} to {end_date.date()}...")
    
    data = yf.download(symbol, start=start_date, end=end_date)
    
    if data.empty:
        st.error("⚠️ No data found for this symbol.")
        st.stop()

    df = data.copy()

    # --- Detect Close column robustly ---
    close_col = None
    for col in df.columns:
        # Handle MultiIndex columns
        col_name = "_".join(col) if isinstance(col, tuple) else str(col)
        if 'close' in col_name.lower():
            close_col = col
            break

    if close_col is None:
        st.error("⚠️ No 'Close' column found in the downloaded data.")
        st.stop()

    df = df[[close_col]].copy()
    df.rename(columns={close_col: 'Close'}, inplace=True)

    # --- Compute SMA ---
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    if len(df) < 10:
        st.error("⚠️ Not enough data after computing SMAs.")
        st.stop()

    # --- Train Logistic Regression ---
    X = df[['SMA_5','SMA_10']]
    y = df['Target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # --- Find nearest trading date ---
    input_date = pd.Timestamp(predict_date)
    while input_date not in df.index:
        input_date -= pd.Timedelta(days=1)
        if input_date < df.index[0]:
            st.error("⚠️ No trading data available before this date.")
            st.stop()

    future_dates = df.index[df.index > input_date]
    if len(future_dates) == 0:
        st.error("⚠️ No future trading data available for prediction.")
        st.stop()

    next_day = future_dates[0]
    features = df.loc[input_date, ['SMA_5','SMA_10']].to_numpy().reshape(1,-1)
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    # --- Display Prediction ---
    st.subheader("📊 Prediction Result")
    col1, col2, col3 = st.columns(3)
    
    latest_close = float(df['Close'].iloc[-1])  # <-- Fix applied here
    col3.metric("Latest Close Price", f"₹{latest_close:.2f}")
    
    if pred == 1:
        col1.success(f"UP 📈\nConfidence: {proba[1]*100:.2f}%")
        col2.write(" ")
    else:
        col2.error(f"DOWN 📉\nConfidence: {proba[0]*100:.2f}%")
        col1.write(" ")

    st.markdown(f"**Prediction Date:** {next_day.date()} | **Based on SMA Indicators**")

    # --- Plotting with Matplotlib ---
    st.subheader("📈 Price Trend & SMA")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label='Close Price', color='#0f4c81', linewidth=2)
    ax.plot(df.index, df['SMA_5'], label='SMA 5', color='#f39c12', linestyle='--', linewidth=2)
    ax.plot(df.index, df['SMA_10'], label='SMA 10', color='#27ae60', linestyle='--', linewidth=2)

    # Vertical line for selected date
    ax.axvline(x=input_date, color='red', linestyle=':', linewidth=2, label='Selected Date')

    ax.set_title(f"{symbol} Price Trend & SMA Indicators")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)
    st.success("✅ Prediction complete! All lines are visible with the selected date.")
