# ===============================
# Real-Time Stock Price Prediction App
# ===============================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Stock Prediction App", layout="centered")
st.title("📈 Real-Time Stock Price Prediction")

# -------------------------------
# User Input
# -------------------------------
symbol = st.text_input("Enter Stock Symbol (AAPL, MSFT, TSLA)", "AAPL")

# -------------------------------
# Button
# -------------------------------
if st.button("Predict Next Day Price"):

    # 1️⃣ Fetch live stock data (NO CSV)
    df = yf.download(symbol, period="5y")

    if df.empty:
        st.error("❌ Invalid stock symbol")
    else:
        # 2️⃣ Use Close price only
        df = df[['Close']].dropna()

        # 3️⃣ Feature engineering (Day index)
        df['Day'] = np.arange(len(df))

        X = df[['Day']]
        y = df['Close']

        # 4️⃣ Train ML model
        model = LinearRegression()
        model.fit(X, y)

        # 5️⃣ Predict next day price
        next_day = np.array([[len(df)]])
        prediction = model.predict(next_day)

        # ✅ SAFE conversion (NO ERROR)
        predicted_price = prediction.item()

        # 6️⃣ Show result
        st.subheader(f"✅ Predicted Next Close Price: ${predicted_price:.2f}")

        # 7️⃣ Show chart
        st.line_chart(df['Close'])
