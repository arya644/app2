import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📈 Real-Time Stock Price Prediction App")

# User input
symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT)", "AAPL")

if st.button("Predict"):
    # 1. Fetch live data (NO CSV)
    df = yf.download(symbol, period="5y")

    if df.empty:
        st.error("Invalid stock symbol")
    else:
        # 2. Use Close price
        df = df[['Close']].dropna()

        # 3. Feature engineering
        df['Day'] = np.arange(len(df))

        X = df[['Day']]
        y = df['Close']

        # 4. Train model
        model = LinearRegression()
        model.fit(X, y)

        # 5. Predict next day price
        next_day = [[len(df)]]
        prediction = model.predict(next_day)

        # 6. Output
        st.subheader(f"Predicted Next Close Price: ${prediction[0]:.2f}")
        st.line_chart(df['Close'])
