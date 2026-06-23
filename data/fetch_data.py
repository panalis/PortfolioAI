import yfinance as yf
import pandas as pd
import datetime as dt

# func to get and sort data for adjusted close
def fetch_prices(tickers):
    today = dt.datetime.now()
    data = yf.download(tickers, start="2015-01-01", end=today, auto_adjust=False)

    if data is None or data.empty:
        raise ValueError("Error getting data")

    adj_close = data["Adj Close"]

    # Ensure adj_close is always a DataFrame
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    # Reorder columns to match user input
    adj_close = adj_close[tickers]

    return adj_close
