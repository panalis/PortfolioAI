import numpy as np
import pandas as pd

from data.fetch_data import fetch_prices

tickers = [t.strip().upper() for t in input("Enter tickers: ").split(",")]
prices = print(fetch_prices(tickers))