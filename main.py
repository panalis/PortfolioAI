import numpy as np
import pandas as pd

from data.preprocessing_data import get_user_portfolio, compute_log_returns
from risk.metrics import compute_annualized_volatility

ticker, price, weights = get_user_portfolio()      

# convert prices to log returns
log_returns = compute_log_returns(price)
#print(log_returns)

# calculate volatility per asset
annual_vol = compute_annualized_volatility(log_returns)
print("Annual Volatility:")
print((annual_vol * 100).apply(lambda x: f"{x:.2f}%"))
print('')