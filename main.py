import numpy as np
import pandas as pd

from data.preprocessing_data import get_user_portfolio, compute_log_returns

ticker, price, weights = get_user_portfolio()      

# convert prices to log returns
log_returns = compute_log_returns(price)
#print(log_returns)