import numpy as np
import pandas as pd

from data.preprocessing_data import get_user_portfolio, compute_log_returns
from risk.metrics import compute_annualized_volatility, compute_annualized_covariance, compute_portfolio_volatility, compute_portfolio_return, compute_annualized_mean_returns

ticker, price, weights = get_user_portfolio()      

# convert prices to log returns
log_returns = compute_log_returns(price)
#print(log_returns)

# calculate volatility per asset
annual_vol = compute_annualized_volatility(log_returns)
print("Annual Volatility:")
print((annual_vol * 100).apply(lambda x: f"{x:.2f}%"))
print('')

# calculate annualized covariance matrix
annual_cov_matrix = compute_annualized_covariance(log_returns)
print("Annual Covariance Matrix \n", annual_cov_matrix)
print('')

# calculate portfolio volatility
portfolio_volatility = compute_portfolio_volatility(weights, annual_cov_matrix)
print(f"Annual Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
print('')

# calculate annualized mean returns
mean_returns = compute_annualized_mean_returns(log_returns) # dialy log returns
print(f"Mean Returns:\n{mean_returns * 100}")

# calculate portfolio returns
portfolio_returns = compute_portfolio_return(weights, mean_returns) #annualized mean returns
print(f"\nExpected Portfolio Return: {portfolio_returns * 100:.2f}%")
