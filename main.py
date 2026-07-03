import numpy as np
import pandas as pd

from data.preprocessing_data import get_user_portfolio, compute_log_returns
from risk.metrics import compute_annualized_volatility, compute_annualized_covariance, compute_portfolio_volatility, compute_portfolio_return, compute_annualized_mean_returns, compute_sharpe_ratio, compute_historical_var, compute_parametric_var
from risk.monte_carlo import compute_monte_carlo_var
from optimization.simulation import simulate_random_portfolios
from reports.plots import plot_simulation, plot_historical_var_distribution, plot_parametric_var, plot_monte_carlo_var, plot_monte_carlo_paths

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

#calculate Sharpe Ratio
RISK_FREE_RATE = 0.02
sharpe_ratio = compute_sharpe_ratio(portfolio_returns, portfolio_volatility, RISK_FREE_RATE)
print("\n Portfolio Sharpe Ratio: ", sharpe_ratio)

#VaR
print("\n--------------")
print("Value at Risk (VaR)")
print("--------------")

# compute daily portfolio returns
portfolio_daily_returns = log_returns.dot(weights)

# compute VaR
hvar_95 = compute_historical_var(portfolio_daily_returns, 0.95)
hvar_99 = compute_historical_var(portfolio_daily_returns, 0.99)
pvar_95 = compute_parametric_var(portfolio_daily_returns, 0.95) 
pvar_99 = compute_parametric_var(portfolio_daily_returns, 0.99)

print(f"95% 1-Day Historical VaR: {hvar_95 * 100:.2f}%")
print(f"99% 1-Day Historical VaR: {hvar_99 * 100:.2f}%")
print(f"95% 1-Day Parametric VaR: {pvar_95 * 100:.2f}%") 
print(f"99% 1-Day Parametric VaR: {pvar_99 * 100:.2f}%")

# portfolio simulations w/ random weights
print("\n--------------")
print("Portfolio Simulation")
print("--------------")

runs = 10000
simulation = simulate_random_portfolios(runs, mean_returns, annual_cov_matrix, RISK_FREE_RATE)
print(simulation)
best = simulation.loc[simulation["Sharpe Ratio"].idxmax()]
worst = simulation.loc[simulation["Sharpe Ratio"].idxmin()]
print("\n Best Simulation : \n", best)
print("\n Worst Simulation : \n", worst)
plot_simulation(simulation)

print("\n--------------")
print("Monte Carlo VaR")
print("--------------")

mc_var_95 = compute_monte_carlo_var(mean_returns, annual_cov_matrix, weights, 0.95)
mc_var_99 = compute_monte_carlo_var(mean_returns, annual_cov_matrix, weights, 0.99)

print(f"95% 1-Day Monte Carlo VaR: {mc_var_95 * 100:.2f}%")
print(f"99% 1-Day Monte Carlo VaR: {mc_var_99 * 100:.2f}%")

simulations = hvar_95, hvar_99, pvar_95, pvar_99, mc_var_95, mc_var_99
plot_historical_var_distribution(portfolio_daily_returns, -hvar_95, 0.95)
mean = portfolio_daily_returns.mean()
std = portfolio_daily_returns.std()

print(f"Daily Mean Return: {mean * 100:.4f}%")
print(f"Daily Std Dev: {std * 100:.4f}%")

plot_parametric_var(mean, std, -pvar_95, 0.95)
