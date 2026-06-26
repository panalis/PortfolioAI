import numpy as np
import pandas as pd


def compute_annualized_volatility(returns):

    if returns is None or returns.empty:
        raise ValueError("No data for volatility")
    
    daily_vol = returns.std()
    annual_volatility = (daily_vol * np.sqrt(252))
    
    # vol of each year
    #yearly_vol = returns.groupby(returns.index.year).std() * np.sqrt(252)

    return annual_volatility

def compute_annualized_covariance(returns):
    
    if returns is None or returns.empty:
        raise ValueError("No data for covariance")
    
    daily_cov_matrix = returns.cov()
    annual_cov_matrix = daily_cov_matrix * 252

    return annual_cov_matrix

def compute_portfolio_volatility(weights, cov_matrix):

    if weights is None or len(weights) == 0:
        raise ValueError("Weights Error Missing")
    if cov_matrix is None or cov_matrix.empty:
        raise ValueError("Covariance Matrix Error Missing")

    weights = np.array(weights)
    variance = weights.T @ cov_matrix @ weights  # or do np.dot()
    portfolio_volatility = np.sqrt(variance)

    return portfolio_volatility

def compute_portfolio_return(weights, mean_returns):
    
    if weights is None or len(weights) == 0:
        raise ValueError("Weights Error Missing")
    if mean_returns is None or mean_returns.empty:
        raise ValueError("Mean Returns Error Missing")

    weights = np.array(weights)
    dot_product = np.dot(weights, mean_returns)

    return float(dot_product)

def compute_annualized_mean_returns(returns):

    if returns is None or returns.empty:
        raise ValueError("Mean Returns Error")
    
    daily_mean = returns.mean()
    annualized_mean_returns = daily_mean * 252

    return annualized_mean_returns

def compute_sharpe_ratio(returns, volatility, risk_free_rate):
    
    if returns is None or np.isnan(returns):
        raise ValueError("Sharpe Returns Error") 
    if volatility is None or np.isnan(volatility): 
        raise ValueError("Sharpe Volatility Error") 
    if risk_free_rate is None or np.isnan(risk_free_rate): 
        raise ValueError("Sharpe Risk Free Rate Error")
    
    sharpe_ratio = (returns - risk_free_rate) / volatility

    return sharpe_ratio

def compute_historical_var(returns, confidence_level):

    if returns is None or returns.empty:
        raise ValueError("Portfolio Returns for hVaR Error")
    if confidence_level is None:
        raise ValueError("hVaR Condidence Level Error")
    
    percentile = (1 - confidence_level) * 100
    quantile_value = returns.quantile(percentile / 100)

    return -quantile_value
