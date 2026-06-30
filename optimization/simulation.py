# portfolio monte carlo simulation
# randomness = weights
# computes resulting risk and return

import numpy as np
import pandas as pd

from risk.metrics import compute_portfolio_return, compute_portfolio_volatility, compute_sharpe_ratio

def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):

    n_assets = len(mean_returns)
    results = []
    z_95 = 1.65

    for i in range(num_portfolios):

        weights = np.random.rand(n_assets) # random numbers
        weights /= np.sum(weights) # normalization, equal to 1

        port_returns = compute_portfolio_return(weights, mean_returns)
        volatility = compute_portfolio_volatility(weights, cov_matrix)
        sharpe_ratio = compute_sharpe_ratio(port_returns, volatility, risk_free_rate)
        var_95 = -(port_returns - z_95 * volatility)

        
        row = { 
            "Return": port_returns, 
            "Volatility": volatility, 
            "Sharpe Ratio": sharpe_ratio,
            "VaR_95": var_95
        }

        for t, w in zip(mean_returns.index, weights):
            row[f"w_{t}"] = w
        
        results.append(row)

    return pd.DataFrame(results)