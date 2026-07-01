import numpy as np

def compute_monte_carlo_var(mean_returns, cov_matrix, weights, confidence_level=0.95, simulations=10000, return_simulations=False):
    # Convert annual to daily
    daily_mean = mean_returns / 252
    daily_cov = cov_matrix / 252

    # Simulate multivariate normal returns
    simulated_returns = np.random.multivariate_normal(
        daily_mean, 
        daily_cov, 
        size=simulations
    )

    # Convert to portfolio returns
    portfolio_returns = simulated_returns @ weights

    # Compute percentile
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(portfolio_returns, percentile)

    if return_simulations:
        return var, portfolio_returns

    return var
