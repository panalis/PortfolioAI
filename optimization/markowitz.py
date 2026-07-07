import numpy as np
import pandas as pd

from scipy.optimize import minimize
from risk.metrics import compute_portfolio_volatility, compute_portfolio_return

def minimize_volatility_for_target_return(mean_returns, cov_matrix, target_return):

    n_assets = len(mean_returns)

    def obj_func(weights):
        port_vol = compute_portfolio_volatility(weights, cov_matrix)
        return port_vol ** 2
    def sum_to_one_constraint(weights):
        return sum(weights) - 1
    def target_return_constraint(weights):
        return compute_portfolio_return(weights, mean_returns) - target_return

    bounds = []
    for i in range(n_assets):
        bounds.append((0, 1))

    constraints = [
        {"type": "eq", "fun": sum_to_one_constraint},
        {"type": "eq", "fun": target_return_constraint}
    ]

    x0 = np.ones(n_assets) / n_assets
    
    if compute_portfolio_return(x0, mean_returns) < target_return:
        x0 = mean_returns / mean_returns.sum()

    # Optimization
    results = minimize(
        obj_func,
        x0,
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 5000}
    )

    if not results.success:
        raise RuntimeError(results.message)

    return results.x
