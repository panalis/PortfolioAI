import numpy as np
import pandas as pd

from scipy.optimize import minimize
from risk.metrics import compute_portfolio_volatility, compute_portfolio_return

def minimize_volatility_for_target_return(mean_returns, cov_matrix, target_return):