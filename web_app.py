import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import datetime as dt
import re


from data.fetch_data import fetch_prices
from data.preprocessing_data import compute_log_returns
from risk.metrics import compute_annualized_volatility, compute_annualized_covariance, compute_portfolio_volatility, compute_portfolio_return, compute_annualized_mean_returns, compute_sharpe_ratio, compute_historical_var, compute_parametric_var
from risk.monte_carlo import compute_monte_carlo_var
from optimization.markowitz import minimize_volatility_for_target_return
from optimization.simulation import simulate_random_portfolios
from ml.clustering import cluster_portfolios, map_risk_tolerance_to_profile, select_portfolio
from ml.volatility_model import build_volatility_features, train_volatility_model, evaluate_volatility_model