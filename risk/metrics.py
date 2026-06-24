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
