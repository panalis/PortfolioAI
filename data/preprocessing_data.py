import pandas as pd
import numpy as np
import re

from data.fetch_data import fetch_prices

def get_user_portfolio():
    
    while True:

        # get user input for ticker & print prices 
        get_ticker = re.split(r"[,\s]+", input("Enter ticker(s): ").upper()) 
        price = fetch_prices(get_ticker) 
        print(price)

        # initialize weights
        if len(get_ticker) == 1:
            weights = np.array([1.0])
        else:
            # get raw weights from user
            raw_weights = re.split(r"[,\s]+", input("Enter weights: "))
            weights = np.array([float(w) for w in raw_weights])

            # check correct number of weights
            if len(weights) != len(get_ticker):
                raise ValueError("Incorrect number of weight inputs")

            # normalize weights so they sum to 1
            weights = weights / weights.sum()

        print("\nPortfolio Composition:")
        for ticker, weight in zip(get_ticker, weights):
            print(f"{ticker}: {weight:.2f}")

        confirm = input("Are these correct? (y/n): ").lower()

        if confirm == "y":
            break
        else:
            print("\nLet's try again...\n")

    return get_ticker, price, weights # change get_ticker to ticker if break


#func to calc returns, log base = e
def compute_log_returns(prices):

    if prices is None or prices.empty:
        raise ValueError("Prices error")

    # (ln(P_t / P_{t-1}))
    # P_t = daily log return
    # T_{t-1} = adjusted closing price

    log_returns = np.log(prices / prices.shift(1)) 
    log_returns = log_returns.dropna() 
    
    return log_returns
    
 