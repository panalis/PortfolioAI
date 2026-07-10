import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def build_volatility_features(returns, window = 20):

    if returns is None or returns.empty:
        raise ValueError("Returns for Volatility ML are missing!")
    
    returns = returns.astype(float)
    abs_daily_returns = returns.abs()

    vol_5 = returns.rolling(window = 5).std()
    vol_10 = returns.rolling(window = 10).std()
    vol_20 = returns.rolling(window).std()
    
    target_volatility = vol_20.shift(-1)

    df = pd.DataFrame({
        "r_t": returns,
        "abs_r_t": abs_daily_returns,
        "vol_5": vol_5,
        "vol_10": vol_10,
        "vol_20": vol_20,
        "target": target_volatility
    })
    
    df = df.dropna()

    # split data, x = feutures, y = target
    X = df[["r_t", "abs_r_t", "vol_5", "vol_10", "vol_20"]]
    y = df['target']

    return X, y

def train_volatility_model(X, y, model_type):

    split = int(len(X) * 0.8) # 80/20 split

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError("Unknown model type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    

    return model, X_test, y_test, y_pred

def evaluate_volatility_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    return mae, mse