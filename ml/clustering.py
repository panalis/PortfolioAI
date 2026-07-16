import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_portfolios(data, k=3):

    if k != 3:
        raise ValueError("cluster_portfolios supports k=3 only (Conservative / Balanced / Aggressive).")

    # work on a copy so the caller's DataFrame is never mutated in place
    data = data.copy()

    # inputs
    features = ["Return", "Volatility", "Sharpe Ratio", "VaR_95"]
    X = data[features].copy()

    # data scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # fit kmeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # assign clusters to simulation
    data["Cluster"] = clusters

    # interpret clusters 
    cluster_stats = (data.groupby("Cluster")[["Volatility", "VaR_95", "Return"]].mean().sort_values("Volatility"))

    # lowest vol → conservative
    # middle → balanced
    # highest → aggressive
    interpretation = {}
    ordered_clusters = cluster_stats.index.tolist()

    interpretation[ordered_clusters[0]] = "Conservative"
    interpretation[ordered_clusters[1]] = "Balanced"
    interpretation[ordered_clusters[2]] = "Aggressive"

    return data, interpretation, scaler, kmeans

def map_risk_tolerance_to_profile(risk_score):

    if 1 <= risk_score <= 3:
        return "Conservative"
    elif 4 <= risk_score <= 7:
        return "Balanced"
    elif 8 <= risk_score <= 10:
        return "Aggressive"
    else:
        raise ValueError("Risk tolerance must be between 1 and 10.")
    
def select_portfolio(simulation_df, interpretation, risk_profile, method="sharpe"):

    # Find which cluster corresponds to the user's profile
    target_cluster = None
    for cluster_id, profile in interpretation.items():
        if profile == risk_profile:
            target_cluster = cluster_id
            break

    if target_cluster is None:
        raise ValueError("Risk profile not found in interpretation mapping.")

    # Filter portfolios in that cluster
    cluster_df = simulation_df[simulation_df["Cluster"] == target_cluster]

    # Choose best portfolio
    if method == "sharpe":
        best = cluster_df.loc[cluster_df["Sharpe Ratio"].idxmax()]
    elif method == "vol":
        best = cluster_df.loc[cluster_df["Volatility"].idxmin()]
    else:
        raise ValueError("Method must be 'sharpe' or 'vol'.")

    return best
