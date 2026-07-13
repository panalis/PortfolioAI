import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_portfolios(data, k=3):

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