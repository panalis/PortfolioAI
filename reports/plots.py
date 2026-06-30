import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import norm

def plot_simulation(df):
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df["Volatility"] * 100,
        df["Return"] * 100,
        c=df["Sharpe Ratio"],
        cmap="viridis",
        alpha=0.7
    )

    plt.colorbar(scatter, label="Sharpe Ratio")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Return (%)")
    plt.title("Random Portfolio Simulation")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()