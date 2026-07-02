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
 
def plot_historical_var_distribution(returns, var, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.75, color="steelblue", edgecolor="black")

    plt.axvline(var, color="red", linestyle="--",
                label=f"{int(confidence*100)}% VaR: {var*100:.2f}%")

    plt.title("Return Distribution with Historical VaR Threshold")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_parametric_var(mean, std, var, confidence=0.95):
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = norm.pdf(x, mean, std)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Normal Distribution", color="black")

    plt.axvline(var, color="red", linestyle="--",
                label=f"{int(confidence*100)}% VaR: {var*100:.2f}%")

    plt.fill_between(x, 0, y, where=(x <= var), color="red", alpha=0.3)

    plt.title("Parametric VaR Under Normality Assumption")
    plt.xlabel("Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_monte_carlo_var(simulated_returns, var, confidence=0.95):
    plt.figure(figsize=(10, 6))
    plt.hist(simulated_returns, bins=50, alpha=0.75,
             color="steelblue", edgecolor="black")

    plt.axvline(var, color="red", linestyle="--",
                label=f"{int(confidence*100)}% VaR: {var*100:.2f}%")

    plt.title("Monte Carlo Simulated Return Distribution")
    plt.xlabel("Simulated Returns")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_monte_carlo_paths(mean_returns, cov_matrix, weights, days, simulations):
    daily_mean = mean_returns / 252
    daily_cov = cov_matrix / 252

    portfolio_paths = []

    for _ in range(simulations):
        # Simulate daily returns
        simulated_returns = np.random.multivariate_normal(daily_mean, daily_cov, size=days)
        portfolio_returns = simulated_returns @ weights

        # Convert to cumulative value (starting at 100)
        cumulative = 100 * np.cumprod(1 + portfolio_returns)
        portfolio_paths.append(cumulative)

    # Plot
    plt.figure(figsize=(12, 6))
    for path in portfolio_paths:
        plt.plot(path, alpha=0.3)

    plt.title("Monte Carlo Simulated Portfolio Paths")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return np.array(portfolio_paths)