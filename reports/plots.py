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

def plot_correlation_heatmap(returns, title="Correlation Heatmap", figsize=(10, 8)):
    
    corr = returns.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_efficient_frontier(portfolios, optimal_points=None, title="Efficient Frontier", figsize=(10, 7)): 

    plt.figure(figsize=figsize) 
    # Scatter of all random portfolios (converted to %) 
    plt.scatter( portfolios["volatility"] * 100, portfolios["return"] * 100, 
                c=portfolios["sharpe"], 
                cmap="viridis", 
                s=10, 
                alpha=0.6 ) 
    
    # Highlight optimal points 
    if optimal_points is not None: 
        for label, (vol, ret) in optimal_points.items(): 
            plt.scatter(vol * 100, ret * 100, color="red", s=80, marker="X", label=label) 
            
    plt.xlabel("Volatility (%)") 
    plt.ylabel("Expected Return (%)") 
    plt.title(title) 
    plt.colorbar(label="Sharpe Ratio") 
    plt.legend() 
    plt.tight_layout() 
    plt.show()

def plot_volatility_prediction(y_true, y_pred, title="Volatility Forecast"):
    plt.figure(figsize=(12, 6))

    plt.plot(y_true.index, y_true, label="Realized Volatility", linewidth=2)
    plt.plot(y_pred.index, y_pred, label="Predicted Volatility", linestyle="--")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
