import numpy as np
import pandas as pd

from data.preprocessing_data import get_user_portfolio, compute_log_returns
from risk.metrics import compute_annualized_volatility, compute_annualized_covariance, compute_portfolio_volatility, compute_portfolio_return, compute_annualized_mean_returns, compute_sharpe_ratio, compute_historical_var, compute_parametric_var
from risk.monte_carlo import compute_monte_carlo_var
from optimization.simulation import simulate_random_portfolios
from reports.plots import plot_simulation, plot_historical_var_distribution, plot_parametric_var, plot_monte_carlo_var, plot_monte_carlo_paths, plot_correlation_heatmap, plot_efficient_frontier, plot_volatility_prediction
from optimization.markowitz import minimize_volatility_for_target_return
from ml.volatility_model import build_volatility_features, train_volatility_model, evaluate_volatility_model, persistence_baseline
from ml.clustering import cluster_portfolios, map_risk_tolerance_to_profile, select_portfolio

# Reproducibility: seeds the global NumPy RNG used by every randomised routine
# (random portfolio simulation, Monte Carlo VaR, Monte Carlo paths).
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ticker, price, weights = get_user_portfolio()      

# convert prices to log returns
log_returns = compute_log_returns(price)
#print(log_returns)

# calculate volatility per asset
annual_vol = compute_annualized_volatility(log_returns)
print("Annual Volatility:")
print((annual_vol * 100).apply(lambda x: f"{x:.2f}%"))
print('')

# calculate annualized covariance matrix
annual_cov_matrix = compute_annualized_covariance(log_returns)
print("Annual Covariance Matrix \n", annual_cov_matrix)
print('')

# calculate portfolio volatility
portfolio_volatility = compute_portfolio_volatility(weights, annual_cov_matrix)
print(f"Annual Portfolio Volatility: {portfolio_volatility * 100:.2f}%")
print('')

# calculate annualized mean returns
mean_returns = compute_annualized_mean_returns(log_returns) # dialy log returns
print(f"Mean Returns:\n{mean_returns * 100}")

# calculate portfolio returns
portfolio_returns = compute_portfolio_return(weights, mean_returns) #annualized mean returns
print(f"\nExpected Portfolio Return: {portfolio_returns * 100:.2f}%")

#calculate Sharpe Ratio
RISK_FREE_RATE = 0.02
sharpe_ratio = compute_sharpe_ratio(portfolio_returns, portfolio_volatility, RISK_FREE_RATE)
print("\n Portfolio Sharpe Ratio: ", sharpe_ratio)

#VaR
print("\n--------------")
print("Value at Risk (VaR)")
print("--------------")

# compute daily portfolio returns
portfolio_daily_returns = log_returns.dot(weights)

# compute VaR
hvar_95 = compute_historical_var(portfolio_daily_returns, 0.95)
hvar_99 = compute_historical_var(portfolio_daily_returns, 0.99)
pvar_95 = compute_parametric_var(portfolio_daily_returns, 0.95) 
pvar_99 = compute_parametric_var(portfolio_daily_returns, 0.99)

print(f"95% 1-Day Historical VaR: {hvar_95 * 100:.2f}%")
print(f"99% 1-Day Historical VaR: {hvar_99 * 100:.2f}%")
print(f"95% 1-Day Parametric VaR: {pvar_95 * 100:.2f}%") 
print(f"99% 1-Day Parametric VaR: {pvar_99 * 100:.2f}%")

# portfolio simulations w/ random weights
print("\n--------------")
print("Portfolio Simulation")
print("--------------")

runs = 10000
simulation = simulate_random_portfolios(runs, mean_returns, annual_cov_matrix, RISK_FREE_RATE)
print(simulation)
best = simulation.loc[simulation["Sharpe Ratio"].idxmax()]
worst = simulation.loc[simulation["Sharpe Ratio"].idxmin()]
print("\n Best Simulation : \n", best)
print("\n Worst Simulation : \n", worst)
plot_simulation(simulation)

print("\n--------------")
print("Monte Carlo VaR")
print("--------------")

# single Monte Carlo draw reused for both the printed VaR and the plotted distribution
mc_var_95, mc_sim_returns = compute_monte_carlo_var(
    mean_returns,
    annual_cov_matrix,
    weights,
    0.95,
    return_simulations=True
)
mc_var_99 = compute_monte_carlo_var(mean_returns, annual_cov_matrix, weights, 0.99)

print(f"95% 1-Day Monte Carlo VaR: {mc_var_95 * 100:.2f}%")
print(f"99% 1-Day Monte Carlo VaR: {mc_var_99 * 100:.2f}%")

plot_historical_var_distribution(portfolio_daily_returns, -hvar_95, 0.95)
mean = portfolio_daily_returns.mean()
std = portfolio_daily_returns.std()

print(f"Daily Mean Return: {mean * 100:.4f}%")
print(f"Daily Std Dev: {std * 100:.4f}%")

plot_parametric_var(mean, std, -pvar_95, 0.95)

plot_monte_carlo_var(mc_sim_returns, -mc_var_95, 0.95)

print("\n--------------")
print("Correlation Heatmap")
print("--------------")

plot_correlation_heatmap(log_returns)

#MARKOQITZ
print("\n--------------")
print("Markowitz Optimization Test")
print("--------------")

target = float(mean_returns.mean())

print(f"Using target return: {target:.4f}")

if target < mean_returns.min() or target > mean_returns.max():
    raise ValueError("Target return is outside the feasible range for long-only portfolios.")

optimal_weights = minimize_volatility_for_target_return(
    mean_returns,
    annual_cov_matrix,
    target
)

print("\nOptimal Weights for Target Return:")
for t, w in zip(ticker, optimal_weights):
    print(f"{t}: {w:.4f}")

opt_vol = compute_portfolio_volatility(optimal_weights, annual_cov_matrix)
opt_ret = compute_portfolio_return(optimal_weights, mean_returns)

print(f"\nPortfolio Return: {opt_ret * 100:.2f}%")
print(f"Portfolio Volatility: {opt_vol * 100:.2f}%")

print("\n--------------")
print("Monte Carlo Paths")
print("--------------")

paths = plot_monte_carlo_paths(mean_returns, annual_cov_matrix, weights, days=300, simulations=5000)
final_values = paths[:, -1]          # last value of each path
best_index = np.argmax(final_values) # index of best path
best_path = paths[best_index]        # the actual path
print("Best final value:", final_values[best_index])
print(f"Min-Vol Portfolio Return: {opt_ret * 100:.2f}%") 
print(f"Min-Vol Portfolio Volatility: {opt_vol * 100:.2f}%") 
print(f"Max-Sharpe Portfolio Return: {best['Return'] * 100:.2f}%") 
print(f"Max-Sharpe Portfolio Volatility: {best['Volatility'] * 100:.2f}%")

print("\n--------------")
print("Efficient Frontier")
print("--------------")

# Convert your simulation output to the expected column names
portfolios_df = simulation.rename(columns={
    "Volatility": "volatility",
    "Return": "return",
    "Sharpe Ratio": "sharpe"
})

# Optimal points: Markowitz + best Sharpe from simulation
optimal_points = {
    "min_vol": (opt_vol, opt_ret),
    "max_sharpe": (best["Volatility"], best["Return"])
}

plot_efficient_frontier(
    portfolios_df,
    optimal_points=optimal_points
)


print("\n-----------------------------") 
print(" VOLATILITY FORECASTING (ML) ") 
print("-----------------------------\n") 

for t in log_returns.columns: 
    print(f"\n----- {t} -----") 

    # build features for this ticker 
    X, y = build_volatility_features(log_returns[t])

    # train Linear Regression 
    model_lin, X_test, y_test, y_pred_lin = train_volatility_model( X, y, model_type="linear" ) 
    mae_lin, mse_lin = evaluate_volatility_model(y_test, y_pred_lin) 

    # train Random Forest 
    model_rf, X_test, y_test, y_pred_rf = train_volatility_model( X, y, model_type="rf" ) 
    mae_rf, mse_rf = evaluate_volatility_model(y_test, y_pred_rf) 

    # persistence baseline: predict next-day vol = today's 20-day vol
    y_pred_persist = persistence_baseline(X_test)
    mae_persist, mse_persist = evaluate_volatility_model(y_test, y_pred_persist)

    # results 
    print(f"Persistence baseline (no-change):") 
    print(f" MAE: {mae_persist:.6f}") 
    print(f" MSE: {mse_persist:.6f}") 
    print(f"Linear Regression:") 
    print(f" MAE: {mae_lin:.6f}  ({'beats' if mae_lin < mae_persist else 'does NOT beat'} baseline)") 
    print(f" MSE: {mse_lin:.6f}") 
    print(f"Random Forest:") 
    print(f" MAE: {mae_rf:.6f}  ({'beats' if mae_rf < mae_persist else 'does NOT beat'} baseline)") 
    print(f" MSE: {mse_rf:.6f}")

    plot_volatility_prediction(
        y_test,
        pd.Series(y_pred_lin, index=y_test.index),
        title=f"{t} — Linear Regression Volatility Forecast"
        
    )
    
    plot_volatility_prediction(
        y_test,
        pd.Series(y_pred_rf, index=y_test.index),
        title=f"{t} — Random Forest Volatility Forecast"
    )
 
print("\n--------------------------------") 
print(" Portfolio Selection K-Means (ML)") 
print("--------------------------------\n") 

#  clustering
simulation_df, interpretation, scaler, kmeans = cluster_portfolios(simulation)

# ask user
print("\n 1-3 = Conservative - Low Risk \n 4-7 = Balanced - Medium Risk \n 8-10 = Aggressive - High Risk")

risk_score = int(input("Enter risk tolerance (1-10): "))
risk_profile = map_risk_tolerance_to_profile(risk_score)

# selection
best_portfolio = select_portfolio(simulation_df, interpretation, risk_profile)

print("\nRecommended Portfolio Based on Your Risk Profile:")
print(best_portfolio)