import numpy as np
import matplotlib.pyplot as plt

# PARAMETER
n_assets = 3 # number of assets

# generate random expected returns and variances for n assets
#np.random.seed(42)  # set seed if want reproducibility
expected_returns = np.random.uniform(0.05, 0.15, n_assets)
variances = np.random.uniform(0.02, 0.1, n_assets)

# choose risk-free rate
risk_free_rate = 0.03

# calculates portfolio return and variance
def portfolio_stats(weights, expected_returns, variances):
    portfolio_return = np.sum(weights * expected_returns)
    portfolio_variance = np.sum(weights**2 * variances)
    return portfolio_return, np.sqrt(portfolio_variance)  # returns: return and standard deviation

# generates portfolios with controlled weight increments/step sizes
def generate_portfolios(n_assets, n_portfolios, step_size):
    portfolio_returns = []
    portfolio_risks = []
    portfolio_weights = []
    
    for _ in range(n_portfolios):
        # random weights summing to 1 with controlled increments
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)  # normalize so they sum to 1
        
        port_return, port_risk = portfolio_stats(weights, expected_returns, variances)
        portfolio_returns.append(port_return)
        portfolio_risks.append(port_risk)
        portfolio_weights.append(weights)

    return np.array(portfolio_risks), np.array(portfolio_returns), np.array(portfolio_weights)

# calculates efficient frontier!
def calculate_efficient_frontier(risks, returns):
    # sort by risk and find the max return for each risk level (efficient frontier)
    sorted_indices = np.argsort(risks)
    sorted_risks = risks[sorted_indices]
    sorted_returns = returns[sorted_indices]

    # track max return for each risk level
    max_return = -np.inf
    efficient_risks = []
    efficient_returns = []
    
    for risk, ret in zip(sorted_risks, sorted_returns):
        if ret > max_return:
            max_return = ret
            efficient_risks.append(risk)
            efficient_returns.append(ret)

    return np.array(efficient_risks), np.array(efficient_returns)

# PARAMETERS
n_portfolios = 10000 # number of simulated portfolios
step_size = 0.01  # randomness of portfolio weights

# generate portfolios
portfolio_risks, portfolio_returns, portfolio_weights = generate_portfolios(n_assets, n_portfolios, step_size)

# get efficient frontier
efficient_risks, efficient_returns = calculate_efficient_frontier(portfolio_risks, portfolio_returns)

# calculate Sharpe ratios for all portfolios
sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_risks

# find portfolio with the max Sharpe ratio (tangency portfolio)
max_sharpe_idx = np.argmax(sharpe_ratios)
tangent_risk = portfolio_risks[max_sharpe_idx]
tangent_return = portfolio_returns[max_sharpe_idx]
tangent_weights = portfolio_weights[max_sharpe_idx]

# print allocations at the tangency portfolio point
print("Allocations at the tangent portfolio (max Sharpe ratio):")
for i, weight in enumerate(tangent_weights):
    print(f"Asset {i+1}: {weight:.2%}")

# lot portfolios, efficient frontier, and CML
plt.figure(figsize=(10, 6))

# scatterplot of portfolios
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')

# plot efficient frontier
plt.plot(efficient_risks, efficient_returns, color='red', linewidth=2, label='Efficient Frontier')

# plot tangent portfolio (maximum Sharpe ratio)
plt.scatter(tangent_risk, tangent_return, color='orange', marker='*', s=100, label='Tangent Portfolio')

# Capital Market Line (CML)
cml_x = np.linspace(0, max(portfolio_risks), 100)
cml_y = risk_free_rate + (tangent_return - risk_free_rate) / tangent_risk * cml_x
plt.plot(cml_x, cml_y, color='blue', linestyle='--', label='Capital Market Line (CML)')

plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier and Capital Market Line')
plt.grid(True)
plt.legend()

plt.show()


# stats for each asset (expected return and variance)
print("Asset Stats:")
for i in range(n_assets):
    print(f"Asset {i+1}:")
    print(f"  Expected Return: {expected_returns[i]:.2%}")
    print(f"  Variance: {variances[i]:.4f}")
    print(f"  Standard Deviation: {np.sqrt(variances[i]):.2%}")
    
