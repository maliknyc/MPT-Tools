import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# prompt user for gamma
gamma = float(input("Enter your γ (γ >= 0): "))

# read in expected returns and variances from CSV file
# assume CSV file has columns 'Expected Return', 'Variance'
asset_file = "test_assets_1.csv"
asset_data = pd.read_csv(asset_file)

expected_returns = asset_data['Expected Return'].values
variances = asset_data['Variance'].values
n_assets = len(expected_returns)

# ASSUME ASSETS ARE UNCORRELATED
# generate random portfolios
n_portfolios = 100000

portfolio_weights = np.random.dirichlet(np.ones(n_assets), n_portfolios)
portfolio_returns = portfolio_weights.dot(expected_returns)

# portfolio variance is sum of individual variances times weights squared (since no covariance)
portfolio_variances = np.sum((portfolio_weights ** 2) * variances, axis=1)
portfolio_std_devs = np.sqrt(portfolio_variances)

# calculate expected utility for each portfolio
if gamma == 1:
    expected_utilities = portfolio_returns - 0.5 * portfolio_variances
else:
    expected_utilities = (portfolio_returns - 0.5 * gamma * portfolio_variances) / (1 - gamma)

# since (1 - gamma) is negative when gamma > 1, we have to maximize expected utilities accordingly
if gamma > 1:
    # need to minimize expected_utilities since (1 - gamma) is negative
    max_utility_idx = np.argmin(expected_utilities)
else:
    max_utility_idx = np.argmax(expected_utilities)

optimal_weights = portfolio_weights[max_utility_idx]
optimal_return = portfolio_returns[max_utility_idx]
optimal_std_dev = portfolio_std_devs[max_utility_idx]

# print out optimal allocations based on utility
print("\nOptimal Portfolio based on your utility function:")
for i, weight in enumerate(optimal_weights):
    print(f"Asset {i+1}: {weight:.2%}")

print(f"Expected Return: {optimal_return:.2%}")
print(f"Portfolio Risk (Standard Deviation): {optimal_std_dev:.2%}")

# plot the portfolios
plt.figure(figsize=(10, 6))

# scatterplot of portfolios
plt.scatter(portfolio_std_devs, portfolio_returns, c='lightblue', alpha=0.5, label='Portfolios')

# EFFICIENT FRONTIER: take the portfolios with the highest return for a given risk
# sort portfolios by risk
sorted_indices = np.argsort(portfolio_std_devs)
sorted_risks = portfolio_std_devs[sorted_indices]
sorted_returns = portfolio_returns[sorted_indices]

# find the efficient frontier
efficient_risks = []
efficient_returns = []
max_return = -np.inf

for risk, ret in zip(sorted_risks, sorted_returns):
    if ret > max_return:
        max_return = ret
        efficient_risks.append(risk)
        efficient_returns.append(ret)

plt.plot(efficient_risks, efficient_returns, color='red', linewidth=2, label='Efficient Frontier')

# mark the optimal portfolio
plt.scatter(optimal_std_dev, optimal_return, color='green', marker='*', s=200, label='Optimal Portfolio')

# let user input risk-free rate
risk_free_rate = float(input("\nEnter the risk-free rate (e.g., 0.02 for 2%): "))

# calculate Sharpe ratios
sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_std_devs

# find the tangency portfolio
max_sharpe_idx = np.argmax(sharpe_ratios)
tangent_weights = portfolio_weights[max_sharpe_idx]
tangent_return = portfolio_returns[max_sharpe_idx]
tangent_std_dev = portfolio_std_devs[max_sharpe_idx]

# print out allocations for the tangency portfolio
print("\nTangency Portfolio (Maximum Sharpe Ratio):")
for i, weight in enumerate(tangent_weights):
    print(f"Asset {i+1}: {weight:.2%}")

print(f"Expected Return: {tangent_return:.2%}")
print(f"Portfolio Risk (Standard Deviation): {tangent_std_dev:.2%}")

# plot tangency portfolio
plt.scatter(tangent_std_dev, tangent_return, color='orange', marker='D', s=100, label='Tangency Portfolio')

# plot CML
cml_x = np.linspace(0, max(portfolio_std_devs.max(), tangent_std_dev * 1.5), 100)
cml_y = risk_free_rate + (tangent_return - risk_free_rate) / tangent_std_dev * cml_x
plt.plot(cml_x, cml_y, color='blue', linestyle='--', label='Capital Market Line (CML)')

plt.xlabel('Portfolio Risk (Standard Deviation)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier and Capital Market Line')
plt.legend()
plt.grid(True)

plt.show()
