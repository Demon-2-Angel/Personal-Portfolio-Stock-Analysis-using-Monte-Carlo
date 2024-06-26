# Monte Carlo Simulation for Stock Portfolio

This code performs a Monte Carlo simulation to estimate the future value of a stock portfolio based on historical stock data.

## 1. Importing Libraries

```
python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import yfinance as yf
```

- **pandas**: For data manipulation and analysis.
- **matplotlib.pyplot**: For plotting graphs.
- **numpy**: For numerical computations.
- **datetime**: For date manipulations.
- **yfinance**: For downloading stock data from Yahoo Finance.

## 2. Defining the `get_data` Function


```
python
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix
```

- Downloads historical stock data for the given stocks and date range using Yahoo Finance.
- Calculates daily returns for each stock.
- Computes the mean returns and the covariance matrix of the returns, which are essential for the Monte Carlo simulation.

## 3. Setting Up the Stocks and Date Range

```
python
stockList = ['TSLA', 'GOOG', 'META']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
```

## 3. Setting Up the Stocks and Date Range

```
python
stockList = ['TSLA', 'GOOG', 'META']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
```

- Defines the list of stock symbols.
- Sets the date range for the past 300 days from the current date.

## 4. Generating Random Portfolio Weights

```
python
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)
```

- Generates random weights for each stock in the portfolio.
- Normalizes the weights so that their sum equals 1.

## 5. Monte Carlo Simulation Parameters

```
python
mc_sims = 100  # Number of simulations
T = 100        # Number of time steps (days)
```

- **mc_sims**: The number of Monte Carlo simulations to run.
- **T**: The number of days to simulate.

## 6. Preparing the Mean Returns Matrix

```
python
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
```

- Creates a matrix filled with mean returns for each stock, repeated for `T` days.

## 7. Initializing Portfolio Simulations

```
python
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000
```

- Initializes a matrix to store the simulated portfolio values.
- Sets the initial portfolio value to 10,000.

## 8. Running the Monte Carlo Simulations

```
python
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio
```

- For each simulation, it generates random values from a normal distribution.
- Uses the Cholesky decomposition of the covariance matrix to generate correlated random returns.
- Computes daily returns for the portfolio by multiplying the weights by the daily returns.
- Computes the cumulative product of these returns to simulate the portfolio value over time.

## 9. Plotting the Results

```
python
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value (Rs.)')
plt.xlabel('Days')
plt.title('Monte Carlo Simulation of a Stock Portfolio')
plt.show()
```

- Plots the simulated portfolio values over time.
- Labels the axes and gives the plot a title.

## Summary

- The code downloads historical stock data.
- It calculates mean returns and the covariance matrix.
- It generates random portfolio weights and runs a Monte Carlo simulation to estimate the future value of the portfolio.
- It plots the results of the simulations to visualize the potential future performance of the portfolio.

## Additional Analysis

### Risk Assessment

#### Value at Risk (VaR)

```
python
portfolio_end_values = portfolio_sims[-1, :]
var_95 = np.percentile(portfolio_end_values, 5)  # 5th percentile for 95% confidence level
print(f"Value at Risk (95% confidence level): {initialPortfolio - var_95:.2f} Rs.")
```

#### Expected Return

```
python
expected_return = np.mean(portfolio_end_values)
print(f"Expected Portfolio Value: {expected_return:.2f} Rs.")
```

#### Probability of Loss

```
python
loss_probability = np.mean(portfolio_end_values < initialPortfolio)
print(f"Probability of Loss: {loss_probability * 100:.2f}%")
```

#### Confidence Interval

```
python
ci_lower = np.percentile(portfolio_end_values, 2.5)
ci_upper = np.percentile(portfolio_end_values, 97.5)
print(f"95% Confidence Interval: {ci_lower:.2f} Rs. to {ci_upper:.2f} Rs.")
```

### Portfolio Optimization

```python
from scipy.optimize import minimize

def portfolio_performance(weights, meanReturns, covMatrix):
    portfolio_return = np.sum(meanReturns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev

def negative_sharpe_ratio(weights, meanReturns, covMatrix, risk_free_rate=0):
    p_returns, p_stddev = portfolio_performance(weights, meanReturns, covMatrix)
    return -(p_returns - risk_free_rate) / p_stddev

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for x in range(len(meanReturns)))
result = minimize(negative_sharpe_ratio, len(meanReturns) * [1. / len(meanReturns)], args=(meanReturns, covMatrix), method='SLSQP', bounds=bounds, constraints=constraints)

optimized_weights = result.x
print(f"Optimized Portfolio Weights: {optimized_weights}")
```

### Histogram of Final Portfolio Values

```
python
plt.hist(portfolio_end_values, bins=50, alpha=0.75)
plt.axvline(initialPortfolio, color='r', linestyle='dashed', linewidth=2)
plt.xlabel('Final Portfolio Value (Rs.)')
plt.ylabel('Frequency')
plt.title('Distribution of Final Portfolio Values')
plt.show()
```

