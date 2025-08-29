"""
Portfolio Analysis using yfinance, pandas, matplotlib, and seaborn
-------------------------------------------------------------------
This script downloads stock data from Yahoo Finance for selected NSE-listed
companies, performs analysis, and generates visualizations:
- Price history
- Correlation heatmap
- Daily returns & volatility
- Cumulative returns

Author: Kuntal Vasoya
Date: 2025-08-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
from datetime import date

# --------------------------------------------
#  CONFIGURATION
# --------------------------------------------
plt.style.use('fivethirtyeight')

# List of NSE stock symbols (without .NS suffix)
stocksymbols = ['NHPC', 'NTPC', 'HDFCBANK', 'BEL', 'POWERGRID', 'ITC', 'COALINDIA']
startdate = "2008-03-16"
end_date  = date.today()

print(f"You have {len(stocksymbols)} assets in your portfolio")
print("Fetching data from:", startdate, "to", end_date)

# --------------------------------------------
#  DOWNLOAD STOCK DATA
# --------------------------------------------
df = pd.concat(
    [
        yf.download(symbol + ".NS", start=startdate, end=end_date)['Close']
          .rename(columns={'Close': symbol})  # <-- fix
        for symbol in stocksymbols
    ],
    axis=1
)

# --------------------------------------------
#  VISUALIZATION 1: Price History
# --------------------------------------------
plt.figure(figsize=(15,8))
for col in df.columns:
    plt.plot(df[col], label=col)
plt.title("Portfolio Close Price History")
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")
plt.legend()
plt.show()

# --------------------------------------------
#  VISUALIZATION 2: Correlation Heatmap
# --------------------------------------------
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sb.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidth=0.5)
plt.title("Correlation Matrix of Portfolio Stocks")
plt.show()

# --------------------------------------------
#  DAILY RETURNS ANALYSIS
# --------------------------------------------
daily_returns = df.pct_change().dropna()

plt.figure(figsize=(15,8))
for col in daily_returns.columns:
    plt.plot(daily_returns[col], lw=1, label=col)
plt.title("Volatility in Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Returns")
plt.legend()
plt.show()

print("Average Daily Returns (%):")
print(daily_returns.mean() * 100)

print("\nAnnualized Volatility (%):")
print(daily_returns.std() * np.sqrt(252) * 100)

print("\nReturn per Unit of Risk (%):")
print((daily_returns.mean() / (daily_returns.std() * np.sqrt(252))) * 100)

# --------------------------------------------
#  VISUALIZATION 3: Cumulative Returns
# --------------------------------------------
cumulative_returns = (1 + daily_returns).cumprod()

plt.figure(figsize=(15,8))
for col in cumulative_returns.columns:
    plt.plot(cumulative_returns[col], lw=2, label=col)
plt.title("Cumulative Returns (Growth of ₹1 Investment)")
plt.xlabel("Date")
plt.ylabel("Growth")
plt.legend()
plt.show()