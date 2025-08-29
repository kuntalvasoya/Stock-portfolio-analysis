# 📈 Portfolio Analysis with Python

This project analyzes a sample stock portfolio using **Yahoo Finance data** and Python libraries like `pandas`, `matplotlib`, and `seaborn`.

## 🚀 Features
- Fetches stock price history from Yahoo Finance
- Plots:
  - Close Price History
  - Correlation Heatmap
  - Daily Returns & Volatility
  - Cumulative Returns
- Calculates:
  - Average Daily Returns
  - Annualized Volatility
  - Return per Unit of Risk (Sharpe-like ratio)

## 🛠️ Tech Stack
- Python 3
- [yfinance](https://pypi.org/project/yfinance/)
- pandas, numpy
- matplotlib, seaborn

## 📊 Example Output
- Portfolio Price History  
- Correlation Matrix  
- Daily Returns Volatility  
- Cumulative Returns  

(All plots are saved as `.png` files automatically)

## ⚡ Usage
```bash
# Clone repo
git clone https://github.com/yourusername/portfolio-analysis.git
cd portfolio-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python portfolio_analysis.py
