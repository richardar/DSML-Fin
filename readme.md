## Stock Analysis Dashboard

A Streamlit-based web application for in-depth stock analysis, forecasting, and visualization. Powered by Yahoo Finance data (`yfinance`), technical indicators (`ta`), statistical models (ARIMA/SARIMA), and machine learning models (Random Forest, XGBoost).

---

### 🛠️ Features

* **Real-time Data Refresh**: Fetch up-to-date stock data on demand.
* **Custom Date Range**: Select any historical date range for analysis.
* **Price Trends & Metrics**: View market cap, 52-week high/low, and daily closing prices.
* **Heikin-Ashi Candlesticks**: Enhanced candlestick chart for trend analysis.
* **Technical Indicators**:

  * Bollinger Bands
  * RSI (Relative Strength Index)
  * Volatility (10-day rolling STD of returns)
  * MACD (Moving Average Convergence Divergence)
  * OBV (On-Balance Volume)
  * ADX (Average Directional Index)
* **Forecasting Models**:

  * ARIMA and SARIMA for time-series forecasting
  * MSE & MAE metrics for model performance
* **Machine Learning Forecasts**:

  * Random Forest Regression
  * XGBoost Regression
  * Visual comparisons of actual vs. predicted values
  * Next-day price predictions

---

### 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/stock-analysis-dashboard.git
   cd stock-analysis-dashboard
   ```

2. **Create & activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

### ⚙️ Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run scraper.py
   ```

2. **Interact via sidebar**:

   * Enter a **Ticker Symbol** (default: `AAPL`).
   * Select **Start** and **End** dates.
   * Click **Refresh Real-time Data** to reload.

3. **Explore**:

   * View overview metrics at the top.
   * Navigate through various charts and forecast sections.

---

### 📈 Output Screenshots

![Dashboard Overview](docs/screenshots/overview.png)
![Technical Indicators](docs/screenshots/indicators.png)

---

### ⚠️ Disclaimer

This project is for **educational purposes only**. Do **not** use this dashboard for live trading or financial advice. Data is fetched from Yahoo Finance (`yfinance`) and may be subject to terms of service limitations.

---

### 🙌 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repo and submit a pull request.

---

### 🧑‍💻 Author

* **Your Name** - [GitHub](https://github.com/yourusername)

---

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
