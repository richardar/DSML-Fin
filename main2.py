import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objs as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title=" Stock Analysis Dashboard", layout="wide")

# Sidebar Inputs
st.sidebar.header("Data Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-20"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-20"))

if st.sidebar.button("Refresh Real-time Data"):
    st.rerun()

# Fetch stock info
try:
    stock_info = yf.Ticker(ticker).info
    company_name = stock_info.get("shortName", ticker)
except:
    company_name = ticker

st.title(f"📈 {company_name} ({ticker}) Stock Analysis Dashboard")
st.markdown("Explore **price trends**, **technical indicators**, **machine learning forecasts**, and **ARIMA models**.")

# Fetch and preprocess data
df = yf.download(ticker, start=start_date, end=end_date, progress=False)
df = df.dropna()
df.index = pd.to_datetime(df.index)

# Technical Indicators
df['Daily_returns'] = df['Close'].pct_change()
df['Log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['Volatility_10'] = df['Daily_returns'].rolling(window=10).std()
df['Cumulative_returns'] = (1 + df['Daily_returns']).cumprod()
df['MA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'].squeeze()).rsi()
macd = ta.trend.MACD(close=df['Close'].squeeze())
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()
boll = ta.volatility.BollingerBands(close=df['Close'].squeeze())
df['BB_High'] = boll.bollinger_hband()
df['BB_Low'] = boll.bollinger_lband()
df['BB_Mid'] = boll.bollinger_mavg()
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()
df['ADX'] = ta.trend.ADXIndicator(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()).adx()
df.dropna(inplace=True)

st.header(" Overview and Price Trends")
col1, col2, col3 = st.columns(3)
col1.metric("Market Cap", f"${stock_info.get('marketCap', 0):,}")
col2.metric("52 Week High", f"${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")
col3.metric("52 Week Low", f"${stock_info.get('fiftyTwoWeekLow', 0):.2f}")

# Heikin-Ashi Chart
st.subheader(" Heikin-Ashi Chart")

df_ha = df.copy()
df_ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
df_ha['HA_Open'] = ((df['Open'] + df['Close']) / 2).shift(1)
df_ha['HA_Open'].iloc[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
df_ha['HA_High'] = df_ha[['High', 'HA_Open', 'HA_Close']].max(axis=1)
df_ha['HA_Low'] = df_ha[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

fig_ha = go.Figure(data=[go.Candlestick(
    x=df_ha.index,
    open=df_ha['HA_Open'],
    high=df_ha['HA_High'],
    low=df_ha['HA_Low'],
    close=df_ha['HA_Close'],
    name="Heikin-Ashi"
)])
fig_ha.update_layout(title="Heikin-Ashi Chart", template="plotly_white")
st.plotly_chart(fig_ha, use_container_width=True)


st.header("Technical Indicators")

# tabs = st.tabs([" Bollinger Bands", " RSI & Volatility", " MACD", "OBV & ADX"])


st.header("Bollinger Bands")

fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', line=dict(color='red')))
fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low', line=dict(color='green'), fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))
fig_bb.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Mid', line=dict(color='blue')))
fig_bb.update_layout(title="Bollinger Bands", template='plotly_white')
st.plotly_chart(fig_bb, use_container_width=True)
    
    
st.header("RSI")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
# fig_rsi.add_trace(go.Scatter(x=df.index, y=df['Volatility_10'], name='Volatility', line=dict(color='orange')))
fig_rsi.update_layout(title="RSI", template='plotly_white')
st.plotly_chart(fig_rsi, use_container_width=True)

st.header("Volatility")

fig_vol = go.Figure()
# fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
fig_vol.add_trace(go.Scatter(x=df.index, y=df['Volatility_10'], name='Volatility', line=dict(color='orange')))
fig_vol.update_layout(title="Volatility", template='plotly_white')
st.plotly_chart(fig_vol, use_container_width=True)

st.header(" MACD")



fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')))
fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='MACD Diff', marker_color='gray'))
fig_macd.update_layout(title="MACD", template='plotly_white')
st.plotly_chart(fig_macd, use_container_width=True)

    

# st.header("Correlation Heatmap")

# Calculate the correlation matrix
corr_matrix = df.corr()

# 
# heatmap = go.Figure(data=go.Heatmap(
#     z=corr_matrix.values,
#     x=corr_matrix.columns.tolist(),
#     y=corr_matrix.index.tolist(),
#     colorscale='Viridis',
#     zmin=-1,
#     zmax=1,
#     colorbar=dict(title="Correlation"),
#     hoverongaps=False
# ))

# # Improve layout for label readability
# heatmap.update_layout(
#     title="Colinearity Heatmap",
#     template='plotly_white',
#     xaxis=dict(tickangle=45, tickfont=dict(size=10), side='bottom'),
#     yaxis=dict(tickfont=dict(size=10), autorange='reversed'),
#     width=800,
#     height=800,
#     margin=dict(l=100, b=100, t=50, r=50)
# )

# # Display in Streamlit
# st.plotly_chart(heatmap, use_container_width=True)

    
st.header("OBV")



fig_obv = go.Figure()
fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='On Balance Volume', line=dict(color='darkgreen')))
fig_obv.update_layout(title="On Balance Volume (OBV)", template='plotly_white')
st.plotly_chart(fig_obv, use_container_width=True)

st.header("ADX")

fig_adx = go.Figure()
fig_adx.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX', line=dict(color='firebrick')))
fig_adx.update_layout(title="Average Directional Index (ADX)", template='plotly_white')
st.plotly_chart(fig_adx, use_container_width=True)


# st.header("= Forecasting (ARIMA / SARIMA)")
from sklearn.metrics import mean_absolute_error

st.header(" Forecasting (ARIMA & SARIMA)")

df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# ARIMA
model_arima = ARIMA(df['Close'], order=(1, 1, 1)).fit()
forecast_arima = model_arima.forecast(steps=20)
mse_arima = mean_squared_error(df['Close'].iloc[-20:], forecast_arima)
mae_arima = mean_absolute_error(df['Close'].iloc[-20:], forecast_arima)

# SARIMA
model_sarima = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
forecast_sarima = model_sarima.forecast(steps=20)
mse_sarima = mean_squared_error(df['Close'].iloc[-20:], forecast_sarima)
mae_sarima = mean_absolute_error(df['Close'].iloc[-20:], forecast_sarima)

future_dates = pd.date_range(df.index[-1], periods=21, freq='B')[1:]

# ARIMA Plot
st.subheader("ARIMA Forecast")
fig_arima = go.Figure()
fig_arima.add_trace(go.Scatter(x=future_dates, y=forecast_arima, name="ARIMA Forecast", line=dict(color='orange')))
fig_arima.update_layout(title="ARIMA Forecast (Next 20 Days)", template='plotly_white')
st.plotly_chart(fig_arima, use_container_width=True)
st.info(f"ARIMA MSE: {mse_arima:.4f} | MAE: {mae_arima:.4f}")

# SARIMA Plot
st.subheader("SARIMA Forecast")
fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=future_dates, y=forecast_sarima, name="SARIMA Forecast", line=dict(color='blue')))
fig_sarima.update_layout(title="SARIMA Forecast (Next 20 Days)", template='plotly_white')
st.plotly_chart(fig_sarima, use_container_width=True)
st.info(f"SARIMA MSE: {mse_sarima:.4f} | MAE: {mae_sarima:.4f}")


# Machine Learning - Visualization
st.header("Machine Learning - Random Forest & XGBoost")

features = ['Close', 'Daily_returns', 'Log_returns', 'Volatility_10', 'MA_10', 'EMA_10',
            'RSI', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low', 'OBV', 'ADX']
X = df[features]
y = df['Target']

X_train, X_test = X[:-60], X[-60:]
y_train, y_test = y[:-60], y[-60:]

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
st.success(f"Random Forest MSE: {mse_rf:.4f}")

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
st.success(f"XGBoost MSE: {mse_xgb:.4f}")

# Visualize Predictions
col1, col2 = st.columns(2)

with col1:
    fig_rf = go.Figure()
    fig_rf.add_trace(go.Scatter(y=y_test.values, name='Actual', line=dict(color='yellow')))
    fig_rf.add_trace(go.Scatter(y=y_pred_rf, name='Predicted (RF)', line=dict(color='green')))
    fig_rf.update_layout(title="Random Forest: Actual vs Predicted", template="plotly_white")
    st.plotly_chart(fig_rf, use_container_width=True)

with col2:
    fig_xgb = go.Figure()
    fig_xgb.add_trace(go.Scatter(y=y_test.values, name='Actual', line=dict(color='yellow')))
    fig_xgb.add_trace(go.Scatter(y=y_pred_xgb, name='Predicted (XGB)', line=dict(color='blue')))
    fig_xgb.update_layout(title="XGBoost: Actual vs Predicted", template="plotly_white")
    st.plotly_chart(fig_xgb, use_container_width=True)

# Next Day Prediction
next_day_pred_rf = rf.predict(X.tail(1))[0]
next_day_pred_xgb = xgb_model.predict(X.tail(1))[0]
st.metric("Random Forest Prediction (Next Day)", f"${next_day_pred_rf:.2f}")
st.metric("XGBoost Prediction (Next Day)", f"${next_day_pred_xgb:.2f}")


st.header("Investment Signal Recommendation")

# Latest values
latest = df.iloc[-1]

signal_score = 0

# RSI Signal
if latest['RSI'] < 30:
    signal_score += 1  # Oversold -> Buy
elif latest['RSI'] > 70:
    signal_score -= 1  # Overbought -> Sell

# MACD Signal
if latest['MACD'] > latest['MACD_Signal']:
    signal_score += 1  # Positive momentum
else:
    signal_score -= 1

# ADX Signal
if latest['ADX'] > 25:
    if latest['MACD'] > latest['MACD_Signal']:
        signal_score += 1  # Strong uptrend
    else:
        signal_score -= 1  # Strong downtrend

# Moving Average Signal
if latest['Close'] > latest['MA_10']:
    signal_score += 1  # Above MA -> Bullish
else:
    signal_score -= 1

if signal_score >= 2:
    recommendation = "Recommendation: Go LONG (Buy)"
    color = "green"
elif signal_score <= -2:
    recommendation = "Recommendation: Go SHORT (Sell)"
    color = "red"
else:
    recommendation = "⏸ Recommendation: HOLD (Wait)"
    color = "orange"

# Display
st.markdown(f"<h3 style='color:{color};'>{recommendation}</h3>", unsafe_allow_html=True)
st.markdown(f"**Signal Score:** `{signal_score}` (Range: -4 to +4)")
