import ta
import pandas as pd

preprocessed_file_path = 'C:/Users/gwitt/MidasTechnologies/API/spy_1min_preprocessed.csv'  # Replace with your file path
df = pd.read_csv(preprocessed_file_path, index_col='Date', parse_dates=True)

# **Trend Indicators**
# Simple Moving Averages
df['SMA_20'] = ta.trend.sma_indicator(close=df['Close'], window=20)
df['SMA_50'] = ta.trend.sma_indicator(close=df['Close'], window=50)
df['SMA_200'] = ta.trend.sma_indicator(close=df['Close'], window=200)

# Exponential Moving Averages
df['EMA_20'] = ta.trend.ema_indicator(close=df['Close'], window=20)
df['EMA_50'] = ta.trend.ema_indicator(close=df['Close'], window=50)

# MACD
macd = ta.trend.MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()

# ADX
df['ADX_14'] = ta.trend.adx(high=df['High'], low=df['Low'], close=df['Close'], window=14)

# **Momentum Indicators**
# RSI
df['RSI_14'] = ta.momentum.rsi(close=df['Close'], window=14)

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_%K'] = stoch.stoch()
df['Stoch_%D'] = stoch.stoch_signal()

# Rate of Change
df['ROC_10'] = ta.momentum.roc(close=df['Close'], window=10)

# **Volatility Indicators**
# Bollinger Bands
bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
df['Bollinger_High'] = bollinger.bollinger_hband()
df['Bollinger_Low'] = bollinger.bollinger_lband()
df['Bollinger_Middle'] = bollinger.bollinger_mavg()

# Average True Range
df['ATR_14'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=14)

# **Volume Indicators**
# On-Balance Volume
df['OBV'] = ta.volume.on_balance_volume(close=df['Close'], volume=df['Volume'])

# Volume Weighted Average Price
df['VWAP'] = ta.volume.volume_weighted_average_price(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])

# Chaikin Money Flow
df['CMF_20'] = ta.volume.chaikin_money_flow(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=20)

# **Composite Indicators**
# # Ichimoku Cloud
# ichimoku = ta.trend.IchimokuIndicator(high=df['High'], low=df['Low'], close=df['Close'], window1=9, window2=26, window3=52)
# df['Ichimoku_A'] = ichimoku.ichimoku_a()
# df['Ichimoku_B'] = ichimoku.ichimoku_b()
# df['Ichimoku_Base_Line'] = ichimoku.ichimoku_base_line()
# df['Ichimoku_Conversion_Line'] = ichimoku.ichimoku_conversion_line()

# Parabolic SAR
df['PSAR'] = ta.trend.psar_up(close=df['Close'], high=df['High'], low=df['Low'], step=0.02, max_step=0.2)


# **Classification Target:** 1 if next minute's close > current close, else 0
df['Target_Class'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# **Regression Target:** Percentage change in close price
df['Target_Change'] = ((df['Close'].shift(-1) - df['Close']) / df['Close']) * 100

# Display targets
print("\nTarget Variables:")
print(df[['Close', 'Target_Class', 'Target_Change']].head())

# Define lag periods
lag_periods = [1, 2, 3]

# Create lagged features for key indicators
key_indicators = ['RSI_14', 'MACD', 'ADX_14', 'ATR_14', 'OBV', 'CMF_20']

for indicator in key_indicators:
    for lag in lag_periods:
        df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)

# Display lagged features
print("\nLagged Features:")
print(df[[f'RSI_14_lag{lag}' for lag in lag_periods]].head())

# Rolling mean of RSI over past 5 minutes
df['RSI_14_roll_mean_5'] = df['RSI_14'].rolling(window=5).mean()

# Rolling standard deviation of ATR over past 10 minutes
df['ATR_14_roll_std_10'] = df['ATR_14'].rolling(window=10).std()

# Display rolling features
print("\nRolling Features:")
print(df[['RSI_14_roll_mean_5', 'ATR_14_roll_std_10']].head())

# Interaction between MACD and RSI
df['MACD_RSI'] = df['MACD'] * df['RSI_14']

# Interaction between ATR and ADX
df['ATR_ADX'] = df['ATR_14'] * df['ADX_14']

# Display interaction features
print("\nInteraction Features:")
print(df[['MACD_RSI', 'ATR_ADX']].head())


# Save dataset with technical indicators
indicators_file_path = 'C:/Users/gwitt/MidasTechnologies/API/spy_1min_with_indicators.csv'  # Replace with your desired path
df.to_csv(indicators_file_path)

print(f"Data with technical indicators saved to {indicators_file_path}")
