import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


def ticker_info():
    ticker = "gush"
    return ticker.upper()


def fetch_expiration_dates(ticker):
    print(f"Fetching available expiration dates for {ticker}...")
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    print(f"Available expiration dates: {expiration_dates}")
    return expiration_dates


def select_expiration_date(expiration_dates):
    print("Selecting the first available expiration date...")
    expiration_date = expiration_dates[0]
    print(f"Selected expiration date: {expiration_date}")
    return expiration_date


def fetch_option_chain(ticker, expiration_date):
    print(f"Fetching option chain for {ticker} with expiration date {expiration_date}...")
    stock = yf.Ticker(ticker)
    options_chain = stock.option_chain(expiration_date)
    print("Option chain fetched successfully!")
    return options_chain


def get_price_data(ticker, start_date, end_date):
    print(f"Fetching price data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Price data fetched successfully for {ticker}!")
    return data


def moving_average_strategy(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, -1)
    return data['Signal']

def rsi_strategy(data, window=14, overbought=70, oversold=30):
    delta = data['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0).flatten()  # Flatten to 1D array
    loss = np.where(delta < 0, abs(delta), 0).flatten()  # Flatten to 1D array
    
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    
    # Avoid division by zero by using np.where to replace 0 with np.nan in avg_loss
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)  
    
    rsi = 100 - (100 / (1 + rs))
    
    signal = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
    return pd.Series(signal, index=data.index)

def bollinger_bands_strategy(data, window=20, num_std=2):
    # Calculate moving average
    data['Moving_Avg'] = data['Close'].rolling(window=window).mean()

    # Calculate rolling standard deviation and force it to be a Series
    rolling_std = data['Close'].rolling(window).std()
    rolling_std = rolling_std.squeeze()  # Ensure rolling_std is a Series

    # Print shapes for debugging
    print(f"Shape of Moving_Avg: {data['Moving_Avg'].shape}")
    print(f"Shape of Rolling Std: {rolling_std.shape}")

    # Calculate upper and lower bands
    data['Band_Upper'] = data['Moving_Avg'] + (num_std * rolling_std)
    data['Band_Lower'] = data['Moving_Avg'] - (num_std * rolling_std)

    # Print shapes after assignments for debugging
    print(f"Shape of Band_Upper: {data['Band_Upper'].shape}")
    print(f"Shape of Band_Lower: {data['Band_Lower'].shape}")

    # Check for NaN values
    print(f"NaNs in Close: {data['Close'].isna().sum()}")
    print(f"NaNs in Band_Upper: {data['Band_Upper'].isna().sum()}")
    print(f"NaNs in Band_Lower: {data['Band_Lower'].isna().sum()}")

    # Print the columns of the DataFrame
    print(f"Columns in data before dropping NaNs: {data.columns.tolist()}")

    # Optionally drop rows with NaNs
    data = data.dropna(subset=['Close', 'Band_Upper', 'Band_Lower'])

    # Generate signals based on the bands
    signal = np.where(data['Close'] < data['Band_Lower'], 1, 
                      np.where(data['Close'] > data['Band_Upper'], -1, 0))
    
    return pd.Series(signal, index=data.index)

def generate_signals(data):
    ma_signal = moving_average_strategy(data)
    rsi_signal = rsi_strategy(data)
    bollinger_signal = bollinger_bands_strategy(data)
    return pd.DataFrame({'MA': ma_signal, 'RSI': rsi_signal, 'Bollinger': bollinger_signal})


def backtest_option_trades(option_chain, signals, stock_data):
    """
    Backtest option trades based on the given signals and stock data.
    """
    trades = []
    current_position = None

    # Ensure both stock_data and option_chain indices are sorted in ascending order
    stock_data = stock_data.sort_index()

    # Convert 'lastTradeDate' or any date-related columns to datetime in option_chain
    if 'lastTradeDate' in option_chain.columns:
        option_chain['lastTradeDate'] = pd.to_datetime(option_chain['lastTradeDate'])
        option_chain = option_chain.set_index('lastTradeDate')

    # If option_chain index isn't datetime, convert it to datetime (ensuring compatibility)
    option_chain.index = pd.to_datetime(option_chain.index)

    # Remove the timezone from option_chain index
    option_chain.index = option_chain.index.tz_localize(None)

    # Now reindex the option chain to match the stock data index (forward fill missing option prices)
    option_chain = option_chain.sort_index()
    option_chain = option_chain.reindex(stock_data.index, method='ffill')

    for i in range(len(signals)):
        if signals.iloc[i]['MA'] == 1 and current_position is None:
            # BUY signal
            entry_price = option_chain['lastPrice'].iloc[i]
            if pd.isna(entry_price):  # If price is nan, log the error and continue
                print(f"Missing entry price on {stock_data.index[i]}, skipping trade.")
                continue
            entry_date = stock_data.index[i]
            current_position = {
                'entry_price': entry_price,
                'entry_date': entry_date
            }
            print(f"BUY signal on {entry_date}: Entry Price = {entry_price}")
        
        elif signals.iloc[i]['MA'] == -1 and current_position is not None:
            # SELL signal
            exit_price = option_chain['lastPrice'].iloc[i]
            if pd.isna(exit_price):  # If price is nan, log the error and continue
                print(f"Missing exit price on {stock_data.index[i]}, skipping trade.")
                continue
            exit_date = stock_data.index[i]
            pnl = (exit_price - current_position['entry_price']) * 100
            print(f"SELL signal on {exit_date}: Exit Price = {exit_price}, P&L = {pnl}")

            trades.append({
                'entry_date': current_position['entry_date'],
                'entry_price': current_position['entry_price'],
                'exit_date': exit_date,
                'exit_price': exit_price,
                'pnl': pnl
            })
            current_position = None

    cumulative_pnl = sum(trade['pnl'] for trade in trades)
    total_wins = sum(1 for trade in trades if trade['pnl'] > 0)
    total_trades = len(trades)
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    return cumulative_pnl, trades, win_rate


def objective_function_profit(weights, strategy_signals, data, option_chain):
    weights = np.array(weights)
    weights /= np.sum(weights)  # Normalize weights
    weighted_signals = np.sum([signal * weight for signal, weight in zip(strategy_signals.T.values, weights)], axis=0)

    # Since `backtest_option_trades` returns 3 values, we only unpack those
    cumulative_pnl, _, _ = backtest_option_trades(option_chain, weighted_signals, data)

    # Return negative cumulative P&L to maximize profit
    return -cumulative_pnl


def optimize_weights(strategy_signals, data, option_chain):
    initial_weights = [1 / len(strategy_signals.columns)] * len(strategy_signals.columns)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = [(0, 1)] * len(strategy_signals.columns)

    result = minimize(objective_function_profit, initial_weights, args=(strategy_signals, data, option_chain),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x  # Optimal weights


def weighted_signal_combination(strategy_signals, weights):
    weighted_signals = np.sum([signal * weight for signal, weight in zip(strategy_signals.T.values, weights)], axis=0)
    return weighted_signals


def main_decision(weighted_signals):
    last_signal = weighted_signals[-1]  # Latest signal
    if last_signal > 0:
        return "BUY"
    elif last_signal < 0:
        return "SELL"
    else:
        return "HOLD"


def run_backtest():
    ticker = ticker_info()
    expiration_dates = fetch_expiration_dates(ticker)
    expiration_date = select_expiration_date(expiration_dates)
    options_chain = fetch_option_chain(ticker, expiration_date)

    # Fetch training data
    train_data = get_price_data(ticker, '2010-01-01', '2022-01-01')

    # Generate signals
    strategy_signals_train = generate_signals(train_data)

    # Optimize weights
    optimal_weights = optimize_weights(strategy_signals_train, train_data, options_chain.calls)

    # Fetch test data
    test_data = get_price_data(ticker, '2022-01-02', '2024-01-01')

    # Generate test signals
    strategy_signals_test = generate_signals(test_data)

    # Combine signals and backtest
    weighted_signals = weighted_signal_combination(strategy_signals_test, optimal_weights)
    cumulative_pnl, trades, win_rate = backtest_option_trades(options_chain.calls, weighted_signals, test_data)

    # Make final decision
    decision = main_decision(weighted_signals)
    print(f"Final decision: {decision}")

    # Output results
    print(f"Cumulative P&L: {cumulative_pnl}")
    print(f"Win Rate: {win_rate * 100:.2f}%")


# Call the main function
run_backtest()
