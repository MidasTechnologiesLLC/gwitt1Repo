import signal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.df = pd.DataFrame()
        self.data_retrieved = False

    def historicalData(self, reqId, bar):
        # Debug: Print each received bar
        print(f"Received bar: Date={bar.date}, Open={bar.open}, High={bar.high}, Low={bar.low}, Close={bar.close}, Volume={bar.volume}")
        self.data.append({
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        # Debug: Indicate end of data reception
        print(f"HistoricalDataEnd received. Start: {start}, End: {end}. Number of bars fetched: {len(self.data)}")
        chunk_df = pd.DataFrame(self.data)
        if not chunk_df.empty:
            self.df = pd.concat([self.df, chunk_df], ignore_index=True)
        else:
            print("No data received in this request.")
        self.data_retrieved = True
        self.data = []  # Reset data list for next request

class IBApp:
    def __init__(self):
        self.app = IBApi()

    def connect(self):
        # Connect to IB API (ensure IB Gateway or TWS is running)
        print("Connecting to IB API...")
        self.app.connect("127.0.0.1", 4002, clientId=1)
        # Start the API thread
        thread = threading.Thread(target=self.run_app, daemon=True)
        thread.start()
        time.sleep(1)  # Allow time for connection
        print("Connected to IB API.")

    def run_app(self):
        self.app.run()

    def request_data(self, contract, end_date, duration, bar_size):
        # Request historical data
        print(f"Requesting data: endDateTime={end_date}, durationStr={duration}, barSizeSetting={bar_size}")
        self.app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,  # Use regular trading hours
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        # Wait until data is retrieved
        while not self.app.data_retrieved:
            time.sleep(0.1)
        self.app.data_retrieved = False  # Reset flag for next request

    def fetch_recent_data(self, symbol, sec_type, exchange, currency):
        try:
            # Define the contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

            # Set duration and bar size for last 2 days
            duration = "2 D"        # 2 days
            bar_size = "1 min"      # 1-minute intervals

            # Set end_date to now in UTC
            end_date = datetime.now(timezone.utc)
            end_date_str = end_date.strftime("%Y%m%d %H:%M:%S UTC")
            print(f"Fetching data up to {end_date_str} for the last {duration} with bar size {bar_size}")
            self.request_data(contract, end_date_str, duration, bar_size)

        except Exception as e:
            print(f"Error fetching data: {e}")

    def disconnect(self):
        self.app.disconnect()
        print("Disconnected from IB API.")

def get_user_input():
    print("Provide the stock details for historical data retrieval.")
    try:
        symbol = input("Enter the stock symbol (e.g., 'AAPL'): ").strip().upper()
        sec_type = "STK"        # Automatically set to Stock
        exchange = "SMART"      # Automatically set to SMART routing
        currency = "USD"        # Automatically set to USD

        if not symbol:
            raise ValueError("Stock symbol is required. Please try again.")

        return symbol, sec_type, exchange, currency
    except Exception as e:
        print(f"Input Error: {e}")
        return None

def graceful_exit(signal_received, frame):
    print("\nTerminating program...")
    app.disconnect()
    exit(0)

# Handle graceful exit on Ctrl+C
signal.signal(signal.SIGINT, graceful_exit)

# Initialize and connect the IBApp
app = IBApp()
app.connect()

try:
    user_input = get_user_input()
    if user_input:
        symbol, sec_type, exchange, currency = user_input

        # Define the filename (save directly in current directory)
        filename = f"{symbol}_recent_data.csv"

        # Fetch recent data (last 2 days)
        app.fetch_recent_data(symbol, sec_type, exchange, currency)

        # Retrieve fetched data
        data = app.app.df
        if not data.empty:
            print(f"Number of data points fetched: {len(data)}")
            # Clean and parse the 'Date' column
            # Attempt multiple formats
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

            # Check if timezone is present; if not, localize to UTC
            if data['Date'].dt.tz is None:
                data['Date'] = data['Date'].dt.tz_localize(timezone.utc, ambiguous='NaT', nonexistent='NaT')

            # Remove any rows with NaT in 'Date'
            data.dropna(subset=['Date'], inplace=True)

            # Sort by 'Date' ascending
            data.sort_values(by='Date', inplace=True)

            # Save to CSV
            data.to_csv(filename, index=False)
            print(f"Data saved to {filename}.")
            print(data.tail())
        else:
            print("No new data fetched.")
except Exception as e:
    print(f"Error: {e}")
finally:
    app.disconnect()
