import signal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm  # For progress bar
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

    def fetch_historical_data_yearly(self, symbol, sec_type, exchange, currency, start_date, end_date, bar_size="1 day"):
        """
        Fetch historical data in yearly chunks to cover 3 years.
        """
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

            delta = timedelta(days=365)
            current_end_date = end_date

            total_years = 3  # Fetch 3 years of data
            with tqdm(total=total_years, desc="Fetching Data", unit="year") as pbar:
                for _ in range(total_years):
                    current_start_date = current_end_date - delta
                    end_date_str = current_end_date.strftime("%Y%m%d %H:%M:%S UTC")
                    self.request_data(contract, end_date_str, "1 Y", bar_size)
                    pbar.update(1)
                    current_end_date = current_start_date
                    time.sleep(1)  # Respect IB API pacing
        except Exception as e:
            print(f"Error fetching data: {e}")

    def fetch_historical_data(self, symbol, sec_type, exchange, currency, existing_df=None):
        """
        Fetch historical data for the given symbol.
        If existing_df is provided, fetch data after the last date in existing_df.
        Otherwise, fetch the entire 3 years of data.
        """
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

            bar_size = "1 day"   # Set bar size to 1 day for daily data
            duration = "1 Y"      # Fetch 1 year at a time

            if existing_df is not None and not existing_df.empty:
                # Get the last date from existing data
                last_date_str = existing_df['Date'].iloc[-1]
                # Clean up the date string to have single space
                last_date_str = last_date_str.strip().replace('  ', ' ')
                # Parse the last date as timezone-aware datetime (assuming UTC)
                try:
                    # Try parsing in 'YYYYMMDD HH:MM:SS' format
                    last_date = datetime.strptime(last_date_str, "%Y%m%d %H:%M:%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    try:
                        # If that fails, try 'YYYY-MM-DD HH:MM:SS' format
                        last_date = datetime.strptime(last_date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    except ValueError:
                        print(f"Error parsing last_date_str: {last_date_str}")
                        return

                # Remove any future dates if present
                current_time = datetime.now(timezone.utc)
                existing_df = existing_df[existing_df['Date'] <= current_time]
                print(f"Last valid date after cleaning: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")

                # Fetch new data in yearly chunks
                # Since we need 3 years of data, and assuming existing_df has some, adjust accordingly
                # For simplicity, fetch the entire 3 years again
                # Alternatively, fetch data from last_date forward

                # Here, we'll fetch 3 years of data up to current_date
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=365 * 3)
                self.fetch_historical_data_yearly(symbol, sec_type, exchange, currency, start_date, end_date, bar_size)
            else:
                # No existing data, fetch all 3 years
                end_date = datetime.now(timezone.utc)
                self.fetch_historical_data_yearly(symbol, sec_type, exchange, currency, end_date - timedelta(days=365*3), end_date, bar_size)
        except Exception as e:
            print(f"Error fetching data: {e}")

    def disconnect(self):
        self.app.disconnect()
        print("Disconnected from IB API.")

def get_user_input():
    print("Provide the stock details for historical data retrieval.")
    try:
        symbol = input("Enter the stock symbol (e.g., 'AAPL'): ").strip().upper()
        sec_type = "STK"          # Automatically set to Stock
        exchange = "SMART"        # Automatically set to SMART routing
        currency = "USD"          # Automatically set to USD

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
        filename = f"{symbol}_3yr_daily_data.csv"

        # Fetch historical data
        app.fetch_historical_data(symbol, sec_type, exchange, currency)

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

            # Reset index
            data.reset_index(drop=True, inplace=True)

            # Save to CSV
            data.to_csv(filename, index=False)
            print(f"Data saved to {filename}.")
            print(data.head())
        else:
            print("No data retrieved.")
except Exception as e:
    print(f"Error: {e}")
finally:
    app.disconnect()
