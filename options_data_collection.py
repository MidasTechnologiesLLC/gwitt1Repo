import signal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm  # For progress bar


class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.df = pd.DataFrame()
        self.data_retrieved = False

    def historicalData(self, reqId, bar):
        self.data.append({
            "Date": bar.date,
            "Open": bar.open,
            "High": bar.high,
            "Low": bar.low,
            "Close": bar.close,
            "Volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        chunk_df = pd.DataFrame(self.data)
        self.df = pd.concat([self.df, chunk_df], ignore_index=True)
        self.data_retrieved = True
        self.data = []


class IBApp:
    def __init__(self):
        self.app = IBApi()

    def connect(self):
        self.app.connect("127.0.0.1", 4002, clientId=1)
        thread = threading.Thread(target=self.run_app, daemon=True)
        thread.start()
        time.sleep(1)

    def run_app(self):
        self.app.run()

    def request_data(self, contract, end_date, duration, bar_size):
        self.app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_date,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=0,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        # Ensure pacing between API calls
        while not self.app.data_retrieved:
            time.sleep(0.1)

    def fetch_options_data(self, symbol, exchange, currency, right, strike, expiry):
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "OPT"  # Set security type to options
            contract.exchange = exchange
            contract.currency = currency
            contract.right = right  # 'C' for Call, 'P' for Put
            contract.strike = float(strike)  # Strike price
            contract.lastTradeDateOrContractMonth = expiry  # Expiry date in YYYYMMDD format

            # Set duration and bar size for options data
            duration = "1 D"  # 1 day chunks
            bar_size = "1 min"  # 1-minute intervals

            end_date = datetime.now(timezone.utc)

            # Since options data typically spans less than a year, we fetch for the expiry
            with tqdm(total=1, desc=f"Fetching {right} {strike} {expiry} data", unit="contract") as pbar:
                end_date_str = end_date.strftime("%Y%m%d %H:%M:%S UTC")
                try:
                    self.request_data(contract, end_date_str, duration, bar_size)
                    pbar.update(1)
                    time.sleep(15)  # Sleep to avoid pacing violations
                except Exception as e:
                    print(f"Error fetching data for contract {contract.symbol}: {e}")
        except Exception as e:
            print(f"Error fetching data: {e}")

    def disconnect(self):
        self.app.disconnect()


def get_user_input():
    print("Provide the options contract details for data retrieval.")
    try:
        symbol = input("Enter the stock symbol (e.g., 'AAPL'): ").strip().upper()
        exchange = "SMART"  # Automatically set to SMART routing
        currency = "USD"  # Automatically set to USD
        right = input("Enter the option type ('C' for Call, 'P' for Put): ").strip().upper()
        strike = input("Enter the strike price (e.g., '150'): ").strip()
        expiry = input("Enter the expiry date (YYYYMMDD): ").strip()

        if not all([symbol, right, strike, expiry]):
            raise ValueError("All fields are required. Please try again.")

        return symbol, exchange, currency, right, strike, expiry
    except Exception as e:
        print(f"Input Error: {e}")
        return None


def graceful_exit(signal_received, frame):
    print("\nTerminating program...")
    app.disconnect()
    exit(0)


signal.signal(signal.SIGINT, graceful_exit)

app = IBApp()
app.connect()

try:
    user_input = get_user_input()
    if user_input:
        symbol, exchange, currency, right, strike, expiry = user_input
        app.fetch_options_data(symbol, exchange, currency, right, strike, expiry)
        data = app.app.df
        if not data.empty:
            filename = f"{symbol}_{strike}_{right}_{expiry}_options_data.csv"
            data.to_csv(filename, index=False)
            print(f"Options data saved to {filename}.")
            print(data.head())
        else:
            print("No options data retrieved.")
except Exception as e:
    print(f"Error: {e}")
finally:
    app.disconnect()
