import signal
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
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

    def fetch_historical_data(self, symbol, sec_type, exchange, currency):
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

            # Set duration and bar size
            duration = "1 D"  # 1 day chunks
            bar_size = "5 mins"  # 1-minute intervals

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=365) #Can multiply for more years

            total_days = (end_date - start_date).days
            with tqdm(total=total_days, desc="Fetching Data", unit="day") as pbar:
                current_date = end_date
                while current_date > start_date:
                    end_date_str = current_date.strftime("%Y%m%d %H:%M:%S UTC")
                    try:
                        self.request_data(contract, end_date_str, duration, bar_size)
                        pbar.update(1)
                        time.sleep(5)  # Sleep to avoid pacing violations
                    except Exception as e:
                        print(f"Error fetching data for {end_date_str}: {e}")
                    current_date -= timedelta(days=1)
        except Exception as e:
            print(f"Error fetching data: {e}")

    def disconnect(self):
        self.app.disconnect()


def get_user_input():
    print("Provide the stock details for historical data retrieval.")
    try:
        symbol = input("Enter the stock symbol (e.g., 'AAPL'): ").strip().upper()
        sec_type = "STK"  # Automatically set to Stock
        exchange = "SMART"  # Automatically set to SMART routing
        currency = "USD"  # Automatically set to USD

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


signal.signal(signal.SIGINT, graceful_exit)

app = IBApp()
app.connect()

try:
    user_input = get_user_input()
    if user_input:
        symbol, sec_type, exchange, currency = user_input
        app.fetch_historical_data(symbol, sec_type, exchange, currency)
        data = app.app.df
        if not data.empty:
            filename = f"{symbol}_1yr_5min_data.csv"
            data.to_csv(filename, index=False)
            print(f"Data saved to {filename}.")
            print(data.head())
        else:
            print("No data retrieved.")
except Exception as e:
    print(f"Error: {e}")
finally:
    app.disconnect()
