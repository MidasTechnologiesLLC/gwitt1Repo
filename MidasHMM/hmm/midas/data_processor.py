# midas/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def _load_raw_data(self, ticker: str) -> pd.DataFrame:
        file_path = Path(self.config['data_dir']) / f"{ticker}_5min_3years.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_csv(
            file_path,
            parse_dates=['timestamp'],
            usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            dtype={
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            }
        )
        return df.sort_values('timestamp').set_index('timestamp')

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Robust data cleaning pipeline"""
        # Handle missing values
        df = df.dropna()
        
        # Validate price data
        self._validate_prices(df)
        
        # Validate trading hours
        df = self._filter_trading_hours(df)
        
        # Remove outliers
        df = self._remove_price_outliers(df)
        df = self._remove_volume_outliers(df)
        
        # Resample and forward fill missing intervals
        df = df.resample(self.config['resample_freq']).last().ffill()
        
        return df

    def _validate_prices(self, df: pd.DataFrame):
        """Ensure price data integrity"""
        if (df['close'] <= 0).any():
            bad_values = df[df['close'] <= 0]
            self.logger.error(f"Invalid close prices: {bad_values.index}")
            raise ValueError("Negative/zero close prices detected")
            
        if not (df['high'] >= df['low']).all():
            raise ValueError("High prices < Low prices detected")
            
        if not df['close'].is_monotonic_increasing:
            self.logger.warning("Non-monotonic timestamps detected")

    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-market hours (if intraday data)"""
        if pd.infer_freq(df.index) in ('H', 'T'):
            return df.between_time('09:30', '16:00')
        return df

    def _remove_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove prices beyond 10 standard deviations"""
        returns = np.log(df['close']).diff().dropna()
        mask = (returns.abs() < 10 * returns.std()).reindex(df.index).ffill()
        return df[mask]

    def _remove_volume_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove volume spikes beyond 20 standard deviations"""
        log_volume = np.log(df['volume'])
        mask = (log_volume < log_volume.mean() + 20*log_volume.std())
        return df[mask]

    def process_tickers(self) -> Dict[str, pd.DataFrame]:
        """Process all tickers with cleaning and resampling"""
        processed = {}
        for ticker in self.config['tickers']:
            try:
                self.logger.info(f"Processing {ticker}")
                df = self._load_raw_data(ticker)
                df = self._clean_data(df)
                processed[ticker] = df
            except Exception as e:
                self.logger.error(f"Failed processing {ticker}: {str(e)}")
                raise
        return processed
