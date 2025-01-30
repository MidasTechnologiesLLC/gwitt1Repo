# midas/feature_engineer.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = RobustScaler()  # Handles outliers
        
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv.pct_change(periods=14)  # Normalized OBV

    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        all_features = []
        
        for ticker, df in data.items():
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            if 'returns' in self.config['features']:
                features['returns'] = np.log(df['close']).diff()
                
            if 'volatility' in self.config['features']:
                features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
                
            if 'rsi' in self.config['features']:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                features['rsi'] = 100 - (100 / (1 + (gain / loss)))
                
            if 'macd' in self.config['features']:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                features['macd'] = ema12 - ema26
                
            if 'atr' in self.config['features']:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['atr'] = tr.rolling(14).mean() / df['close']
                
            if 'volume_change' in self.config['features']:
                features['volume_change'] = np.log(df['volume'] / df['volume'].shift(1))
                
            if 'obv' in self.config['features']:
                features['obv'] = self._calculate_obv(df)
                
            # Add ticker identifier if combining
            if self.config['combine_tickers']:
                features['ticker'] = ticker
                
            all_features.append(features.dropna())
            
        combined = pd.concat(all_features).sort_index()
        
        # Encode tickers if combining
        if self.config['combine_tickers']:
            combined = pd.get_dummies(combined, columns=['ticker'], prefix='', prefix_sep='')
            
        # Scale features
        scaled = pd.DataFrame(
            self.scaler.fit_transform(combined),
            index=combined.index,
            columns=combined.columns
        )
        return scaled
