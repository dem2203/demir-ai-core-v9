import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("FEATURE_ENGINEERING")

class FeatureEngineer:
    """
    Ham piyasa verilerini AI modellerinin anlayabileceği matematiksel vektörlere dönüştürür.
    Mock veri kullanılmaz, tamamen Pandas ve Numpy ile gerçek hesaplama yapılır.
    """

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) Hesaplaması"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2):
        """Bollinger Bantları Hesaplaması"""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD (Moving Average Convergence Divergence) Hesaplaması"""
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    @classmethod
    def process_data(cls, raw_data: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Ham veri listesini alır, DataFrame'e çevirir ve indikatörleri ekler.
        """
        if not raw_data:
            logger.warning("No data provided for feature engineering.")
            return None

        try:
            df = pd.DataFrame(raw_data)
            
            # Veri tiplerini zorla (Güvenlik)
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)
            
            # İndikatörleri Hesapla (Gerçek Matematik)
            df['rsi'] = cls.calculate_rsi(df)
            df['bb_upper'], df['bb_lower'] = cls.calculate_bollinger_bands(df)
            df['macd'], df['macd_signal'] = cls.calculate_macd(df)
            
            # Volatilite (Log Return)
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
            # NaN değerleri temizle (İlk hesaplama satırları boş kalır)
            df.dropna(inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Feature Engineering Error: {e}")
            return None