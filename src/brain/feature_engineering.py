import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("ULTIMATE_FEATURE_ENGINEERING")

class FeatureEngineer:
    """
    DEMIR AI V13.0 - QUANTITATIVE MAP MAKER (FULL)
    
    Özellikler:
    1. Klasik Teknik Analiz (RSI, MACD, Bollinger, ATR, ADX)
    2. İleri Matematik (Hurst Exponent, Ichimoku)
    3. Formasyon Tanıma (Candlestick Patterns)
    4. Veri Füzyonu (Crypto + Macro)
    5. Fiyat Haritalama (Pivot Points & Fibonacci)
    """

    # ==========================================
    # 1. MOMENTUM VE OSİLATÖRLER
    # ==========================================
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        exp1 = data['close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_roc(series: pd.Series, period: int = 9) -> pd.Series:
        return series.pct_change(periods=period) * 100

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        positive_mf = pd.Series(positive_flow).rolling(window=period).mean()
        negative_mf = pd.Series(negative_flow).rolling(window=period).mean()
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi

    # ==========================================
    # 2. TREND VE VOLATİLİTE
    # ==========================================

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        df = data.copy()
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()

        df['up'] = df['high'] - df['high'].shift(1)
        df['down'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
        df['minus_dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
        plus_di = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
        minus_di = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
        
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, 1) 
        dx = 100 * abs(plus_di - minus_di) / di_sum
        return dx.rolling(window=period).mean()

    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame) -> pd.DataFrame:
        tenkan = (data['high'].rolling(9).max() + data['low'].rolling(9).min()) / 2
        kijun = (data['high'].rolling(26).max() + data['low'].rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((data['high'].rolling(52).max() + data['low'].rolling(52).min()) / 2).shift(26)
        return pd.DataFrame({'tenkan': tenkan, 'kijun': kijun, 'span_a': span_a, 'span_b': span_b})

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def calculate_bollinger_width(data: pd.DataFrame, period: int = 20) -> pd.Series:
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return (upper - lower) / sma

    @staticmethod
    def calculate_z_score(series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        std = std.replace(0, 0.0001)
        return (series - mean) / std

    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        v = data['volume']
        tp = (data['high'] + data['low'] + data['close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    # ==========================================
    # 3. İLERİ MATEMATİK (KAOS & FORMASYON)
    # ==========================================

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        try:
            if len(series) < max_lag + 2: return 0.5
            if series.std() == 0: return 0.5
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            if len(tau) < 2: return 0.5
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            val = poly[0] * 2.0
            if np.isnan(val) or np.isinf(val): return 0.5
            return val
        except:
            return 0.5

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        body = np.abs(df['close'] - df['open'])
        range_len = df['high'] - df['low']
        range_len = range_len.replace(0, 0.0001)
        
        df['is_doji'] = np.where(body <= (range_len * 0.1), 1, 0)
        df['is_hammer'] = np.where((np.minimum(df['close'], df['open']) - df['low']) >= (body * 2), 1, 0)
        df['is_engulfing'] = np.where(
            (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & 
            (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)), 1, 0
        )
        return df[['is_doji', 'is_hammer', 'is_engulfing']]

    # ==========================================
    # 4. YENİ: PIVOT & HARİTALAMA
    # ==========================================
    
    @staticmethod
    def calculate_pivots(df: pd.DataFrame) -> pd.DataFrame:
        """
        Destek/Direnç seviyelerini hesaplar.
        """
        # Pivotlar bir önceki mumun verisine göre hesaplanır
        high = df['high'].shift(1)
        low = df['low'].shift(1)
        close = df['close'].shift(1)
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        return pd.DataFrame({'pivot': pivot, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2})

    # ==========================================
    # 5. VERİ FÜZYONU (KORUNDU)
    # ==========================================

    @staticmethod
    def merge_crypto_and_macro(crypto_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        if macro_df is None or macro_df.empty: return crypto_df
        crypto_df = crypto_df.sort_values('timestamp')
        macro_df = macro_df.sort_values('timestamp')
        merged_df = pd.merge_asof(crypto_df, macro_df, on='timestamp', direction='backward')
        merged_df = merged_df.ffill().bfill()
        return merged_df

    # ==========================================
    # ANA İŞLEMCİ
    # ==========================================

    @classmethod
    def process_data(cls, raw_data: List[Dict]) -> Optional[pd.DataFrame]:
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data)
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)

            # İndikatörler
            df['rsi'] = cls.calculate_rsi(df)
            macd, macd_signal = cls.calculate_macd(df)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['roc'] = cls.calculate_roc(df['close'])
            df['adx'] = cls.calculate_adx(df)
            df['atr'] = cls.calculate_atr(df)
            df['mfi'] = cls.calculate_mfi(df)
            
            ichi = cls.calculate_ichimoku(df)
            df = pd.concat([df, ichi], axis=1)
            
            df['bb_width'] = cls.calculate_bollinger_width(df)
            df['vwap'] = cls.calculate_vwap(df)
            df['z_score'] = cls.calculate_z_score(df['close'])
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['hurst'] = df['close'].rolling(window=100).apply(lambda x: cls.calculate_hurst_exponent(x))

            patterns = cls.detect_patterns(df)
            df = pd.concat([df, patterns], axis=1)

            # YENİ: PIVOTS
            pivots = cls.calculate_pivots(df)
            df = pd.concat([df, pivots], axis=1)

            for lag in [1, 2, 3, 5, 8]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'vol_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)

            df = df.iloc[100:]
            df = df.ffill() 
            df.fillna(0, inplace=True)
            
            logger.info(f"Feature Engineering Complete. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"FEATURE ENGINEERING CRITICAL FAIL: {e}")
            return None
