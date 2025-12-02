import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("ULTIMATE_FEATURE_ENGINEERING")

class FeatureEngineer:
    """
    DEMIR AI V9.0 - MONSTER EDITION FEATURE ENGINEER
    
    Bu modül, aşağıdaki tüm disiplinleri tek bir veri setinde birleştirir:
    1. İstatistiksel Analiz (Z-Score, Log Returns, Standard Deviation)
    2. Momentum & Trend (RSI, ADX, MACD, ROC)
    3. Volatilite (ATR, Bollinger Band Width)
    4. Kurumsal Hacim (VWAP, MFI)
    5. Gelişmiş Japon Teknikleri (Ichimoku Cloud)
    6. Kaos Teorisi (Hurst Exponent - Piyasa Rastgeleliği)
    7. Formasyon Tanıma (Doji, Hammer, Engulfing)
    """

    # --- YARDIMCI FONKSİYONLAR ---
    @staticmethod
    def _calculate_sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    # --- 1. MOMENTUM & OSİLATÖRLER ---
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_roc(series: pd.Series, period: int = 9) -> pd.Series:
        """Rate of Change: Fiyatın değişim hızı"""
        return series.pct_change(periods=period) * 100

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index: Hacim ağırlıklı RSI"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        # Pozitif ve Negatif Akış
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).mean()
        negative_mf = pd.Series(negative_flow).rolling(window=period).mean()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi

    # --- 2. TREND GÖSTERGELERİ ---
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
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()

    @staticmethod
    def calculate_ichimoku(data: pd.DataFrame) -> pd.DataFrame:
        tenkan = (data['high'].rolling(9).max() + data['low'].rolling(9).min()) / 2
        kijun = (data['high'].rolling(26).max() + data['low'].rolling(26).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(26)
        span_b = ((data['high'].rolling(52).max() + data['low'].rolling(52).min()) / 2).shift(26)
        chikou = data['close'].shift(-26)
        return pd.DataFrame({'tenkan': tenkan, 'kijun': kijun, 'span_a': span_a, 'span_b': span_b, 'chikou': chikou})

    # --- 3. VOLATİLİTE VE İSTATİSTİK ---
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
        return (upper - lower) / sma  # Bant genişliği yüzdesi

    @staticmethod
    def calculate_z_score(series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std

    # --- 4. HACİM VE KURUMSAL İZLER ---
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        v = data['volume']
        tp = (data['high'] + data['low'] + data['close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    # --- 5. KAOS TEORİSİ (YENİ) ---
    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """Hurst Exponent: 0.5=Random, >0.5=Trend, <0.5=Mean Reversion"""
        try:
            lags = range(2, max_lag)
            # Standart sapma farklarının logaritması
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5

    # --- 6. MUM FORMASYONLARI (YENİ) ---
    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        body = np.abs(df['close'] - df['open'])
        range_len = df['high'] - df['low']
        
        # Doji
        df['is_doji'] = np.where(body <= (range_len * 0.1), 1, 0)
        
        # Hammer (Çekiç)
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        df['is_hammer'] = np.where((lower_shadow >= (body * 2)) & (df['is_doji'] == 0), 1, 0)
        
        # Engulfing (Yutan Ayı/Boğa)
        df['is_engulfing'] = np.where(
            (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & # Bullish
            (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)), 1, 0
        )
        return df[['is_doji', 'is_hammer', 'is_engulfing']]

    @classmethod
    def process_data(cls, raw_data: List[Dict]) -> Optional[pd.DataFrame]:
        """ANA İŞLEMCİ: Tüm indikatörleri hesaplar ve tek bir DataFrame döner."""
        if not raw_data: return None

        try:
            df = pd.DataFrame(raw_data)
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)

            # 1. Temel Hesaplamalar
            df['rsi'] = cls.calculate_rsi(df)
            df['roc'] = cls.calculate_roc(df['close'])
            df['adx'] = cls.calculate_adx(df)
            df['atr'] = cls.calculate_atr(df)
            df['mfi'] = cls.calculate_mfi(df)
            
            # 2. İleri Seviye (Ichimoku & Bollinger & VWAP)
            ichi = cls.calculate_ichimoku(df)
            df = pd.concat([df, ichi], axis=1)
            df['bb_width'] = cls.calculate_bollinger_width(df)
            df['vwap'] = cls.calculate_vwap(df)
            
            # 3. İstatistiksel (Z-Score & Log Return)
            df['z_score'] = cls.calculate_z_score(df['close'])
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

            # 4. Kaos Analizi (Hurst) - İşlemciyi yormamak için son 100 barlık pencerede hesaplanır
            df['hurst'] = df['close'].rolling(window=100).apply(lambda x: cls.calculate_hurst_exponent(x))

            # 5. Formasyonlar
            patterns = cls.detect_patterns(df)
            df = pd.concat([df, patterns], axis=1)

            # 6. AI Hafızası (Lag Features)
            # Geçmiş 8 periyoda kadar fiyat ve hacim hareketlerini kaydet
            for lag in [1, 2, 3, 5, 8]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'vol_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag) # RSI geçmişini de ekledik

            # NaN temizliği
            df.dropna(inplace=True)
            
            logger.info(f"Feature Engineering Complete. Shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"FEATURE ENGINEERING CRITICAL FAIL: {e}")
            return None