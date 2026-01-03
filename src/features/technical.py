# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - TECHNICAL FEATURES
==================================
Profesyonel feature engineering modülü.

50+ teknik indikatör hesaplar ve ML modeli için hazırlar.

KURAL: ASLA MOCK/FALLBACK VERİ YOK - SADECE GERÇEK HESAPLANMIŞ DEĞERLER!

Author: DEMIR AI Team
Date: 2026-01-03
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("FEATURES_TECHNICAL")


class TechnicalFeatures:
    """
    50+ teknik indikatör hesaplama sınıfı.
    
    Kategoriler:
    - Trend (EMA, SMA, MACD, ADX)
    - Momentum (RSI, Stochastic, Williams %R)
    - Volatility (BB, ATR, Keltner Channel)
    - Volume (OBV, MFI, VWAP)
    - Pattern (Candlestick patterns)
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm feature'ları hesapla ve DataFrame'e ekle."""
        logger.info(f"Calculating features for {len(df)} candles...")
        
        # Kopyala (orijinali bozmamak için)
        result = df.copy()
        
        # === TREND INDICATORS ===
        result = self._add_ema(result, [9, 21, 55, 200])
        result = self._add_sma(result, [20, 50, 200])
        result = self._add_macd(result)
        result = self._add_adx(result)
        
        # === MOMENTUM INDICATORS ===
        result = self._add_rsi(result, [7, 14, 21])
        result = self._add_stochastic(result)
        result = self._add_williams_r(result)
        result = self._add_roc(result)
        result = self._add_momentum(result)
        
        # === VOLATILITY INDICATORS ===
        result = self._add_bollinger(result)
        result = self._add_atr(result)
        result = self._add_keltner(result)
        result = self._add_volatility_ratio(result)
        
        # === VOLUME INDICATORS ===
        result = self._add_obv(result)
        result = self._add_mfi(result)
        result = self._add_volume_ema(result)
        result = self._add_volume_ratio(result)
        
        # === PRICE FEATURES ===
        result = self._add_price_features(result)
        result = self._add_candle_patterns(result)
        
        # === TREND FEATURES ===
        result = self._add_trend_features(result)
        
        # === LABELS (Hedef Değişkenler) ===
        result = self._add_labels(result)
        
        # NaN'ları temizle (ilk N mum)
        initial_len = len(result)
        result = result.dropna()
        logger.info(f"Features calculated: {initial_len} → {len(result)} rows (NaN removed)")
        
        return result
    
    # === TREND INDICATORS ===
    
    def _add_ema(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Exponential Moving Average."""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # EMA Cross signals
        if 9 in periods and 21 in periods:
            df['ema_9_21_cross'] = (df['ema_9'] > df['ema_21']).astype(int)
            df['ema_9_21_diff'] = (df['ema_9'] - df['ema_21']) / df['close'] * 100
        
        return df
    
    def _add_sma(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Simple Moving Average."""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD (12, 26, 9)."""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        return df
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index (Trend Strength)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    # === MOMENTUM INDICATORS ===
    
    def _add_rsi(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """Relative Strength Index."""
        for period in periods:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        return df
    
    def _add_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Rate of Change."""
        df['roc'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return df
    
    def _add_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Momentum indicator."""
        df['momentum'] = df['close'] - df['close'].shift(period)
        return df
    
    # === VOLATILITY INDICATORS ===
    
    def _add_bollinger(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range."""
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr'] = tr.rolling(window=period).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        return df
    
    def _add_keltner(self, df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
        """Keltner Channel."""
        ema = df['close'].ewm(span=period, adjust=False).mean()
        atr = df['atr'] if 'atr' in df.columns else self._add_atr(df.copy())['atr']
        
        df['keltner_upper'] = ema + (mult * atr)
        df['keltner_middle'] = ema
        df['keltner_lower'] = ema - (mult * atr)
        
        return df
    
    def _add_volatility_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatilite oranı (son 14 gün / son 100 gün)."""
        atr_14 = df['close'].diff().abs().rolling(14).mean()
        atr_100 = df['close'].diff().abs().rolling(100).mean()
        df['volatility_ratio'] = atr_14 / atr_100
        return df
    
    # === VOLUME INDICATORS ===
    
    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """On Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['obv'] = obv
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
        
        return df
    
    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(len(df)):
            if i == 0:
                positive_flow.append(0)
                negative_flow.append(0)
            else:
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.append(raw_money_flow.iloc[i])
                    negative_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(raw_money_flow.iloc[i])
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        df['mfi'] = mfi.values
        
        return df
    
    def _add_volume_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume EMA."""
        df['volume_ema_20'] = df['volume'].ewm(span=20, adjust=False).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ema_20']
        return df
    
    def _add_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Taker buy volume / Total volume ratio."""
        df['taker_ratio'] = df['taker_buy_volume'] / df['volume']
        return df
    
    # === PRICE FEATURES ===
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat bazlı özellikler."""
        # Returns
        df['return_1'] = df['close'].pct_change(1) * 100
        df['return_5'] = df['close'].pct_change(5) * 100
        df['return_15'] = df['close'].pct_change(15) * 100
        df['return_60'] = df['close'].pct_change(60) * 100
        df['return_240'] = df['close'].pct_change(240) * 100  # 4 saat
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Close position in candle
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        return df
    
    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mum kalıpları."""
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        
        # Doji (küçük body)
        df['is_doji'] = (body / range_hl < 0.1).astype(int)
        
        # Bullish engulfing
        prev_bearish = (df['close'].shift(1) < df['open'].shift(1))
        curr_bullish = (df['close'] > df['open'])
        engulf = (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
        df['bullish_engulfing'] = (prev_bearish & curr_bullish & engulf).astype(int)
        
        # Bearish engulfing
        prev_bullish = (df['close'].shift(1) > df['open'].shift(1))
        curr_bearish = (df['close'] < df['open'])
        engulf_bear = (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
        df['bearish_engulfing'] = (prev_bullish & curr_bearish & engulf_bear).astype(int)
        
        # Hammer (long lower wick, small body at top)
        lower_wick = df['open'].combine(df['close'], min) - df['low']
        upper_wick = df['high'] - df['open'].combine(df['close'], max)
        df['is_hammer'] = ((lower_wick > body * 2) & (upper_wick < body * 0.5)).astype(int)
        
        return df
    
    # === TREND FEATURES ===
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend özellikleri."""
        # Price vs EMA
        if 'ema_21' in df.columns:
            df['above_ema_21'] = (df['close'] > df['ema_21']).astype(int)
            df['dist_from_ema_21'] = (df['close'] - df['ema_21']) / df['ema_21'] * 100
        
        if 'ema_200' in df.columns:
            df['above_ema_200'] = (df['close'] > df['ema_200']).astype(int)
            df['dist_from_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200'] * 100
        
        # Trend strength
        if 'adx' in df.columns:
            df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Consecutive up/down
        df['up_candle'] = (df['close'] > df['open']).astype(int)
        df['consecutive_up'] = df['up_candle'].rolling(5).sum()
        df['consecutive_down'] = (1 - df['up_candle']).rolling(5).sum()
        
        return df
    
    # === LABELS (HEDEF DEĞİŞKENLER) ===
    
    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model için hedef değişkenler (GERÇEK GELECEKTEKİ FİYAT!)."""
        # Gelecekteki fiyat değişimi
        df['future_return_60'] = df['close'].shift(-60).pct_change(60) * 100  # 1 saat sonra
        df['future_return_240'] = df['close'].shift(-240).pct_change(240) * 100  # 4 saat sonra
        
        # Binary labels (UP/DOWN)
        df['label_1h'] = (df['close'].shift(-60) > df['close']).astype(int)
        df['label_4h'] = (df['close'].shift(-240) > df['close']).astype(int)
        
        # Triple labels (UP/DOWN/NEUTRAL) - %0.5'ten az değişim = NEUTRAL
        def classify_move(pct):
            if pct > 0.5:
                return 2  # UP
            elif pct < -0.5:
                return 0  # DOWN
            else:
                return 1  # NEUTRAL
        
        df['label_4h_triple'] = df['future_return_240'].apply(lambda x: classify_move(x) if pd.notna(x) else np.nan)
        
        return df


# Singleton
_tech_features: Optional[TechnicalFeatures] = None


def get_technical_features() -> TechnicalFeatures:
    """Get or create TechnicalFeatures singleton."""
    global _tech_features
    if _tech_features is None:
        _tech_features = TechnicalFeatures()
    return _tech_features


# Test
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 90000,
        'high': np.random.randn(1000).cumsum() + 90100,
        'low': np.random.randn(1000).cumsum() + 89900,
        'close': np.random.randn(1000).cumsum() + 90000,
        'volume': np.random.rand(1000) * 1000000,
        'taker_buy_volume': np.random.rand(1000) * 500000
    })
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Calculate features
    features = get_technical_features()
    result = features.calculate_all(df)
    
    print(f"\n📊 Features generated: {len(result.columns)} columns")
    print(f"📊 Sample size: {len(result)} rows")
    print(f"\n📋 Feature list:")
    for col in result.columns:
        print(f"  - {col}")
