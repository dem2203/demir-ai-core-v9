import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("ULTIMATE_FEATURE_ENGINEERING")

class FeatureEngineer:
    """
    DEMIR AI V15.0 - SUPERHUMAN VISION
    Eklenen: Hurst Exponent (Fraktal), Volume Profile (VAH/VAL), ATR Bands
    """

    # --- KLASİK İNDİKATÖRLER ---
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
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    # --- YENİ: SUPERHUMAN İNDİKATÖRLER ---

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Piyasanın 'Trend' mi yoksa 'Mean Reverting' (Ortalamaya Dönüş) mi olduğunu anlar.
        H < 0.5: Mean Reverting (Tersine işlem yap)
        H > 0.5: Trending (Trendi takip et)
        H = 0.5: Random Walk (İşlem yapma)
        """
        try:
            if len(series) < max_lag + 2: return 0.5
            lags = range(2, max_lag)
            # Standart sapma farklarını hesapla
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            if len(tau) < 2: return 0.5
            # Log-Log grafiğinin eğimi Hurst üssünü verir
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            val = poly[0] * 2.0
            if np.isnan(val) or np.isinf(val): return 0.5
            return val
        except: return 0.5

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
        """
        Son 'lookback' mumdaki Hacim Profilini çıkarır.
        VAH (Value Area High) ve VAL (Value Area Low) seviyelerini belirler.
        Bu seviyeler kurumsal destek/dirençtir.
        """
        try:
            # Son N mumu al
            subset = df.tail(lookback).copy()
            if subset.empty: return pd.DataFrame({'vah': 0, 'val': 0, 'poc': 0}, index=df.index)
            
            # Fiyat aralığını dilimlere böl (Price Buckets)
            price_min = subset['low'].min()
            price_max = subset['high'].max()
            if price_min == price_max: return pd.DataFrame({'vah': price_max, 'val': price_min, 'poc': price_min}, index=df.index)
            
            bins = np.linspace(price_min, price_max, num=50)
            # Hangi fiyatta ne kadar hacim dönmüş?
            # Yaklaşık hesap: Her mumun hacmini o mumun ortalama fiyatına atıyoruz
            subset['avg_price'] = (subset['high'] + subset['low'] + subset['close']) / 3
            volume_profile, bin_edges = np.histogram(subset['avg_price'], bins=bins, weights=subset['volume'])
            
            # POC (Point of Control): En çok hacim dönen fiyat
            max_vol_idx = np.argmax(volume_profile)
            poc = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
            
            # Value Area (%70 hacmin döndüğü alan)
            total_volume = np.sum(volume_profile)
            value_area_vol = total_volume * 0.70
            
            # Merkezden dışa doğru toplayarak %70'i bul
            sorted_indices = np.argsort(volume_profile)[::-1] # Büyükten küçüğe
            cum_vol = 0
            va_indices = []
            for idx in sorted_indices:
                cum_vol += volume_profile[idx]
                va_indices.append(idx)
                if cum_vol >= value_area_vol:
                    break
            
            # VAH ve VAL hesapla
            va_prices = [(bin_edges[i] + bin_edges[i+1])/2 for i in va_indices]
            vah = max(va_prices)
            val = min(va_prices)
            
            # Bu değerleri tüm DataFrame'e yay (Son durum olarak)
            # Gerçek zamanlı hesaplama için rolling window gerekir ama çok ağırdır.
            # Şimdilik son durumu statik olarak ekliyoruz.
            return pd.DataFrame({'vah': vah, 'val': val, 'poc': poc}, index=df.index)
            
        except Exception as e:
            logger.error(f"Volume Profile Error: {e}")
            return pd.DataFrame({'vah': 0, 'val': 0, 'poc': 0}, index=df.index)

    @staticmethod
    def calculate_atr_bands(df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Keltner Kanalları benzeri, ATR tabanlı dinamik destek/direnç bantları.
        """
        atr = FeatureEngineer.calculate_atr(df, period)
        ema = df['close'].ewm(span=period, adjust=False).mean()
        upper = ema + (atr * multiplier)
        lower = ema - (atr * multiplier)
        return pd.DataFrame({'atr_upper': upper, 'atr_lower': lower})

    @staticmethod
    def detect_anomalies(df: pd.DataFrame) -> pd.Series:
        """
        Hacim verisindeki anormal artışları (Balina Aktivitesi) tespit eder.
        """
        try:
            model = IsolationForest(contamination=0.05, random_state=42)
            df['vol_anomaly'] = model.fit_predict(df[['volume']])
            return df['vol_anomaly']
        except:
            return pd.Series(1, index=df.index)

    # ==========================================
    # YENİ: PATTERN DETECTION FOR LSTM
    # ==========================================
    
    @staticmethod
    def detect_candlestick_patterns_numerical(df: pd.DataFrame) -> pd.DataFrame:
        """
        Mum formasyonlarını sayısal değerlere çevir.
        LSTM bu değerleri görerek pattern'ları ÖĞRENIR.
        
        Çıktı: Her satır için pattern skorları
        - Bullish patterns: Pozitif değerler (0 to 1)
        - Bearish patterns: Negatif değerler (-1 to 0)
        """
        n = len(df)
        
        # Pattern skorları
        bullish_pattern = np.zeros(n)
        bearish_pattern = np.zeros(n)
        reversal_signal = np.zeros(n)
        continuation_signal = np.zeros(n)
        
        for i in range(3, n):
            o, h, l, c = df['open'].iloc[i], df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
            po, ph, pl, pc = df['open'].iloc[i-1], df['high'].iloc[i-1], df['low'].iloc[i-1], df['close'].iloc[i-1]
            p2o, p2c = df['open'].iloc[i-2], df['close'].iloc[i-2]
            
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l
            
            if total_range == 0: continue
            
            # DOJI (reversal warning)
            if body < total_range * 0.1:
                reversal_signal[i] = 0.6
            
            # HAMMER (bullish reversal)
            if body > 0 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if c > o:
                    bullish_pattern[i] = 0.75
                    reversal_signal[i] = 0.7
            
            # SHOOTING STAR (bearish reversal)
            if body > 0 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                if c < o:
                    bearish_pattern[i] = 0.75
                    reversal_signal[i] = 0.7
            
            # BULLISH ENGULFING
            if pc < po and c > o:  # Önceki bearish, şimdiki bullish
                if o < pc and c > po:  # Yutuyor
                    bullish_pattern[i] = 0.85
                    reversal_signal[i] = 0.8
            
            # BEARISH ENGULFING
            if pc > po and c < o:  # Önceki bullish, şimdiki bearish
                if o > pc and c < po:  # Yutuyor
                    bearish_pattern[i] = 0.85
                    reversal_signal[i] = 0.8
            
            # THREE WHITE SOLDIERS (very strong bullish)
            if (p2c > p2o and pc > po and c > o and  # Üç bullish mum
                pc > p2c and c > pc):  # Her biri öncekinden yüksek
                bullish_pattern[i] = 0.95
                continuation_signal[i] = 0.9
            
            # THREE BLACK CROWS (very strong bearish)
            if (p2c < p2o and pc < po and c < o and  # Üç bearish mum
                pc < p2c and c < pc):  # Her biri öncekinden düşük
                bearish_pattern[i] = 0.95
                continuation_signal[i] = 0.9
            
            # MORNING STAR (bullish reversal)
            if (p2c < p2o and  # İlk mum bearish
                abs(pc - po) < abs(p2c - p2o) * 0.3 and  # Orta mum küçük
                c > o and c > (p2o + p2c) / 2):  # Son mum bullish ve yükseliş
                bullish_pattern[i] = 0.9
                reversal_signal[i] = 0.85
            
            # EVENING STAR (bearish reversal)
            if (p2c > p2o and  # İlk mum bullish
                abs(pc - po) < abs(p2c - p2o) * 0.3 and  # Orta mum küçük
                c < o and c < (p2o + p2c) / 2):  # Son mum bearish ve düşüş
                bearish_pattern[i] = 0.9
                reversal_signal[i] = 0.85
        
        return pd.DataFrame({
            'bullish_pattern': bullish_pattern,
            'bearish_pattern': bearish_pattern,
            'reversal_signal': reversal_signal,
            'continuation_signal': continuation_signal,
            'pattern_strength': np.maximum(bullish_pattern, bearish_pattern)
        }, index=df.index)
    
    @staticmethod
    def detect_chart_patterns_numerical(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
        """
        Grafik formasyonlarını sayısal değerlere çevir.
        
        Çıktı: 
        - double_top_signal: 0-1 (1 = aktif double top)
        - double_bottom_signal: 0-1 (1 = aktif double bottom)
        - trend_direction: -1 (down), 0 (sideways), 1 (up)
        """
        n = len(df)
        
        double_top = np.zeros(n)
        double_bottom = np.zeros(n)
        head_shoulders = np.zeros(n)
        trend_direction = np.zeros(n)
        
        for i in range(lookback, n):
            subset = df.iloc[i-lookback:i+1]
            highs = subset['high'].values
            lows = subset['low'].values
            closes = subset['close'].values
            
            # Trend direction
            if closes[-1] > closes[0]:
                trend_direction[i] = min((closes[-1] - closes[0]) / closes[0] * 20, 1.0)
            else:
                trend_direction[i] = max((closes[-1] - closes[0]) / closes[0] * 20, -1.0)
            
            # Find swing highs/lows
            swing_highs = []
            swing_lows = []
            
            for j in range(5, len(highs) - 5):
                if highs[j] == max(highs[j-5:j+6]):
                    swing_highs.append((j, highs[j]))
                if lows[j] == min(lows[j-5:j+6]):
                    swing_lows.append((j, lows[j]))
            
            # Double Top Detection
            if len(swing_highs) >= 2:
                h1, h2 = swing_highs[-2:]
                if abs(h1[1] - h2[1]) / h1[1] < 0.02:  # %2 within
                    double_top[i] = 0.8
            
            # Double Bottom Detection
            if len(swing_lows) >= 2:
                l1, l2 = swing_lows[-2:]
                if abs(l1[1] - l2[1]) / l1[1] < 0.02:
                    double_bottom[i] = 0.8
            
            # Head and Shoulders (simplified)
            if len(swing_highs) >= 3:
                h1, h2, h3 = swing_highs[-3:]
                if h2[1] > h1[1] and h2[1] > h3[1]:  # Orta tepe en yüksek
                    if abs(h1[1] - h3[1]) / h1[1] < 0.03:  # Omuzlar eşit
                        head_shoulders[i] = 0.9
        
        return pd.DataFrame({
            'double_top_signal': double_top,
            'double_bottom_signal': double_bottom,
            'head_shoulders_signal': head_shoulders,
            'trend_direction': trend_direction
        }, index=df.index)

    @staticmethod
    def merge_crypto_and_macro(crypto_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        if macro_df is None or macro_df.empty: return crypto_df
        crypto_df = crypto_df.sort_values('timestamp')
        macro_df = macro_df.sort_values('timestamp')
        merged_df = pd.merge_asof(crypto_df, macro_df, on='timestamp', direction='backward')
        merged_df = merged_df.ffill().bfill()
        return merged_df

    @classmethod
    def process_data(cls, raw_data: List[Dict]) -> Optional[pd.DataFrame]:
        if not raw_data: return None
        try:
            df = pd.DataFrame(raw_data)
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)

            # Temel İndikatörler
            df['rsi'] = cls.calculate_rsi(df)
            macd, signal = cls.calculate_macd(df)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['atr'] = cls.calculate_atr(df)
            
            # Bollinger Bands (bb_width için gerekli)
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # Normalized width
            
            # ADX (Average Directional Index) - Trend gücü ölçer
            adx_period = 14
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs() * -1
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            tr = cls.calculate_atr(df) * adx_period  # True Range
            plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / tr.replace(0, 1))
            minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / tr.replace(0, 1))
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
            df['adx'] = dx.rolling(window=adx_period).mean().fillna(25)  # Default 25 (neutral)
            
            # VWAP (Volume Weighted Average Price) - Kurumsal referans fiyat
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])  # NaN varsa close kullan
            
            # --- SUPERHUMAN KATMANI ---
            
            # 1. Hurst Exponent (Rolling)
            # Son 100 mumluk pencerede hesapla
            df['hurst'] = df['close'].rolling(window=100).apply(lambda x: cls.calculate_hurst_exponent(x))
            
            # 2. Volume Profile (Son 24 mum - Günlük Profil)
            vp_df = cls.calculate_volume_profile(df, lookback=24)
            df = pd.concat([df, vp_df], axis=1)
            
            # 3. ATR Bands
            bands = cls.calculate_atr_bands(df)
            df = pd.concat([df, bands], axis=1)
            
            # 4. Anomali Tespiti
            df['vol_anomaly'] = cls.detect_anomalies(df)
            
            # --- YENİ: PATTERN DETECTION FOR LSTM ---
            # AI artık pattern'ları "görüyor"!
            
            # 5. Candlestick Pattern Signals
            candle_patterns = cls.detect_candlestick_patterns_numerical(df)
            df = pd.concat([df, candle_patterns], axis=1)
            
            # 6. Chart Pattern Signals
            chart_patterns = cls.detect_chart_patterns_numerical(df)
            df = pd.concat([df, chart_patterns], axis=1)

            # Veri Temizliği
            df = df.iloc[100:] # Rolling windowlar otursun diye baştan kes
            df = df.ffill() 
            df.fillna(0, inplace=True)
            
            logger.info(f"Feature Engineering Complete. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"FEATURE ENGINEERING FAIL: {e}")
            return None

