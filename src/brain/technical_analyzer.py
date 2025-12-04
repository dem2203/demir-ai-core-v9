"""
DEMIR AI - ADVANCED TECHNICAL ANALYSIS ENGINE
Candlestick, Chart Patterns, Divergence, Fibonacci, Volume, Pivots

Tüm teknik analiz güçleri tek modülde
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("TECHNICAL_ANALYSIS")


class CandlestickPattern(Enum):
    """Mum formasyonları"""
    DOJI = "DOJI"
    HAMMER = "HAMMER"
    INVERTED_HAMMER = "INVERTED_HAMMER"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"
    THREE_WHITE_SOLDIERS = "THREE_WHITE_SOLDIERS"
    THREE_BLACK_CROWS = "THREE_BLACK_CROWS"
    SHOOTING_STAR = "SHOOTING_STAR"
    HANGING_MAN = "HANGING_MAN"
    PIERCING_LINE = "PIERCING_LINE"
    DARK_CLOUD = "DARK_CLOUD"
    NONE = "NONE"


class ChartPattern(Enum):
    """Grafik formasyonları"""
    HEAD_SHOULDERS = "HEAD_AND_SHOULDERS"
    INV_HEAD_SHOULDERS = "INVERSE_HEAD_AND_SHOULDERS"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    TRIPLE_TOP = "TRIPLE_TOP"
    TRIPLE_BOTTOM = "TRIPLE_BOTTOM"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRIC_TRIANGLE = "SYMMETRIC_TRIANGLE"
    BULL_FLAG = "BULL_FLAG"
    BEAR_FLAG = "BEAR_FLAG"
    RISING_WEDGE = "RISING_WEDGE"
    FALLING_WEDGE = "FALLING_WEDGE"
    CUP_AND_HANDLE = "CUP_AND_HANDLE"
    NONE = "NONE"


class TechnicalAnalyzer:
    """
    GELİŞMİŞ TEKNİK ANALİZ MOTORU
    
    1. Candlestick Pattern Recognition
    2. Chart Pattern Detection
    3. Divergence Detection (RSI/MACD)
    4. Fibonacci Levels
    5. Volume Analysis
    6. Pivot Points
    7. Trend Line Detection
    """
    
    def __init__(self):
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618]
    
    # ==========================================
    # CANDLESTICK PATTERN RECOGNITION
    # ==========================================
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Son 10 mum için pattern tarama.
        """
        if len(df) < 10:
            return []
        
        patterns = []
        recent = df.tail(10).reset_index(drop=True)
        
        for i in range(2, len(recent)):
            candle = recent.iloc[i]
            prev = recent.iloc[i-1]
            prev2 = recent.iloc[i-2] if i >= 2 else None
            
            o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
            body = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            total_range = h - l
            
            # DOJI: Çok küçük gövde
            if body < total_range * 0.1 and total_range > 0:
                patterns.append({
                    'pattern': CandlestickPattern.DOJI.value,
                    'index': i,
                    'signal': 'REVERSAL_WARNING',
                    'strength': 0.6
                })
            
            # HAMMER: Küçük gövde üstte, uzun alt fitil
            if lower_shadow > body * 2 and upper_shadow < body * 0.5 and body > 0:
                if c > o:  # Bullish
                    patterns.append({
                        'pattern': CandlestickPattern.HAMMER.value,
                        'index': i,
                        'signal': 'BULLISH_REVERSAL',
                        'strength': 0.75
                    })
            
            # SHOOTING STAR: Küçük gövde altta, uzun üst fitil
            if upper_shadow > body * 2 and lower_shadow < body * 0.5 and body > 0:
                if c < o:  # Bearish
                    patterns.append({
                        'pattern': CandlestickPattern.SHOOTING_STAR.value,
                        'index': i,
                        'signal': 'BEARISH_REVERSAL',
                        'strength': 0.75
                    })
            
            # BULLISH ENGULFING
            if i >= 1:
                po, pc = prev['open'], prev['close']
                if pc < po and c > o:  # Önceki bearish, şimdiki bullish
                    if o < pc and c > po:  # Şimdiki öncekini yutuyor
                        patterns.append({
                            'pattern': CandlestickPattern.BULLISH_ENGULFING.value,
                            'index': i,
                            'signal': 'STRONG_BULLISH',
                            'strength': 0.85
                        })
            
            # BEARISH ENGULFING
            if i >= 1:
                po, pc = prev['open'], prev['close']
                if pc > po and c < o:  # Önceki bullish, şimdiki bearish
                    if o > pc and c < po:  # Şimdiki öncekini yutuyor
                        patterns.append({
                            'pattern': CandlestickPattern.BEARISH_ENGULFING.value,
                            'index': i,
                            'signal': 'STRONG_BEARISH',
                            'strength': 0.85
                        })
            
            # MORNING STAR (3 mum)
            if i >= 2 and prev2 is not None:
                p2o, p2c = prev2['open'], prev2['close']
                po, pc = prev['open'], prev['close']
                
                if (p2c < p2o and  # İlk mum bearish
                    abs(pc - po) < abs(p2c - p2o) * 0.3 and  # Orta mum küçük
                    c > o and c > (p2o + p2c) / 2):  # Son mum bullish ve yükseliş
                    patterns.append({
                        'pattern': CandlestickPattern.MORNING_STAR.value,
                        'index': i,
                        'signal': 'STRONG_BULLISH_REVERSAL',
                        'strength': 0.9
                    })
            
            # EVENING STAR (3 mum)
            if i >= 2 and prev2 is not None:
                p2o, p2c = prev2['open'], prev2['close']
                po, pc = prev['open'], prev['close']
                
                if (p2c > p2o and  # İlk mum bullish
                    abs(pc - po) < abs(p2c - p2o) * 0.3 and  # Orta mum küçük
                    c < o and c < (p2o + p2c) / 2):  # Son mum bearish ve düşüş
                    patterns.append({
                        'pattern': CandlestickPattern.EVENING_STAR.value,
                        'index': i,
                        'signal': 'STRONG_BEARISH_REVERSAL',
                        'strength': 0.9
                    })
            
            # THREE WHITE SOLDIERS
            if i >= 2 and prev2 is not None:
                p2o, p2c = prev2['open'], prev2['close']
                po, pc = prev['open'], prev['close']
                
                if (p2c > p2o and pc > po and c > o and  # Üç bullish mum
                    pc > p2c and c > pc and  # Her biri öncekinden yüksek kapatıyor
                    o > po and po > p2o):  # Her biri öncekinin içinden açılıyor
                    patterns.append({
                        'pattern': CandlestickPattern.THREE_WHITE_SOLDIERS.value,
                        'index': i,
                        'signal': 'VERY_STRONG_BULLISH',
                        'strength': 0.95
                    })
            
            # THREE BLACK CROWS
            if i >= 2 and prev2 is not None:
                p2o, p2c = prev2['open'], prev2['close']
                po, pc = prev['open'], prev['close']
                
                if (p2c < p2o and pc < po and c < o and  # Üç bearish mum
                    pc < p2c and c < pc and  # Her biri öncekinden düşük kapatıyor
                    o < po and po < p2o):  # Her biri öncekinin içinden açılıyor
                    patterns.append({
                        'pattern': CandlestickPattern.THREE_BLACK_CROWS.value,
                        'index': i,
                        'signal': 'VERY_STRONG_BEARISH',
                        'strength': 0.95
                    })
        
        if patterns:
            logger.info(f"🕯️ Found {len(patterns)} candlestick patterns")
        
        return patterns
    
    # ==========================================
    # CHART PATTERN DETECTION
    # ==========================================
    
    def detect_chart_patterns(self, df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """
        Grafik formasyonlarını tespit et.
        """
        if len(df) < lookback:
            return []
        
        patterns = []
        recent = df.tail(lookback).reset_index(drop=True)
        
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        
        # Swing High/Low bul
        swing_highs = self._find_swing_points(highs, is_high=True)
        swing_lows = self._find_swing_points(lows, is_high=False)
        
        # DOUBLE TOP
        if len(swing_highs) >= 2:
            last_two = swing_highs[-2:]
            price_diff = abs(last_two[0]['price'] - last_two[1]['price']) / last_two[0]['price']
            if price_diff < 0.02:  # %2'den az fark
                patterns.append({
                    'pattern': ChartPattern.DOUBLE_TOP.value,
                    'signal': 'BEARISH_REVERSAL',
                    'neckline': min(lows[last_two[0]['index']:last_two[1]['index']]),
                    'strength': 0.8
                })
        
        # DOUBLE BOTTOM
        if len(swing_lows) >= 2:
            last_two = swing_lows[-2:]
            price_diff = abs(last_two[0]['price'] - last_two[1]['price']) / last_two[0]['price']
            if price_diff < 0.02:
                patterns.append({
                    'pattern': ChartPattern.DOUBLE_BOTTOM.value,
                    'signal': 'BULLISH_REVERSAL',
                    'neckline': max(highs[last_two[0]['index']:last_two[1]['index']]),
                    'strength': 0.8
                })
        
        # HEAD AND SHOULDERS
        if len(swing_highs) >= 3:
            h1, h2, h3 = swing_highs[-3:]
            # Orta tepe en yüksek olmalı
            if h2['price'] > h1['price'] and h2['price'] > h3['price']:
                # Omuzlar yaklaşık eşit
                shoulder_diff = abs(h1['price'] - h3['price']) / h1['price']
                if shoulder_diff < 0.03:
                    patterns.append({
                        'pattern': ChartPattern.HEAD_SHOULDERS.value,
                        'signal': 'STRONG_BEARISH_REVERSAL',
                        'target': h2['price'] - (h2['price'] - min(lows[h1['index']:h3['index']])),
                        'strength': 0.9
                    })
        
        # INVERSE HEAD AND SHOULDERS
        if len(swing_lows) >= 3:
            l1, l2, l3 = swing_lows[-3:]
            # Orta dip en düşük olmalı
            if l2['price'] < l1['price'] and l2['price'] < l3['price']:
                shoulder_diff = abs(l1['price'] - l3['price']) / l1['price']
                if shoulder_diff < 0.03:
                    patterns.append({
                        'pattern': ChartPattern.INV_HEAD_SHOULDERS.value,
                        'signal': 'STRONG_BULLISH_REVERSAL',
                        'target': l2['price'] + (max(highs[l1['index']:l3['index']]) - l2['price']),
                        'strength': 0.9
                    })
        
        # ASCENDING TRIANGLE
        # Düz üst, yükselen altlar
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            resistance_flat = abs(swing_highs[-1]['price'] - swing_highs[-2]['price']) / swing_highs[-1]['price'] < 0.01
            support_rising = swing_lows[-1]['price'] > swing_lows[-2]['price']
            if resistance_flat and support_rising:
                patterns.append({
                    'pattern': ChartPattern.ASCENDING_TRIANGLE.value,
                    'signal': 'BULLISH_CONTINUATION',
                    'breakout_level': swing_highs[-1]['price'],
                    'strength': 0.75
                })
        
        # DESCENDING TRIANGLE
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            support_flat = abs(swing_lows[-1]['price'] - swing_lows[-2]['price']) / swing_lows[-1]['price'] < 0.01
            resistance_falling = swing_highs[-1]['price'] < swing_highs[-2]['price']
            if support_flat and resistance_falling:
                patterns.append({
                    'pattern': ChartPattern.DESCENDING_TRIANGLE.value,
                    'signal': 'BEARISH_CONTINUATION',
                    'breakdown_level': swing_lows[-1]['price'],
                    'strength': 0.75
                })
        
        # BULL FLAG
        # Güçlü yükseliş sonrası küçük düşen kanal
        if len(recent) >= 20:
            initial_move = (closes[10] - closes[0]) / closes[0]
            flag_move = (closes[-1] - closes[10]) / closes[10]
            if initial_move > 0.05 and -0.03 < flag_move < 0:  # %5+ rally sonrası küçük düzeltme
                patterns.append({
                    'pattern': ChartPattern.BULL_FLAG.value,
                    'signal': 'BULLISH_CONTINUATION',
                    'target': closes[-1] + (closes[10] - closes[0]),
                    'strength': 0.7
                })
        
        # BEAR FLAG
        if len(recent) >= 20:
            initial_move = (closes[10] - closes[0]) / closes[0]
            flag_move = (closes[-1] - closes[10]) / closes[10]
            if initial_move < -0.05 and 0 < flag_move < 0.03:
                patterns.append({
                    'pattern': ChartPattern.BEAR_FLAG.value,
                    'signal': 'BEARISH_CONTINUATION',
                    'target': closes[-1] - (closes[0] - closes[10]),
                    'strength': 0.7
                })
        
        if patterns:
            logger.info(f"📐 Found {len(patterns)} chart patterns")
        
        return patterns
    
    def _find_swing_points(self, prices: np.ndarray, is_high: bool = True, window: int = 5) -> List[Dict]:
        """Swing high/low noktalarını bul."""
        swings = []
        for i in range(window, len(prices) - window):
            if is_high:
                if prices[i] == max(prices[i-window:i+window+1]):
                    swings.append({'index': i, 'price': prices[i]})
            else:
                if prices[i] == min(prices[i-window:i+window+1]):
                    swings.append({'index': i, 'price': prices[i]})
        return swings
    
    # ==========================================
    # DIVERGENCE DETECTION
    # ==========================================
    
    def detect_divergences(self, df: pd.DataFrame) -> List[Dict]:
        """
        RSI ve MACD ile fiyat arasındaki uyumsuzlukları tespit et.
        
        Bullish Divergence: Fiyat düşük yapıyor, RSI yükseliyor
        Bearish Divergence: Fiyat yüksek yapıyor, RSI düşüyor
        """
        divergences = []
        
        if len(df) < 30:
            return divergences
        
        recent = df.tail(30).reset_index(drop=True)
        
        prices = recent['close'].values
        
        # RSI hesapla (eğer yoksa)
        if 'rsi' in recent.columns:
            rsi = recent['rsi'].values
        else:
            rsi = self._calculate_rsi(prices)
        
        # MACD hesapla (eğer yoksa)
        if 'macd' in recent.columns:
            macd = recent['macd'].values
        else:
            macd = self._calculate_macd(prices)
        
        # Son 2 swing low/high bul
        price_lows = self._find_swing_points(prices, is_high=False, window=3)
        price_highs = self._find_swing_points(prices, is_high=True, window=3)
        
        # BULLISH DIVERGENCE (RSI)
        if len(price_lows) >= 2:
            pl1, pl2 = price_lows[-2:]
            if pl2['price'] < pl1['price']:  # Fiyat düşük yapıyor
                rsi_at_l1 = rsi[pl1['index']]
                rsi_at_l2 = rsi[pl2['index']]
                if rsi_at_l2 > rsi_at_l1:  # RSI yükseliyor
                    divergences.append({
                        'type': 'BULLISH_RSI_DIVERGENCE',
                        'signal': 'BUY',
                        'strength': 0.8,
                        'price_low': pl2['price'],
                        'rsi': rsi_at_l2
                    })
        
        # BEARISH DIVERGENCE (RSI)
        if len(price_highs) >= 2:
            ph1, ph2 = price_highs[-2:]
            if ph2['price'] > ph1['price']:  # Fiyat yüksek yapıyor
                rsi_at_h1 = rsi[ph1['index']]
                rsi_at_h2 = rsi[ph2['index']]
                if rsi_at_h2 < rsi_at_h1:  # RSI düşüyor
                    divergences.append({
                        'type': 'BEARISH_RSI_DIVERGENCE',
                        'signal': 'SELL',
                        'strength': 0.8,
                        'price_high': ph2['price'],
                        'rsi': rsi_at_h2
                    })
        
        # MACD Divergences
        if len(price_lows) >= 2:
            pl1, pl2 = price_lows[-2:]
            if pl2['price'] < pl1['price']:
                macd_at_l1 = macd[pl1['index']]
                macd_at_l2 = macd[pl2['index']]
                if macd_at_l2 > macd_at_l1:
                    divergences.append({
                        'type': 'BULLISH_MACD_DIVERGENCE',
                        'signal': 'BUY',
                        'strength': 0.75
                    })
        
        if len(price_highs) >= 2:
            ph1, ph2 = price_highs[-2:]
            if ph2['price'] > ph1['price']:
                macd_at_h1 = macd[ph1['index']]
                macd_at_h2 = macd[ph2['index']]
                if macd_at_h2 < macd_at_h1:
                    divergences.append({
                        'type': 'BEARISH_MACD_DIVERGENCE',
                        'signal': 'SELL',
                        'strength': 0.75
                    })
        
        if divergences:
            logger.info(f"📊 Found {len(divergences)} divergences")
        
        return divergences
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI hesapla."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period
        
        rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> np.ndarray:
        """MACD hesapla."""
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        return ema12 - ema26
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """EMA hesapla."""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    # ==========================================
    # FIBONACCI LEVELS
    # ==========================================
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Otomatik Fibonacci seviyeleri.
        Swing high/low'dan hesaplar.
        """
        if len(df) < lookback:
            return {}
        
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        current_price = df['close'].iloc[-1]
        
        # Trend yönünü belirle
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        is_uptrend = low_idx < high_idx  # Önce dip, sonra tepe = uptrend
        
        diff = swing_high - swing_low
        
        levels = {}
        for fib in self.fib_levels:
            if is_uptrend:
                # Retracement (yukarıdan aşağı)
                levels[f'fib_{fib}'] = swing_high - diff * fib
            else:
                # Extension (aşağıdan yukarı)
                levels[f'fib_{fib}'] = swing_low + diff * fib
        
        # En yakın destek ve direnç
        sorted_levels = sorted(levels.values())
        support = max([l for l in sorted_levels if l < current_price], default=swing_low)
        resistance = min([l for l in sorted_levels if l > current_price], default=swing_high)
        
        result = {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'current_price': current_price,
            'is_uptrend': is_uptrend,
            'levels': levels,
            'nearest_support': support,
            'nearest_resistance': resistance,
            'fib_position': (current_price - swing_low) / diff if diff > 0 else 0.5
        }
        
        logger.info(f"📏 Fibonacci: Support=${support:,.0f} | Resistance=${resistance:,.0f}")
        return result
    
    # ==========================================
    # VOLUME ANALYSIS
    # ==========================================
    
    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """
        Hacim analizi.
        - Volume spike detection
        - Volume trend
        - Volume-Price divergence
        """
        if len(df) < 20:
            return {}
        
        recent = df.tail(50)
        volume = recent['volume'].values
        prices = recent['close'].values
        
        # Ortalama hacim
        avg_volume = np.mean(volume[:-5])
        current_volume = np.mean(volume[-5:])
        volume_ratio = current_volume / max(avg_volume, 1)
        
        # Volume spike
        last_volume = volume[-1]
        is_spike = last_volume > avg_volume * 2
        
        # Volume trend
        vol_sma_10 = np.mean(volume[-10:])
        vol_sma_20 = np.mean(volume[-20:])
        volume_trend = "INCREASING" if vol_sma_10 > vol_sma_20 else "DECREASING"
        
        # Price-Volume relationship
        price_change = (prices[-1] - prices[-5]) / prices[-5]
        vol_change = (current_volume - avg_volume) / avg_volume
        
        if price_change > 0 and vol_change > 0:
            pv_signal = "BULLISH_CONFIRMATION"
        elif price_change > 0 and vol_change < 0:
            pv_signal = "WEAK_RALLY"
        elif price_change < 0 and vol_change > 0:
            pv_signal = "DISTRIBUTION"
        elif price_change < 0 and vol_change < 0:
            pv_signal = "WEAK_DECLINE"
        else:
            pv_signal = "NEUTRAL"
        
        # Climax detection
        is_climax = last_volume > avg_volume * 3 and abs(prices[-1] - prices[-2]) / prices[-2] > 0.02
        climax_type = None
        if is_climax:
            climax_type = "BUYING_CLIMAX" if prices[-1] > prices[-2] else "SELLING_CLIMAX"
        
        result = {
            'avg_volume': avg_volume,
            'current_volume_ratio': volume_ratio,
            'is_spike': is_spike,
            'volume_trend': volume_trend,
            'price_volume_signal': pv_signal,
            'is_climax': is_climax,
            'climax_type': climax_type,
            'interpretation': self._interpret_volume(pv_signal, is_climax, climax_type)
        }
        
        logger.info(f"📊 Volume: {pv_signal} | Ratio: {volume_ratio:.1f}x | Climax: {climax_type or 'None'}")
        return result
    
    def _interpret_volume(self, pv_signal: str, is_climax: bool, climax_type: str) -> str:
        if is_climax and climax_type == "BUYING_CLIMAX":
            return "Potansiyel zirve, dikkatli olun"
        elif is_climax and climax_type == "SELLING_CLIMAX":
            return "Potansiyel dip, fırsat olabilir"
        elif pv_signal == "BULLISH_CONFIRMATION":
            return "Sağlıklı yükseliş, trend devam edebilir"
        elif pv_signal == "WEAK_RALLY":
            return "Zayıf yükseliş, geri çekilme riski"
        elif pv_signal == "DISTRIBUTION":
            return "Dağıtım fazı, düşüş beklentisi"
        elif pv_signal == "WEAK_DECLINE":
            return "Zayıf düşüş, dip yakın olabilir"
        return "Nötr piyasa"
    
    # ==========================================
    # PIVOT POINTS
    # ==========================================
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """
        Günlük pivot noktaları hesapla.
        Standard, Fibonacci ve Camarilla pivotları.
        """
        if len(df) < 2:
            return {}
        
        # Önceki günün verileri (son tamamlanmış bar)
        prev = df.iloc[-2]
        h, l, c = prev['high'], prev['low'], prev['close']
        
        # STANDARD PIVOTS
        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = 2 * pivot - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        
        # FIBONACCI PIVOTS
        fib_r1 = pivot + 0.382 * (h - l)
        fib_r2 = pivot + 0.618 * (h - l)
        fib_r3 = pivot + 1.0 * (h - l)
        fib_s1 = pivot - 0.382 * (h - l)
        fib_s2 = pivot - 0.618 * (h - l)
        fib_s3 = pivot - 1.0 * (h - l)
        
        # CAMARILLA PIVOTS
        cam_r1 = c + (h - l) * 1.1 / 12
        cam_r2 = c + (h - l) * 1.1 / 6
        cam_r3 = c + (h - l) * 1.1 / 4
        cam_r4 = c + (h - l) * 1.1 / 2
        cam_s1 = c - (h - l) * 1.1 / 12
        cam_s2 = c - (h - l) * 1.1 / 6
        cam_s3 = c - (h - l) * 1.1 / 4
        cam_s4 = c - (h - l) * 1.1 / 2
        
        current_price = df['close'].iloc[-1]
        
        # En yakın pivot seviyeleri
        all_levels = [s3, s2, s1, pivot, r1, r2, r3]
        support = max([l for l in all_levels if l < current_price], default=s3)
        resistance = min([l for l in all_levels if l > current_price], default=r3)
        
        result = {
            'pivot': pivot,
            'standard': {'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3},
            'fibonacci': {'r1': fib_r1, 'r2': fib_r2, 'r3': fib_r3, 's1': fib_s1, 's2': fib_s2, 's3': fib_s3},
            'camarilla': {'r1': cam_r1, 'r2': cam_r2, 'r3': cam_r3, 'r4': cam_r4, 
                         's1': cam_s1, 's2': cam_s2, 's3': cam_s3, 's4': cam_s4},
            'current_price': current_price,
            'nearest_support': support,
            'nearest_resistance': resistance,
            'position': 'ABOVE_PIVOT' if current_price > pivot else 'BELOW_PIVOT'
        }
        
        logger.info(f"📍 Pivot: ${pivot:,.0f} | Support: ${support:,.0f} | Resistance: ${resistance:,.0f}")
        return result
    
    # ==========================================
    # FULL TECHNICAL ANALYSIS
    # ==========================================
    
    def get_full_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Tüm teknik analizi birleştir.
        """
        candles = self.detect_candlestick_patterns(df)
        charts = self.detect_chart_patterns(df)
        divergences = self.detect_divergences(df)
        fibonacci = self.calculate_fibonacci_levels(df)
        volume = self.analyze_volume(df)
        pivots = self.calculate_pivot_points(df)
        
        # Composite Signal
        bullish_score = 0
        bearish_score = 0
        
        # Candlestick signals
        for c in candles:
            if 'BULLISH' in c.get('signal', ''):
                bullish_score += c.get('strength', 0.5) * 20
            elif 'BEARISH' in c.get('signal', ''):
                bearish_score += c.get('strength', 0.5) * 20
        
        # Chart pattern signals
        for p in charts:
            if 'BULLISH' in p.get('signal', ''):
                bullish_score += p.get('strength', 0.5) * 30
            elif 'BEARISH' in p.get('signal', ''):
                bearish_score += p.get('strength', 0.5) * 30
        
        # Divergence signals
        for d in divergences:
            if d.get('signal') == 'BUY':
                bullish_score += d.get('strength', 0.5) * 25
            elif d.get('signal') == 'SELL':
                bearish_score += d.get('strength', 0.5) * 25
        
        # Volume signal
        pv_signal = volume.get('price_volume_signal', '')
        if 'BULLISH' in pv_signal:
            bullish_score += 15
        elif 'DISTRIBUTION' in pv_signal or 'WEAK_RALLY' in pv_signal:
            bearish_score += 15
        
        # Pivot position
        if pivots.get('position') == 'ABOVE_PIVOT':
            bullish_score += 10
        else:
            bearish_score += 10
        
        # Final bias
        if bullish_score > bearish_score + 30:
            bias = "STRONG_BULLISH"
        elif bullish_score > bearish_score + 10:
            bias = "BULLISH"
        elif bearish_score > bullish_score + 30:
            bias = "STRONG_BEARISH"
        elif bearish_score > bullish_score + 10:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        
        return {
            'candlestick_patterns': candles,
            'chart_patterns': charts,
            'divergences': divergences,
            'fibonacci': fibonacci,
            'volume': volume,
            'pivot_points': pivots,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'technical_bias': bias,
            'active_patterns_count': len(candles) + len(charts) + len(divergences)
        }
