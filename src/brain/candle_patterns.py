# -*- coding: utf-8 -*-
"""
DEMIR AI - Candle Pattern Recognition
Mum formasyonları - Reversal ve continuation sinyalleri.

PHASE 86: Candle Pattern Recognition
- Engulfing (Bullish/Bearish)
- Doji (Kararsızlık)
- Hammer/Shooting Star
- Morning/Evening Star
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("CANDLE_PATTERNS")


class CandlePatternRecognizer:
    """
    Mum Formasyonu Tanıma Sistemi
    
    Teknik analiz mum formasyonlarını tespit eder.
    """
    
    BINANCE_API = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.min_body_ratio = 0.6  # Body/Total ratio for strong candle
        self.doji_ratio = 0.1  # Body/Total ratio for doji
        logger.info("✅ Candle Pattern Recognizer initialized")
    
    def analyze_patterns(self, symbol: str = 'BTCUSDT', interval: str = '15m', limit: int = 10) -> Dict:
        """
        Son mumlarda pattern analizi yap.
        
        Returns:
            {
                'patterns_found': ['BULLISH_ENGULFING', 'DOJI'],
                'latest_pattern': 'BULLISH_ENGULFING',
                'direction': 'LONG',
                'confidence': 65,
                'candle_details': {...}
            }
        """
        try:
            # Kline verisi al
            resp = requests.get(
                f"{self.BINANCE_API}/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': limit},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            klines = resp.json()
            candles = self._parse_candles(klines)
            
            if len(candles) < 3:
                return self._empty_result()
            
            # Pattern tespiti
            patterns = []
            
            # 1. Engulfing Pattern
            engulfing = self._check_engulfing(candles[-2], candles[-1])
            if engulfing:
                patterns.append(engulfing)
            
            # 2. Doji Pattern
            doji = self._check_doji(candles[-1])
            if doji:
                patterns.append(doji)
            
            # 3. Hammer / Shooting Star
            hammer = self._check_hammer(candles[-1])
            if hammer:
                patterns.append(hammer)
            
            # 4. Morning/Evening Star (3 mum pattern)
            if len(candles) >= 3:
                star = self._check_star_pattern(candles[-3], candles[-2], candles[-1])
                if star:
                    patterns.append(star)
            
            if not patterns:
                return self._empty_result()
            
            # En güçlü pattern'ı seç
            best_pattern = max(patterns, key=lambda x: x.get('confidence', 0))
            
            return {
                'patterns_found': [p['name'] for p in patterns],
                'latest_pattern': best_pattern['name'],
                'direction': best_pattern['direction'],
                'confidence': best_pattern['confidence'],
                'pattern_count': len(patterns),
                'candle_details': {
                    'current': candles[-1],
                    'previous': candles[-2]
                },
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Candle pattern analysis failed: {e}")
            return self._empty_result()
    
    def _parse_candles(self, klines: list) -> List[Dict]:
        """Kline verisini candle dict'e çevir."""
        candles = []
        for k in klines:
            o, h, l, c = float(k[1]), float(k[2]), float(k[3]), float(k[4])
            vol = float(k[5])
            
            body = abs(c - o)
            total_range = h - l if h != l else 0.0001
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            
            candles.append({
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': vol,
                'body': body,
                'range': total_range,
                'body_ratio': body / total_range,
                'upper_wick': upper_wick,
                'lower_wick': lower_wick,
                'is_bullish': c > o,
                'is_bearish': c < o
            })
        return candles
    
    def _check_engulfing(self, prev: dict, curr: dict) -> Optional[Dict]:
        """Engulfing pattern tespit."""
        if prev['is_bullish'] and curr['is_bearish']:
            # Bearish Engulfing
            if curr['body'] > prev['body'] * 1.2 and curr['close'] < prev['open']:
                return {
                    'name': 'BEARISH_ENGULFING',
                    'direction': 'SHORT',
                    'confidence': 70
                }
        elif prev['is_bearish'] and curr['is_bullish']:
            # Bullish Engulfing
            if curr['body'] > prev['body'] * 1.2 and curr['close'] > prev['open']:
                return {
                    'name': 'BULLISH_ENGULFING',
                    'direction': 'LONG',
                    'confidence': 70
                }
        return None
    
    def _check_doji(self, candle: dict) -> Optional[Dict]:
        """Doji tespit."""
        if candle['body_ratio'] < self.doji_ratio:
            # Doji = kararsızlık, yön için ek gösterge gerekli
            return {
                'name': 'DOJI',
                'direction': 'NEUTRAL',
                'confidence': 50
            }
        return None
    
    def _check_hammer(self, candle: dict) -> Optional[Dict]:
        """Hammer / Shooting Star tespit."""
        body = candle['body']
        upper = candle['upper_wick']
        lower = candle['lower_wick']
        
        # Hammer: Küçük body üstte, uzun alt fitil
        if lower > body * 2 and upper < body * 0.5:
            return {
                'name': 'HAMMER',
                'direction': 'LONG',
                'confidence': 65
            }
        
        # Shooting Star: Küçük body altta, uzun üst fitil
        if upper > body * 2 and lower < body * 0.5:
            return {
                'name': 'SHOOTING_STAR',
                'direction': 'SHORT',
                'confidence': 65
            }
        
        return None
    
    def _check_star_pattern(self, first: dict, middle: dict, last: dict) -> Optional[Dict]:
        """Morning Star / Evening Star (3-mum pattern)."""
        # Orta mum küçük olmalı (doji benzeri)
        if middle['body_ratio'] > 0.3:
            return None
        
        # Morning Star: Bearish + Small + Bullish
        if first['is_bearish'] and last['is_bullish']:
            if last['close'] > (first['open'] + first['close']) / 2:
                return {
                    'name': 'MORNING_STAR',
                    'direction': 'LONG',
                    'confidence': 75
                }
        
        # Evening Star: Bullish + Small + Bearish
        if first['is_bullish'] and last['is_bearish']:
            if last['close'] < (first['open'] + first['close']) / 2:
                return {
                    'name': 'EVENING_STAR',
                    'direction': 'SHORT',
                    'confidence': 75
                }
        
        return None
    
    def _empty_result(self) -> Dict:
        return {
            'patterns_found': [],
            'latest_pattern': None,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'pattern_count': 0,
            'available': False
        }


# Convenience function
def get_candle_patterns(symbol: str = 'BTCUSDT', interval: str = '15m') -> Dict:
    """Quick candle pattern check."""
    recognizer = CandlePatternRecognizer()
    return recognizer.analyze_patterns(symbol, interval)
