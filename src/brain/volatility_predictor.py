# -*- coding: utf-8 -*-
"""
DEMIR AI - Volatility Predictor
Volatilite tahmini - Patlama öncesi uyarı.

PHASE 87: Volatility Predictor (GARCH-like)
- Volatilite sıkışması tespit
- Breakout zamanlaması tahmini
- ATR bazlı analiz
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List
import numpy as np

logger = logging.getLogger("VOLATILITY_PREDICTOR")


class VolatilityPredictor:
    """
    Volatilite Tahmin Sistemi
    
    Düşük volatilite dönemlerini tespit edip breakout öncesi uyarı verir.
    """
    
    BINANCE_API = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.squeeze_threshold = 0.5  # Normal volatilitenin %50'si altı = squeeze
        self.breakout_threshold = 1.5  # Normal volatilitenin 1.5 katı = breakout
        logger.info("✅ Volatility Predictor initialized")
    
    def predict_volatility(self, symbol: str = 'BTCUSDT', interval: str = '1h', lookback: int = 24) -> Dict:
        """
        Volatilite tahmini yap.
        
        Returns:
            {
                'current_volatility': 1.2,
                'avg_volatility': 1.5,
                'volatility_ratio': 0.8,
                'state': 'SQUEEZE' / 'NORMAL' / 'EXPANSION',
                'breakout_probability': 75,
                'direction': 'LONG' / 'SHORT' / 'NEUTRAL',
                'confidence': 65
            }
        """
        try:
            # Kline verisi al
            resp = requests.get(
                f"{self.BINANCE_API}/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': lookback + 10},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            klines = resp.json()
            
            if len(klines) < lookback:
                return self._empty_result()
            
            # ATR hesapla (Average True Range)
            atr_values = []
            closes = []
            
            for i, k in enumerate(klines):
                h, l, c = float(k[2]), float(k[3]), float(k[4])
                closes.append(c)
                
                if i > 0:
                    prev_c = float(klines[i-1][4])
                    tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                    atr_values.append(tr)
            
            if len(atr_values) < 10:
                return self._empty_result()
            
            # ATR yüzdesi olarak (fiyata göre normalize)
            current_price = closes[-1]
            atr_pct = [(atr / current_price) * 100 for atr in atr_values]
            
            current_volatility = atr_pct[-1]
            avg_volatility = np.mean(atr_pct[-lookback:])
            volatility_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1
            
            # Volatilite trendi
            recent_vol = np.mean(atr_pct[-5:])
            older_vol = np.mean(atr_pct[-15:-5]) if len(atr_pct) >= 15 else avg_volatility
            vol_trend = (recent_vol - older_vol) / older_vol * 100 if older_vol > 0 else 0
            
            # State belirleme
            if volatility_ratio < self.squeeze_threshold:
                state = 'SQUEEZE'
                breakout_prob = 80  # Squeeze sonrası breakout yüksek
                confidence = 70
            elif volatility_ratio > self.breakout_threshold:
                state = 'EXPANSION'
                breakout_prob = 30  # Zaten patlama oldu
                confidence = 55
            else:
                state = 'NORMAL'
                breakout_prob = 40
                confidence = 45
            
            # Yön tahmini (fiyat trendi)
            price_change = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
            
            if state == 'SQUEEZE':
                # Squeeze sonrası yön: trend yönünde devam
                direction = 'LONG' if price_change > 0 else 'SHORT'
            elif state == 'EXPANSION':
                direction = 'LONG' if price_change > 0 else 'SHORT'
            else:
                direction = 'NEUTRAL'
            
            return {
                'current_volatility': current_volatility,
                'avg_volatility': avg_volatility,
                'volatility_ratio': volatility_ratio,
                'volatility_trend_pct': vol_trend,
                'state': state,
                'breakout_probability': breakout_prob,
                'price_trend_pct': price_change,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Volatility prediction failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'current_volatility': 0,
            'avg_volatility': 0,
            'volatility_ratio': 1,
            'volatility_trend_pct': 0,
            'state': 'UNKNOWN',
            'breakout_probability': 0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def predict_volatility(symbol: str = 'BTCUSDT') -> Dict:
    """Quick volatility prediction."""
    predictor = VolatilityPredictor()
    return predictor.predict_volatility(symbol)
