# -*- coding: utf-8 -*-
"""
DEMIR AI - Volume Spike Detector
Ani hacim patlaması tespit - Büyük oyuncu aktivitesi.

PHASE 73: Volume Spike Detection
- Normal hacmin 3x+ = Spike
- Spike yönü: Mum rengine göre
- Büyük oyuncu harekete geçti sinyali
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List
import numpy as np

logger = logging.getLogger("VOLUME_SPIKE")


class VolumeSpikeDetector:
    """
    Volume Spike Detector
    
    Ani hacim artışı → Büyük oyuncu harekete geçti
    
    Nasıl Çalışır:
    1. Son 20 mumun ortalama hacmini hesapla
    2. Mevcut hacim > 3x ortalama = SPIKE
    3. Spike + Yeşil mum = LONG, Kırmızı = SHORT
    """
    
    SPIKE_MULTIPLIER = 3.0  # 3x = spike
    STRONG_SPIKE = 5.0  # 5x = çok güçlü spike
    LOOKBACK = 20  # 20 mum ortalama
    
    def __init__(self):
        logger.info("✅ Volume Spike Detector initialized")
    
    def detect_spike(self, symbol: str = 'BTCUSDT', interval: str = '15m') -> Dict:
        """
        Volume spike tespit et.
        
        Returns:
            {
                'spike_detected': True/False,
                'spike_strength': 3.5,  # Normal hacmin kaç katı
                'direction': 'LONG'/'SHORT',
                'confidence': 65,
                'current_volume': 1500,
                'avg_volume': 500,
                'candle_type': 'BULLISH'/'BEARISH'
            }
        """
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': 30},
                timeout=5
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            klines = resp.json()
            
            # Volumes ve OHLC
            volumes = [float(k[5]) for k in klines]
            
            # Ortalama ve mevcut
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
            current_volume = volumes[-1]
            
            # Spike hesapla
            spike_strength = current_volume / avg_volume if avg_volume > 0 else 1.0
            spike_detected = spike_strength >= self.SPIKE_MULTIPLIER
            
            # Mum yönü
            last_open = float(klines[-1][1])
            last_close = float(klines[-1][4])
            candle_type = 'BULLISH' if last_close > last_open else 'BEARISH'
            
            # Yön ve güven
            if spike_detected:
                direction = 'LONG' if candle_type == 'BULLISH' else 'SHORT'
                confidence = min(80, 50 + (spike_strength - 2) * 10)
                
                if spike_strength >= self.STRONG_SPIKE:
                    confidence = min(85, confidence + 10)
            else:
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'spike_detected': spike_detected,
                'spike_strength': round(spike_strength, 2),
                'direction': direction,
                'confidence': confidence,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'candle_type': candle_type,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Volume spike detection failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'spike_detected': False,
            'spike_strength': 1.0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'current_volume': 0,
            'avg_volume': 0,
            'candle_type': 'NEUTRAL',
            'available': False
        }


# Convenience function
def detect_volume_spike(symbol: str = 'BTCUSDT') -> Dict:
    """Quick volume spike detection."""
    detector = VolumeSpikeDetector()
    return detector.detect_spike(symbol)
