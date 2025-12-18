# -*- coding: utf-8 -*-
"""
DEMIR AI - Bollinger Squeeze Detector
Volatilite sıkışması tespit - Ani hareket öncesi uyarı.

PHASE 71: Bollinger Squeeze Detection
- Bollinger Band width sıkışması tespit
- Sıkışma + Hacim artışı = Patlama yaklaşıyor
- Yön tahmini: Son mumun kapanışına göre
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("BOLLINGER_SQUEEZE")


class BollingerSqueezeDetector:
    """
    Bollinger Squeeze Detector
    
    Volatilite sıkışması → Ani hareket habercisi
    
    Nasıl Çalışır:
    1. Bollinger Bandwidth hesapla (BB_upper - BB_lower) / BB_middle
    2. Bandwidth < 2% (5+ mumdur) = Squeeze aktif
    3. Squeeze + Volume artışı = "PATLAMA YAKLAŞIYOR" sinyali
    """
    
    # Parametreler
    BB_PERIOD = 20
    BB_STD = 2.0
    SQUEEZE_THRESHOLD = 0.02  # %2 bandwidth = squeeze
    SQUEEZE_CANDLES = 5  # 5 mumdur squeeze olmalı
    VOLUME_MULTIPLIER = 1.5  # Volume artışı tetikleyici
    
    def __init__(self):
        self.last_check = None
        self.squeeze_history = []
        logger.info("✅ Bollinger Squeeze Detector initialized")
    
    def detect_squeeze(self, symbol: str = 'BTCUSDT', interval: str = '15m') -> Dict:
        """
        Bollinger Squeeze tespit et.
        
        Returns:
            {
                'squeeze_active': True/False,
                'squeeze_candles': 5,  # Kaç mumdur squeeze
                'bandwidth': 0.018,  # Mevcut bandwidth
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 65,
                'breakout_imminent': True/False,
                'volume_spike': True/False
            }
        """
        try:
            # Binance'den kline verisi çek
            resp = requests.get(
                f"https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': interval, 'limit': 50},
                timeout=5
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            klines = resp.json()
            
            # OHLCV verisini parse et
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])
            
            # Bollinger Bands hesapla
            bb_data = self._calculate_bollinger_bands(closes)
            
            # Bandwidth hesapla (son N mum için)
            bandwidths = bb_data['bandwidths']
            current_bandwidth = bandwidths[-1] if len(bandwidths) > 0 else 0.05
            
            # Squeeze tespit
            squeeze_count = sum(1 for bw in bandwidths[-self.SQUEEZE_CANDLES:] if bw < self.SQUEEZE_THRESHOLD)
            squeeze_active = squeeze_count >= self.SQUEEZE_CANDLES
            
            # Volume spike tespit
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_spike = current_volume > avg_volume * self.VOLUME_MULTIPLIER
            
            # Breakout imminent (squeeze + volume spike)
            breakout_imminent = squeeze_active and volume_spike
            
            # Yön tahmini
            direction = 'NEUTRAL'
            confidence = 40
            
            if squeeze_active:
                # Son mumun yönü
                last_close = closes[-1]
                last_open = float(klines[-1][1])
                prev_close = closes[-2]
                
                # BB konumu
                bb_middle = bb_data['middle'][-1]
                bb_upper = bb_data['upper'][-1]
                bb_lower = bb_data['lower'][-1]
                
                # Fiyat BB orta üstünde mi altında mı?
                if last_close > bb_middle:
                    direction = 'LONG'
                    confidence = 55
                elif last_close < bb_middle:
                    direction = 'SHORT'
                    confidence = 55
                
                # Volume spike varsa confidence artır
                if volume_spike:
                    confidence += 10
                
                # Breakout imminent ise confidence artır
                if breakout_imminent:
                    confidence += 10
                    if last_close > last_open:  # Yeşil mum
                        direction = 'LONG'
                    else:
                        direction = 'SHORT'
            
            return {
                'squeeze_active': squeeze_active,
                'squeeze_candles': squeeze_count,
                'bandwidth': current_bandwidth,
                'bandwidth_pct': current_bandwidth * 100,
                'direction': direction,
                'confidence': min(80, confidence),
                'breakout_imminent': breakout_imminent,
                'volume_spike': volume_spike,
                'bb_upper': bb_data['upper'][-1] if len(bb_data['upper']) > 0 else 0,
                'bb_lower': bb_data['lower'][-1] if len(bb_data['lower']) > 0 else 0,
                'bb_middle': bb_data['middle'][-1] if len(bb_data['middle']) > 0 else 0,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Bollinger Squeeze detection failed: {e}")
            return self._empty_result()
    
    def _calculate_bollinger_bands(self, closes: np.ndarray) -> Dict:
        """Bollinger Bands hesapla."""
        if len(closes) < self.BB_PERIOD:
            return {'upper': [], 'middle': [], 'lower': [], 'bandwidths': []}
        
        # SMA (middle band)
        middle = []
        upper = []
        lower = []
        bandwidths = []
        
        for i in range(self.BB_PERIOD - 1, len(closes)):
            window = closes[i - self.BB_PERIOD + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            
            bb_upper = sma + (self.BB_STD * std)
            bb_lower = sma - (self.BB_STD * std)
            bandwidth = (bb_upper - bb_lower) / sma if sma > 0 else 0
            
            middle.append(sma)
            upper.append(bb_upper)
            lower.append(bb_lower)
            bandwidths.append(bandwidth)
        
        return {
            'upper': np.array(upper),
            'middle': np.array(middle),
            'lower': np.array(lower),
            'bandwidths': np.array(bandwidths)
        }
    
    def _empty_result(self) -> Dict:
        """Boş sonuç döndür."""
        return {
            'squeeze_active': False,
            'squeeze_candles': 0,
            'bandwidth': 0,
            'bandwidth_pct': 0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'breakout_imminent': False,
            'volume_spike': False,
            'available': False
        }


# Convenience function
def detect_squeeze(symbol: str = 'BTCUSDT') -> Dict:
    """Quick squeeze detection."""
    detector = BollingerSqueezeDetector()
    return detector.detect_squeeze(symbol)
