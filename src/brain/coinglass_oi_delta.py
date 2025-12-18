# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass OI Delta Scraper
Open Interest değişim hızı analizi.

PHASE 80: OI Delta Analysis
- OI hızlı artış = Volatilite yaklaşıyor
- OI hızlı düşüş = Panik kapatma
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("COINGLASS_OI_DELTA")


class CoinGlassOIDelta:
    """
    CoinGlass OI Delta Analyzer
    
    Open Interest değişim hızını analiz eder.
    """
    
    BINANCE_FUTURES = "https://fapi.binance.com"
    
    def __init__(self):
        logger.info("✅ CoinGlass OI Delta Analyzer initialized")
    
    def get_oi_delta(self, symbol: str = 'BTCUSDT', hours: int = 4) -> Dict:
        """
        OI değişim hızını hesapla.
        
        Returns:
            {
                'current_oi': 500000,
                'oi_1h_ago': 490000,
                'oi_4h_ago': 480000,
                'delta_1h_pct': 2.0,
                'delta_4h_pct': 4.2,
                'velocity': 'INCREASING'/'DECREASING'/'STABLE',
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 60
            }
        """
        try:
            # OI geçmişi al
            resp = requests.get(
                f"{self.BINANCE_FUTURES}/futures/data/openInterestHist",
                params={'symbol': symbol, 'period': '1h', 'limit': hours + 1},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            data = resp.json()
            
            if len(data) < 2:
                return self._empty_result()
            
            # OI değerleri
            current_oi = float(data[-1]['sumOpenInterest'])
            oi_1h = float(data[-2]['sumOpenInterest']) if len(data) >= 2 else current_oi
            oi_4h = float(data[0]['sumOpenInterest']) if len(data) >= hours else current_oi
            
            # Delta hesapla
            delta_1h = ((current_oi / oi_1h) - 1) * 100 if oi_1h > 0 else 0
            delta_4h = ((current_oi / oi_4h) - 1) * 100 if oi_4h > 0 else 0
            
            # Velocity (değişim hızı)
            if delta_4h > 5:
                velocity = 'INCREASING'
            elif delta_4h < -3:
                velocity = 'DECREASING'
            else:
                velocity = 'STABLE'
            
            # Fiyat değişimi için
            price_resp = requests.get(
                f"{self.BINANCE_FUTURES}/fapi/v1/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            price_change = float(price_resp.json().get('priceChangePercent', 0)) if price_resp.status_code == 200 else 0
            
            # Yön ve güven
            if velocity == 'INCREASING' and delta_4h > 8:
                # OI hızla artıyor - breakout yaklaşıyor
                direction = 'LONG' if price_change > 0 else 'SHORT'
                confidence = min(75, 50 + delta_4h * 2)
            elif velocity == 'DECREASING' and delta_4h < -5:
                # OI hızla düşüyor - panik
                direction = 'SHORT' if price_change < 0 else 'LONG'
                confidence = min(70, 50 + abs(delta_4h) * 2)
            else:
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'current_oi': current_oi,
                'oi_1h_ago': oi_1h,
                'oi_4h_ago': oi_4h,
                'delta_1h_pct': delta_1h,
                'delta_4h_pct': delta_4h,
                'price_change_pct': price_change,
                'velocity': velocity,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"OI Delta failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'current_oi': 0,
            'oi_1h_ago': 0,
            'oi_4h_ago': 0,
            'delta_1h_pct': 0,
            'delta_4h_pct': 0,
            'velocity': 'UNKNOWN',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_oi_delta(symbol: str = 'BTCUSDT') -> Dict:
    """Quick OI delta check."""
    analyzer = CoinGlassOIDelta()
    return analyzer.get_oi_delta(symbol)
