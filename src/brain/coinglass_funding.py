# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Funding Rate Scraper
Funding rate extreme tespit.

PHASE 81: Funding Rate Extremes
- Aşırı pozitif = Long squeeze riski
- Aşırı negatif = Short squeeze riski
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("COINGLASS_FUNDING")


class CoinGlassFunding:
    """
    CoinGlass Funding Rate Analyzer
    
    Çoklu borsa funding rate karşılaştırması.
    """
    
    BINANCE = "https://fapi.binance.com"
    
    def __init__(self):
        logger.info("✅ CoinGlass Funding Analyzer initialized")
    
    def get_funding_analysis(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Funding rate analizi yap.
        
        Returns:
            {
                'binance_funding': 0.01,
                'avg_funding': 0.008,
                'is_extreme': True,
                'extreme_type': 'LONG_CROWDED'/'SHORT_CROWDED',
                'squeeze_risk': 'HIGH'/'MEDIUM'/'LOW',
                'direction': 'LONG'/'SHORT',
                'confidence': 65
            }
        """
        try:
            # Binance Funding Rate
            resp = requests.get(
                f"{self.BINANCE}/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 8},  # Son 8 funding (1 gün)
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            data = resp.json()
            
            if not data:
                return self._empty_result()
            
            # Funding değerleri
            current_funding = float(data[-1]['fundingRate'])
            funding_rates = [float(d['fundingRate']) for d in data]
            avg_funding = sum(funding_rates) / len(funding_rates)
            
            # Extreme tespit
            is_extreme = abs(current_funding) > 0.03  # %0.03+ = extreme
            
            if current_funding > 0.05:
                extreme_type = 'LONG_CROWDED'
                squeeze_risk = 'HIGH'
                direction = 'SHORT'  # Long squeeze bekleniyor
                confidence = min(80, 55 + current_funding * 500)
            elif current_funding > 0.03:
                extreme_type = 'LONG_CROWDED'
                squeeze_risk = 'MEDIUM'
                direction = 'SHORT'
                confidence = min(65, 50 + current_funding * 400)
            elif current_funding < -0.03:
                extreme_type = 'SHORT_CROWDED'
                squeeze_risk = 'MEDIUM'
                direction = 'LONG'  # Short squeeze bekleniyor
                confidence = min(70, 55 + abs(current_funding) * 400)
            elif current_funding < -0.05:
                extreme_type = 'SHORT_CROWDED'
                squeeze_risk = 'HIGH'
                direction = 'LONG'
                confidence = min(80, 55 + abs(current_funding) * 500)
            else:
                extreme_type = 'NEUTRAL'
                squeeze_risk = 'LOW'
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'current_funding': current_funding,
                'current_funding_pct': current_funding * 100,
                'avg_funding': avg_funding,
                'avg_funding_pct': avg_funding * 100,
                'is_extreme': is_extreme,
                'extreme_type': extreme_type,
                'squeeze_risk': squeeze_risk,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Funding analysis failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'current_funding': 0,
            'current_funding_pct': 0,
            'avg_funding': 0,
            'avg_funding_pct': 0,
            'is_extreme': False,
            'extreme_type': 'UNKNOWN',
            'squeeze_risk': 'LOW',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_funding_analysis(symbol: str = 'BTCUSDT') -> Dict:
    """Quick funding analysis."""
    analyzer = CoinGlassFunding()
    return analyzer.get_funding_analysis(symbol)
