# -*- coding: utf-8 -*-
"""
DEMIR AI - Liquidation Cascade Predictor
Long/Short squeeze riski tespit - Ani hareket öncesi uyarı.

PHASE 72: Liquidation Cascade Detection
- Binance Futures OI değişimi
- Funding Rate extreme değerler
- Long/Short Ratio dengesizliği
- Cascade riski hesaplama
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("LIQUIDATION_CASCADE")


class LiquidationCascadePredictor:
    """
    Liquidation Cascade Predictor
    
    Kaldıraçlı pozisyon tasfiye riskini önceden tespit eder.
    
    Nasıl Çalışır:
    1. OI (Open Interest) çok yüksek = Çok fazla pozisyon açık
    2. Funding Rate extreme = Bir taraf aşırı kaldıraçlı
    3. L/S Ratio dengesiz = Bir taraf aşırı yoğun
    → Cascade riski = Ani tasfiye dalgası = SERT HAREKET
    """
    
    # Thresholds
    FUNDING_EXTREME_LONG = 0.05   # %0.05+ = Long squeeze riski
    FUNDING_EXTREME_SHORT = -0.03  # -%0.03 = Short squeeze riski
    LS_RATIO_EXTREME_LONG = 2.0   # Long/Short > 2 = Long squeeze
    LS_RATIO_EXTREME_SHORT = 0.5  # Long/Short < 0.5 = Short squeeze
    OI_CHANGE_THRESHOLD = 5.0     # %5+ OI değişimi = Pozisyon birikimi
    
    def __init__(self):
        self.api_base = "https://fapi.binance.com"
        logger.info("✅ Liquidation Cascade Predictor initialized")
    
    def predict_cascade(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Cascade riski tahmin et.
        
        Returns:
            {
                'cascade_risk': 'HIGH'/'MEDIUM'/'LOW',
                'risk_score': 75,  # 0-100
                'squeeze_type': 'LONG_SQUEEZE'/'SHORT_SQUEEZE'/None,
                'direction': 'LONG'/'SHORT',  # Squeeze sonrası beklenen yön
                'funding_rate': 0.05,
                'long_short_ratio': 1.8,
                'oi_change_24h': 8.5,
                'confidence': 65
            }
        """
        try:
            # 1. Funding Rate çek
            funding = self._get_funding_rate(symbol)
            
            # 2. Long/Short Ratio çek
            ls_ratio = self._get_long_short_ratio(symbol)
            
            # 3. OI değişimi çek
            oi_change = self._get_oi_change(symbol)
            
            # 4. Cascade riski hesapla
            risk_score = 0
            squeeze_type = None
            direction = 'NEUTRAL'
            
            # Funding Rate analizi
            funding_rate = funding.get('funding_rate', 0)
            if funding_rate > self.FUNDING_EXTREME_LONG:
                risk_score += 30
                squeeze_type = 'LONG_SQUEEZE'
                direction = 'SHORT'
            elif funding_rate < self.FUNDING_EXTREME_SHORT:
                risk_score += 30
                squeeze_type = 'SHORT_SQUEEZE'
                direction = 'LONG'
            elif abs(funding_rate) > 0.02:
                risk_score += 15
            
            # L/S Ratio analizi
            ratio = ls_ratio.get('ratio', 1.0)
            if ratio > self.LS_RATIO_EXTREME_LONG:
                risk_score += 25
                if squeeze_type is None:
                    squeeze_type = 'LONG_SQUEEZE'
                    direction = 'SHORT'
            elif ratio < self.LS_RATIO_EXTREME_SHORT:
                risk_score += 25
                if squeeze_type is None:
                    squeeze_type = 'SHORT_SQUEEZE'
                    direction = 'LONG'
            elif ratio > 1.5 or ratio < 0.7:
                risk_score += 10
            
            # OI değişimi analizi
            oi_pct = oi_change.get('change_pct', 0)
            if abs(oi_pct) > self.OI_CHANGE_THRESHOLD:
                risk_score += 20
            elif abs(oi_pct) > 2.0:
                risk_score += 10
            
            # Risk seviyesi
            if risk_score >= 60:
                cascade_risk = 'HIGH'
                confidence = min(80, 55 + risk_score * 0.3)
            elif risk_score >= 35:
                cascade_risk = 'MEDIUM'
                confidence = min(65, 45 + risk_score * 0.3)
            else:
                cascade_risk = 'LOW'
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'cascade_risk': cascade_risk,
                'risk_score': risk_score,
                'squeeze_type': squeeze_type,
                'direction': direction,
                'funding_rate': funding_rate,
                'funding_rate_pct': funding_rate * 100,
                'long_short_ratio': ratio,
                'oi_change_24h': oi_pct,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Cascade prediction failed: {e}")
            return self._empty_result()
    
    def _get_funding_rate(self, symbol: str) -> Dict:
        """Binance Futures funding rate çek."""
        try:
            resp = requests.get(
                f"{self.api_base}/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return {'funding_rate': float(data[0]['fundingRate'])}
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")
        return {'funding_rate': 0}
    
    def _get_long_short_ratio(self, symbol: str) -> Dict:
        """Binance Futures Long/Short ratio çek."""
        try:
            resp = requests.get(
                f"{self.api_base}/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return {'ratio': float(data[0]['longShortRatio'])}
        except Exception as e:
            logger.warning(f"L/S ratio fetch failed: {e}")
        return {'ratio': 1.0}
    
    def _get_oi_change(self, symbol: str) -> Dict:
        """Binance Futures OI değişimi çek (24h)."""
        try:
            resp = requests.get(
                f"{self.api_base}/futures/data/openInterestHist",
                params={'symbol': symbol, 'period': '1h', 'limit': 24},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 2:
                    current_oi = float(data[-1]['sumOpenInterest'])
                    old_oi = float(data[0]['sumOpenInterest'])
                    change_pct = ((current_oi / old_oi) - 1) * 100 if old_oi > 0 else 0
                    return {'change_pct': change_pct}
        except Exception as e:
            logger.warning(f"OI change fetch failed: {e}")
        return {'change_pct': 0}
    
    def _empty_result(self) -> Dict:
        """Boş sonuç döndür."""
        return {
            'cascade_risk': 'LOW',
            'risk_score': 0,
            'squeeze_type': None,
            'direction': 'NEUTRAL',
            'funding_rate': 0,
            'funding_rate_pct': 0,
            'long_short_ratio': 1.0,
            'oi_change_24h': 0,
            'confidence': 0,
            'available': False
        }


# Convenience function
def predict_liquidation_cascade(symbol: str = 'BTCUSDT') -> Dict:
    """Quick cascade prediction."""
    predictor = LiquidationCascadePredictor()
    return predictor.predict_cascade(symbol)
