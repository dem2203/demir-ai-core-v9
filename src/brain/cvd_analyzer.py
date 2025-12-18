# -*- coding: utf-8 -*-
"""
DEMIR AI - CVD Analyzer (Cumulative Volume Delta)
Net alıcı/satıcı baskısı - Order flow analizi.

PHASE 89: CVD (Cumulative Volume Delta)
- Taker buy vs sell volume
- Net pressure tracking
- Divergence detection
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("CVD_ANALYZER")


class CVDAnalyzer:
    """
    Cumulative Volume Delta Analyzer
    
    Taker alım/satım dengesini takip ederek piyasa baskısını ölçer.
    """
    
    BINANCE = "https://api.binance.com/api/v3"
    BINANCE_FUTURES = "https://fapi.binance.com"
    
    def __init__(self):
        logger.info("✅ CVD Analyzer initialized")
    
    def analyze_cvd(self, symbol: str = 'BTCUSDT', lookback_minutes: int = 60) -> Dict:
        """
        CVD analizi yap.
        
        Returns:
            {
                'taker_buy_volume': 1000,
                'taker_sell_volume': 800,
                'cvd': 200,  # Pozitif = net alıcı
                'cvd_trend': 'INCREASING',
                'pressure': 'BUY_PRESSURE',
                'direction': 'LONG',
                'confidence': 65
            }
        """
        try:
            # Son X dakikanın aggTrade verisi
            trades = self._get_recent_trades(symbol, limit=1000)
            
            if not trades:
                return self._empty_result()
            
            # Taker buy/sell ayrımı
            taker_buy_vol = 0
            taker_sell_vol = 0
            
            for trade in trades:
                qty = float(trade.get('qty', 0))
                is_buyer_maker = trade.get('isBuyerMaker', False)
                
                if is_buyer_maker:
                    # Maker is buyer = Taker is seller
                    taker_sell_vol += qty
                else:
                    # Maker is seller = Taker is buyer
                    taker_buy_vol += qty
            
            # CVD hesapla
            cvd = taker_buy_vol - taker_sell_vol
            total_vol = taker_buy_vol + taker_sell_vol
            
            # CVD ratio
            cvd_ratio = taker_buy_vol / taker_sell_vol if taker_sell_vol > 0 else 2.0
            
            # Pressure belirleme
            if cvd_ratio > 1.3:
                pressure = 'STRONG_BUY_PRESSURE'
                direction = 'LONG'
                confidence = min(75, 55 + (cvd_ratio - 1) * 30)
            elif cvd_ratio > 1.1:
                pressure = 'BUY_PRESSURE'
                direction = 'LONG'
                confidence = min(60, 50 + (cvd_ratio - 1) * 20)
            elif cvd_ratio < 0.77:
                pressure = 'STRONG_SELL_PRESSURE'
                direction = 'SHORT'
                confidence = min(75, 55 + (1/cvd_ratio - 1) * 30)
            elif cvd_ratio < 0.9:
                pressure = 'SELL_PRESSURE'
                direction = 'SHORT'
                confidence = min(60, 50 + (1/cvd_ratio - 1) * 20)
            else:
                pressure = 'NEUTRAL'
                direction = 'NEUTRAL'
                confidence = 40
            
            # Trend (ilk yarı vs son yarı)
            mid = len(trades) // 2
            first_half_buy = sum(float(t['qty']) for t in trades[:mid] if not t.get('isBuyerMaker', False))
            second_half_buy = sum(float(t['qty']) for t in trades[mid:] if not t.get('isBuyerMaker', False))
            
            if second_half_buy > first_half_buy * 1.2:
                cvd_trend = 'INCREASING'
            elif second_half_buy < first_half_buy * 0.8:
                cvd_trend = 'DECREASING'
            else:
                cvd_trend = 'STABLE'
            
            return {
                'taker_buy_volume': taker_buy_vol,
                'taker_sell_volume': taker_sell_vol,
                'cvd': cvd,
                'cvd_ratio': cvd_ratio,
                'cvd_trend': cvd_trend,
                'pressure': pressure,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"CVD analysis failed: {e}")
            return self._empty_result()
    
    def _get_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Son trade'leri al."""
        try:
            resp = requests.get(
                f"{self.BINANCE}/trades",
                params={'symbol': symbol, 'limit': limit},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return []
    
    def _empty_result(self) -> Dict:
        return {
            'taker_buy_volume': 0,
            'taker_sell_volume': 0,
            'cvd': 0,
            'cvd_ratio': 1,
            'cvd_trend': 'UNKNOWN',
            'pressure': 'UNKNOWN',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def analyze_cvd(symbol: str = 'BTCUSDT') -> Dict:
    """Quick CVD analysis."""
    analyzer = CVDAnalyzer()
    return analyzer.analyze_cvd(symbol)
