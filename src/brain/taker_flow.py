# -*- coding: utf-8 -*-
"""
DEMIR AI - Taker Flow Delta
Agresif alıcı/satıcı dengesi - Hareket yönü tahmini.

PHASE 74: Taker Buy/Sell Imbalance
- Taker = Market order (agresif)
- Taker Buy > Sell = Yukarı baskı
- Taker Sell > Buy = Aşağı baskı
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List
import numpy as np

logger = logging.getLogger("TAKER_FLOW")


class TakerFlowDelta:
    """
    Taker Flow Delta Analyzer
    
    Agresif alıcı vs satıcı dengesizliği tespit eder.
    
    Nasıl Çalışır:
    1. Binance aggTrades API'den son işlemleri çek
    2. Taker Buy vs Taker Sell hacmini hesapla
    3. Delta > 0 = Alıcı baskısı (LONG)
    4. Delta < 0 = Satıcı baskısı (SHORT)
    """
    
    IMBALANCE_THRESHOLD = 1.3  # %30 dengesizlik = anlamlı
    STRONG_IMBALANCE = 1.6  # %60 dengesizlik = güçlü
    
    def __init__(self):
        logger.info("✅ Taker Flow Delta initialized")
    
    def analyze_flow(self, symbol: str = 'BTCUSDT', minutes: int = 15) -> Dict:
        """
        Taker flow analizi yap.
        
        Returns:
            {
                'taker_buy_vol': 150.5,
                'taker_sell_vol': 100.2,
                'delta': 50.3,  # Buy - Sell
                'ratio': 1.5,  # Buy / Sell
                'direction': 'LONG'/'SHORT',
                'imbalance': 'STRONG'/'MODERATE'/'NONE',
                'confidence': 65
            }
        """
        try:
            # Son aggTrades çek
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (minutes * 60 * 1000)
            
            resp = requests.get(
                f"https://api.binance.com/api/v3/aggTrades",
                params={
                    'symbol': symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': 1000
                },
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            trades = resp.json()
            
            if not trades:
                return self._empty_result()
            
            # Taker Buy vs Sell hesapla
            taker_buy = 0
            taker_sell = 0
            
            for trade in trades:
                qty = float(trade['q'])
                price = float(trade['p'])
                volume = qty * price
                
                if trade['m']:  # maker = sell, yani karşıdaki taker = buy
                    taker_sell += volume
                else:
                    taker_buy += volume
            
            # Delta ve ratio
            delta = taker_buy - taker_sell
            total = taker_buy + taker_sell
            ratio = taker_buy / taker_sell if taker_sell > 0 else 1.0
            
            # Yön ve güven
            if ratio >= self.STRONG_IMBALANCE:
                direction = 'LONG'
                imbalance = 'STRONG'
                confidence = min(80, 55 + (ratio - 1) * 20)
            elif ratio >= self.IMBALANCE_THRESHOLD:
                direction = 'LONG'
                imbalance = 'MODERATE'
                confidence = min(65, 50 + (ratio - 1) * 15)
            elif ratio <= 1 / self.STRONG_IMBALANCE:
                direction = 'SHORT'
                imbalance = 'STRONG'
                confidence = min(80, 55 + (1/ratio - 1) * 20)
            elif ratio <= 1 / self.IMBALANCE_THRESHOLD:
                direction = 'SHORT'
                imbalance = 'MODERATE'
                confidence = min(65, 50 + (1/ratio - 1) * 15)
            else:
                direction = 'NEUTRAL'
                imbalance = 'NONE'
                confidence = 40
            
            return {
                'taker_buy_vol': round(taker_buy / 1e6, 2),  # Million USD
                'taker_sell_vol': round(taker_sell / 1e6, 2),
                'delta': round(delta / 1e6, 2),
                'ratio': round(ratio, 2),
                'direction': direction,
                'imbalance': imbalance,
                'confidence': confidence,
                'trade_count': len(trades),
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Taker flow analysis failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'taker_buy_vol': 0,
            'taker_sell_vol': 0,
            'delta': 0,
            'ratio': 1.0,
            'direction': 'NEUTRAL',
            'imbalance': 'NONE',
            'confidence': 0,
            'trade_count': 0,
            'available': False
        }


# Convenience function
def analyze_taker_flow(symbol: str = 'BTCUSDT') -> Dict:
    """Quick taker flow analysis."""
    analyzer = TakerFlowDelta()
    return analyzer.analyze_flow(symbol)
