# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Top Trader L/S Ratio Scraper
Profesyonel trader pozisyonları.

PHASE 82: Top Trader Long/Short Ratio
- Binance/OKX top trader pozisyonları
- Contrarian sinyal üretimi
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("COINGLASS_LS_RATIO")


class CoinGlassLSRatio:
    """
    CoinGlass Top Trader Long/Short Ratio Analyzer
    
    Binance top trader pozisyonlarını analiz eder.
    """
    
    BINANCE = "https://fapi.binance.com"
    
    def __init__(self):
        logger.info("✅ CoinGlass L/S Ratio Analyzer initialized")
    
    def get_ls_ratio(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Top trader L/S oranını al.
        
        Returns:
            {
                'global_ratio': 1.2,
                'top_trader_ratio': 1.5,
                'top_positions_ratio': 0.8,
                'sentiment': 'LONG_HEAVY'/'SHORT_HEAVY'/'BALANCED',
                'direction': 'LONG'/'SHORT',  # Contrarian
                'confidence': 60
            }
        """
        try:
            # Global Long/Short Account Ratio
            global_resp = requests.get(
                f"{self.BINANCE}/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=10
            )
            
            global_ratio = 1.0
            if global_resp.status_code == 200 and global_resp.json():
                global_ratio = float(global_resp.json()[0]['longShortRatio'])
            
            # Top Trader Long/Short Ratio (Accounts)
            top_resp = requests.get(
                f"{self.BINANCE}/futures/data/topLongShortAccountRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=10
            )
            
            top_trader_ratio = 1.0
            if top_resp.status_code == 200 and top_resp.json():
                top_trader_ratio = float(top_resp.json()[0]['longShortRatio'])
            
            # Top Trader Long/Short Ratio (Positions)
            pos_resp = requests.get(
                f"{self.BINANCE}/futures/data/topLongShortPositionRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=10
            )
            
            top_positions_ratio = 1.0
            if pos_resp.status_code == 200 and pos_resp.json():
                top_positions_ratio = float(pos_resp.json()[0]['longShortRatio'])
            
            # Sentiment analizi
            avg_ratio = (global_ratio + top_trader_ratio + top_positions_ratio) / 3
            
            if avg_ratio > 1.5:
                sentiment = 'LONG_HEAVY'
                direction = 'SHORT'  # Contrarian
                confidence = min(70, 50 + (avg_ratio - 1) * 30)
            elif avg_ratio > 1.2:
                sentiment = 'LONG_LEANING'
                direction = 'SHORT'
                confidence = min(55, 45 + (avg_ratio - 1) * 20)
            elif avg_ratio < 0.7:
                sentiment = 'SHORT_HEAVY'
                direction = 'LONG'  # Contrarian
                confidence = min(70, 50 + (1/avg_ratio - 1) * 30)
            elif avg_ratio < 0.85:
                sentiment = 'SHORT_LEANING'
                direction = 'LONG'
                confidence = min(55, 45 + (1/avg_ratio - 1) * 20)
            else:
                sentiment = 'BALANCED'
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'global_ratio': global_ratio,
                'top_trader_ratio': top_trader_ratio,
                'top_positions_ratio': top_positions_ratio,
                'avg_ratio': avg_ratio,
                'sentiment': sentiment,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"L/S Ratio failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'global_ratio': 1.0,
            'top_trader_ratio': 1.0,
            'top_positions_ratio': 1.0,
            'avg_ratio': 1.0,
            'sentiment': 'UNKNOWN',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_ls_ratio(symbol: str = 'BTCUSDT') -> Dict:
    """Quick L/S ratio check."""
    analyzer = CoinGlassLSRatio()
    return analyzer.get_ls_ratio(symbol)
