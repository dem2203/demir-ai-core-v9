# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Orderbook Delta Scraper
Orderbook likidite dengesizliği.

PHASE 83: Orderbook Liquidity Delta
- Alış tarafı daha güçlü = Yukarı baskı
- Satış tarafı daha güçlü = Aşağı baskı
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List
import numpy as np

logger = logging.getLogger("COINGLASS_ORDERBOOK")


class CoinGlassOrderbook:
    """
    CoinGlass Orderbook Delta Analyzer
    
    Orderbook likidite dengesizliğini ölçer.
    """
    
    BINANCE = "https://api.binance.com/api/v3"
    
    def __init__(self):
        logger.info("✅ CoinGlass Orderbook Delta initialized")
    
    def get_orderbook_delta(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Orderbook delta analizi yap.
        
        Returns:
            {
                'bid_liquidity': 50000000,  # USD
                'ask_liquidity': 45000000,
                'delta': 5000000,  # Bid - Ask
                'delta_pct': 11.1,
                'imbalance': 'BID_HEAVY'/'ASK_HEAVY'/'BALANCED',
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 60
            }
        """
        try:
            # Orderbook al (depth 500)
            resp = requests.get(
                f"{self.BINANCE}/depth",
                params={'symbol': symbol, 'limit': 500},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            data = resp.json()
            
            # Mevcut fiyat
            price_resp = requests.get(
                f"{self.BINANCE}/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            current_price = float(price_resp.json()['price'])
            
            # Bid ve Ask likiditeyi hesapla (fiyata yakın %2 aralık)
            price_range_lower = current_price * 0.98
            price_range_upper = current_price * 1.02
            
            bid_liquidity = 0
            for bid in data.get('bids', []):
                price = float(bid[0])
                qty = float(bid[1])
                if price >= price_range_lower:
                    bid_liquidity += price * qty
            
            ask_liquidity = 0
            for ask in data.get('asks', []):
                price = float(ask[0])
                qty = float(ask[1])
                if price <= price_range_upper:
                    ask_liquidity += price * qty
            
            # Delta hesapla
            delta = bid_liquidity - ask_liquidity
            total = bid_liquidity + ask_liquidity
            delta_pct = (delta / total * 100) if total > 0 else 0
            
            # Imbalance analizi
            ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else 1.0
            
            if ratio > 1.3:
                imbalance = 'BID_HEAVY'
                direction = 'LONG'
                confidence = min(70, 50 + (ratio - 1) * 30)
            elif ratio > 1.1:
                imbalance = 'BID_LEANING'
                direction = 'LONG'
                confidence = min(55, 45 + (ratio - 1) * 20)
            elif ratio < 0.77:
                imbalance = 'ASK_HEAVY'
                direction = 'SHORT'
                confidence = min(70, 50 + (1/ratio - 1) * 30)
            elif ratio < 0.9:
                imbalance = 'ASK_LEANING'
                direction = 'SHORT'
                confidence = min(55, 45 + (1/ratio - 1) * 20)
            else:
                imbalance = 'BALANCED'
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'delta': delta,
                'delta_pct': delta_pct,
                'bid_ask_ratio': ratio,
                'imbalance': imbalance,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Orderbook delta failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'bid_liquidity': 0,
            'ask_liquidity': 0,
            'delta': 0,
            'delta_pct': 0,
            'bid_ask_ratio': 1.0,
            'imbalance': 'UNKNOWN',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_orderbook_delta(symbol: str = 'BTCUSDT') -> Dict:
    """Quick orderbook delta check."""
    analyzer = CoinGlassOrderbook()
    return analyzer.get_orderbook_delta(symbol)
