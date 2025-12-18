# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Whale Orders Scraper
Büyük emir duvarları - Whale alış/satış emirleri.

PHASE 78: Whale Orders Scraping
- $1M+ büyük emirler
- Alış duvarı = Destek, Satış duvarı = Direnç
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("COINGLASS_WHALE_ORDERS")


class CoinGlassWhaleOrders:
    """
    CoinGlass Whale Orders Scraper
    
    Emir defterindeki büyük emirleri ($1M+) tespit eder.
    """
    
    # Binance Orderbook API
    BINANCE_API = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.min_order_usd = 1_000_000  # $1M minimum
        logger.info("✅ CoinGlass Whale Orders Scraper initialized")
    
    def get_whale_orders(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Whale emirlerini tespit et.
        
        Returns:
            {
                'whale_bids': [{'price': 85000, 'volume_usd': 5000000}],
                'whale_asks': [{'price': 90000, 'volume_usd': 3000000}],
                'nearest_bid_wall': price,
                'nearest_ask_wall': price,
                'bid_wall_strength': 0-100,
                'ask_wall_strength': 0-100,
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 60
            }
        """
        try:
            # Binance orderbook al (depth 500)
            resp = requests.get(
                f"{self.BINANCE_API}/depth",
                params={'symbol': symbol, 'limit': 500},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self._empty_result()
            
            data = resp.json()
            
            # Mevcut fiyat
            price_resp = requests.get(
                f"{self.BINANCE_API}/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            current_price = float(price_resp.json()['price'])
            
            # Whale emirlerini bul
            whale_bids = []
            whale_asks = []
            
            # Bids (alış emirleri)
            for bid in data.get('bids', []):
                price = float(bid[0])
                qty = float(bid[1])
                volume_usd = price * qty
                
                if volume_usd >= self.min_order_usd:
                    whale_bids.append({
                        'price': price,
                        'volume_usd': volume_usd,
                        'distance_pct': (current_price - price) / current_price * 100
                    })
            
            # Asks (satış emirleri)
            for ask in data.get('asks', []):
                price = float(ask[0])
                qty = float(ask[1])
                volume_usd = price * qty
                
                if volume_usd >= self.min_order_usd:
                    whale_asks.append({
                        'price': price,
                        'volume_usd': volume_usd,
                        'distance_pct': (price - current_price) / current_price * 100
                    })
            
            # En yakın duvarlar
            nearest_bid = min(whale_bids, key=lambda x: x['distance_pct'])['price'] if whale_bids else 0
            nearest_ask = min(whale_asks, key=lambda x: x['distance_pct'])['price'] if whale_asks else 0
            
            # Toplam duvar gücü
            total_bid_vol = sum(w['volume_usd'] for w in whale_bids)
            total_ask_vol = sum(w['volume_usd'] for w in whale_asks)
            
            # Yön ve güven
            if total_bid_vol > total_ask_vol * 1.5:
                direction = 'LONG'
                confidence = min(75, 50 + (total_bid_vol / total_ask_vol - 1) * 20) if total_ask_vol > 0 else 60
            elif total_ask_vol > total_bid_vol * 1.5:
                direction = 'SHORT'
                confidence = min(75, 50 + (total_ask_vol / total_bid_vol - 1) * 20) if total_bid_vol > 0 else 60
            else:
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'whale_bids': whale_bids[:5],  # Top 5
                'whale_asks': whale_asks[:5],
                'bid_count': len(whale_bids),
                'ask_count': len(whale_asks),
                'total_bid_volume': total_bid_vol,
                'total_ask_volume': total_ask_vol,
                'nearest_bid_wall': nearest_bid,
                'nearest_ask_wall': nearest_ask,
                'current_price': current_price,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Whale orders scraping failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'whale_bids': [],
            'whale_asks': [],
            'bid_count': 0,
            'ask_count': 0,
            'total_bid_volume': 0,
            'total_ask_volume': 0,
            'nearest_bid_wall': 0,
            'nearest_ask_wall': 0,
            'current_price': 0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_whale_orders(symbol: str = 'BTCUSDT') -> Dict:
    """Quick whale orders check."""
    scraper = CoinGlassWhaleOrders()
    return scraper.get_whale_orders(symbol)
