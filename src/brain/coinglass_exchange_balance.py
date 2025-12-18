# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Exchange Balance Scraper
Borsa BTC bakiyesi değişimi.

PHASE 84: Exchange Balance Tracking
- Düşen bakiye = Hodl (Bullish)
- Artan bakiye = Satış baskısı (Bearish)
"""
import logging
import requests
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger("COINGLASS_EXCHANGE_BALANCE")


class CoinGlassExchangeBalance:
    """
    CoinGlass Exchange Balance Tracker
    
    Toplam borsa BTC bakiyesi değişimini takip eder.
    """
    
    # Glassnode alternatifi olarak OI ve volume'dan çıkarım
    BINANCE = "https://fapi.binance.com"
    
    def __init__(self):
        logger.info("✅ CoinGlass Exchange Balance initialized")
    
    def get_exchange_balance(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Exchange balance değişimi tahmin et.
        
        Returns:
            {
                'balance_trend': 'INCREASING'/'DECREASING'/'STABLE',
                'net_change_estimate': 500,  # BTC
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 55
            }
        """
        try:
            # OI değişiminden balance değişimi çıkarımı yap
            # OI artışı ≈ Borsaya giriş, OI düşüşü ≈ Borsadan çıkış
            
            oi_resp = requests.get(
                f"{self.BINANCE}/futures/data/openInterestHist",
                params={'symbol': symbol, 'period': '4h', 'limit': 7},  # Son 28 saat
                timeout=10
            )
            
            if oi_resp.status_code != 200:
                return self._empty_result()
            
            oi_data = oi_resp.json()
            
            if len(oi_data) < 2:
                return self._empty_result()
            
            # OI değerleri
            current_oi = float(oi_data[-1]['sumOpenInterestValue'])
            old_oi = float(oi_data[0]['sumOpenInterestValue'])
            oi_change = current_oi - old_oi
            oi_change_pct = ((current_oi / old_oi) - 1) * 100 if old_oi > 0 else 0
            
            # Tahmini BTC değişimi
            btc_price = current_oi / float(oi_data[-1]['sumOpenInterest']) if float(oi_data[-1]['sumOpenInterest']) > 0 else 85000
            net_btc_change = oi_change / btc_price
            
            # Volume değişimi (24h)
            vol_resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            
            volume_change = 0
            price_change = 0
            if vol_resp.status_code == 200:
                vol_data = vol_resp.json()
                volume_change = float(vol_data.get('volume', 0))
                price_change = float(vol_data.get('priceChangePercent', 0))
            
            # Balance trend
            if oi_change_pct > 3:
                balance_trend = 'INCREASING'
                direction = 'SHORT'  # Artan borsa bakiyesi = satış
                confidence = min(65, 45 + abs(oi_change_pct) * 3)
            elif oi_change_pct < -2:
                balance_trend = 'DECREASING'
                direction = 'LONG'  # Azalan borsa bakiyesi = hodl
                confidence = min(65, 45 + abs(oi_change_pct) * 3)
            else:
                balance_trend = 'STABLE'
                direction = 'NEUTRAL'
                confidence = 40
            
            return {
                'balance_trend': balance_trend,
                'oi_change_pct': oi_change_pct,
                'net_btc_change_estimate': net_btc_change,
                'price_change_24h': price_change,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Exchange balance failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'balance_trend': 'UNKNOWN',
            'oi_change_pct': 0,
            'net_btc_change_estimate': 0,
            'price_change_24h': 0,
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_exchange_balance(symbol: str = 'BTCUSDT') -> Dict:
    """Quick exchange balance check."""
    tracker = CoinGlassExchangeBalance()
    return tracker.get_exchange_balance(symbol)
