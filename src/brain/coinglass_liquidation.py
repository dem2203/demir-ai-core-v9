# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Liquidation Map Scraper
Likidasyon haritası - Hangi fiyatlarda likidasyon patlaması olacak.

PHASE 77: Liquidation Map Scraping
- Web scraping (API key gerektirmez)
- Long/Short likidasyon seviyeleri
- Cascade risk tahmini
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional
import json
import re

logger = logging.getLogger("COINGLASS_LIQUIDATION")


class CoinGlassLiquidation:
    """
    CoinGlass Liquidation Map Scraper
    
    Hangi fiyat seviyelerinde likidasyon yoğun olduğunu tespit eder.
    Fiyat bu bölgelere yaklaştığında cascade riski artar.
    """
    
    BASE_URL = "https://www.coinglass.com"
    
    # Alternatif: CoinGlass'ın gizli API'si (web'de kullanılan)
    HIDDEN_API = "https://fapi.coinglass.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.coinglass.com/'
        })
        logger.info("✅ CoinGlass Liquidation Scraper initialized")
    
    def get_liquidation_levels(self, symbol: str = 'BTC') -> Dict:
        """
        Likidasyon seviyelerini al.
        
        Returns:
            {
                'long_liquidations': {price: volume},  # Long liq at price
                'short_liquidations': {price: volume}, # Short liq at price
                'nearest_long_liq': price,  # En yakın long liq
                'nearest_short_liq': price, # En yakın short liq
                'cascade_risk': 'HIGH'/'MEDIUM'/'LOW',
                'direction': 'LONG'/'SHORT'/'NEUTRAL',
                'confidence': 65
            }
        """
        try:
            # Yöntem 1: CoinGlass hidden API dene
            result = self._try_hidden_api(symbol)
            if result.get('available'):
                return result
            
            # Yöntem 2: Binance Futures doğrudan kullan
            result = self._binance_liquidation_estimate(symbol)
            return result
            
        except Exception as e:
            logger.warning(f"Liquidation scraping failed: {e}")
            return self._empty_result()
    
    def _try_hidden_api(self, symbol: str) -> Dict:
        """CoinGlass'ın gizli API'sini dene."""
        try:
            # Liquidation chart data endpoint
            url = f"{self.HIDDEN_API}/liquidation/liquidation-chart-data"
            params = {
                'symbol': symbol,
                'type': '1',  # 1 = BTC
                'interval': '1d'
            }
            
            resp = self.session.get(url, params=params, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get('success') and data.get('data'):
                    return self._parse_liquidation_data(data['data'], symbol)
            
            return {'available': False}
            
        except Exception as e:
            logger.debug(f"Hidden API failed: {e}")
            return {'available': False}
    
    def _parse_liquidation_data(self, data: dict, symbol: str) -> Dict:
        """API verisini parse et."""
        try:
            # Mevcut fiyatı al
            price_resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': f'{symbol}USDT'},
                timeout=5
            )
            current_price = float(price_resp.json()['price'])
            
            # Long ve Short likidasyon verilerini al
            long_liqs = data.get('longLiquidation', [])
            short_liqs = data.get('shortLiquidation', [])
            
            # En yakın likidasyon seviyelerini bul
            nearest_long = None
            nearest_short = None
            
            for liq in long_liqs:
                price = liq.get('price', 0)
                if price < current_price:
                    if nearest_long is None or price > nearest_long:
                        nearest_long = price
            
            for liq in short_liqs:
                price = liq.get('price', 0)
                if price > current_price:
                    if nearest_short is None or price < nearest_short:
                        nearest_short = price
            
            # Cascade riski hesapla
            distance_to_long = abs(current_price - nearest_long) / current_price * 100 if nearest_long else 100
            distance_to_short = abs(nearest_short - current_price) / current_price * 100 if nearest_short else 100
            
            if distance_to_long < 2 or distance_to_short < 2:
                cascade_risk = 'HIGH'
                confidence = 70
            elif distance_to_long < 5 or distance_to_short < 5:
                cascade_risk = 'MEDIUM'
                confidence = 55
            else:
                cascade_risk = 'LOW'
                confidence = 40
            
            # Yön belirle
            if distance_to_long < distance_to_short:
                direction = 'SHORT'  # Long liq'a yakın = aşağı cascade
            else:
                direction = 'LONG'  # Short liq'a yakın = yukarı cascade
            
            return {
                'nearest_long_liq': nearest_long,
                'nearest_short_liq': nearest_short,
                'distance_to_long_pct': distance_to_long,
                'distance_to_short_pct': distance_to_short,
                'current_price': current_price,
                'cascade_risk': cascade_risk,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Parse liquidation data failed: {e}")
            return {'available': False}
    
    def _binance_liquidation_estimate(self, symbol: str) -> Dict:
        """Binance'den likidasyon tahmini yap."""
        try:
            # OI ve Funding Rate'den likidasyon riski tahmin et
            symbol_full = f"{symbol}USDT"
            
            # Mevcut fiyat
            price_resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol_full},
                timeout=5
            )
            current_price = float(price_resp.json()['price'])
            
            # Funding Rate
            funding_resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol_full, 'limit': 1},
                timeout=5
            )
            funding_rate = float(funding_resp.json()[0]['fundingRate']) if funding_resp.status_code == 200 else 0
            
            # Long/Short Ratio
            ls_resp = requests.get(
                f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol_full, 'period': '5m', 'limit': 1},
                timeout=5
            )
            ls_ratio = float(ls_resp.json()[0]['longShortRatio']) if ls_resp.status_code == 200 else 1.0
            
            # Likidasyon seviyeleri tahmin et
            # Long likidasyon genellikle fiyatın %2-5 altında
            # Short likidasyon genellikle fiyatın %2-5 üstünde
            nearest_long = current_price * (1 - 0.02 - abs(funding_rate) * 10)  # Funding high = closer liq
            nearest_short = current_price * (1 + 0.02 + abs(funding_rate) * 10)
            
            # Cascade riski
            if abs(funding_rate) > 0.05:
                cascade_risk = 'HIGH'
                confidence = 65
            elif abs(funding_rate) > 0.02:
                cascade_risk = 'MEDIUM'
                confidence = 50
            else:
                cascade_risk = 'LOW'
                confidence = 40
            
            # Yön
            if funding_rate > 0.03:  # High positive = Long crowded
                direction = 'SHORT'
            elif funding_rate < -0.02:  # Negative = Short crowded
                direction = 'LONG'
            else:
                direction = 'NEUTRAL'
            
            return {
                'nearest_long_liq': nearest_long,
                'nearest_short_liq': nearest_short,
                'distance_to_long_pct': abs(current_price - nearest_long) / current_price * 100,
                'distance_to_short_pct': abs(nearest_short - current_price) / current_price * 100,
                'current_price': current_price,
                'funding_rate': funding_rate,
                'ls_ratio': ls_ratio,
                'cascade_risk': cascade_risk,
                'direction': direction,
                'confidence': confidence,
                'available': True
            }
            
        except Exception as e:
            logger.warning(f"Binance liquidation estimate failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        return {
            'nearest_long_liq': 0,
            'nearest_short_liq': 0,
            'distance_to_long_pct': 0,
            'distance_to_short_pct': 0,
            'current_price': 0,
            'cascade_risk': 'LOW',
            'direction': 'NEUTRAL',
            'confidence': 0,
            'available': False
        }


# Convenience function
def get_liquidation_levels(symbol: str = 'BTC') -> Dict:
    """Quick liquidation check."""
    scraper = CoinGlassLiquidation()
    return scraper.get_liquidation_levels(symbol)
