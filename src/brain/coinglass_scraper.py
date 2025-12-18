# -*- coding: utf-8 -*-
"""
DEMIR AI - CoinGlass Web Scraper
API rate limit sorunu için web scraping çözümü.

PHASE 116: Web Scraping for Rate-Limited APIs
- CGLiquidationMap
- CGWhaleOrders  
- CGOIDelta
- CGFundingExtreme
- CGExchangeBalance
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
import random

logger = logging.getLogger("CG_SCRAPER")


class CoinGlassWebScraper:
    """
    CoinGlass Web Scraper
    
    API rate limit olmadan veri çeker.
    """
    
    BASE_URL = "https://www.coinglass.com"
    
    # Cache
    CACHE_DURATION = 300  # 5 dakika
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    def __init__(self):
        self.cache: Dict[str, tuple] = {}
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        logger.info("✅ CoinGlass Web Scraper initialized")
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Cache'den al."""
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.CACHE_DURATION:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache'e yaz."""
        self.cache[key] = (time.time(), data)
    
    def get_liquidation_data(self, symbol: str = 'BTC') -> Dict:
        """
        Likidasyon verisi çek.
        CGLiquidationMap modülü için.
        """
        cache_key = f"liq_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # CoinGlass liquidation sayfası
            url = f"{self.BASE_URL}/LiquidationData"
            resp = self.session.get(url, timeout=15)
            
            if resp.status_code != 200:
                return self._fallback_liquidation()
            
            # JSON verisi script tag içinde
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Alternatif: API-benzeri endpoint dene
            api_url = "https://fapi.coinglass.com/api/futures/liquidation/info"
            api_resp = requests.get(api_url, headers=self.HEADERS, timeout=10)
            
            if api_resp.status_code == 200:
                data = api_resp.json()
                result = self._parse_liquidation_api(data, symbol)
                self._set_cache(cache_key, result)
                return result
            
            return self._fallback_liquidation()
            
        except Exception as e:
            logger.debug(f"Liquidation scrape failed: {e}")
            return self._fallback_liquidation()
    
    def _parse_liquidation_api(self, data: Dict, symbol: str) -> Dict:
        """Likidasyon API verisini parse et."""
        try:
            info = data.get('data', {})
            
            # 24h likidasyon
            total_liq_24h = info.get('liquidationUsd24h', 0)
            long_liq = info.get('longLiquidationUsd24h', 0)
            short_liq = info.get('shortLiquidationUsd24h', 0)
            
            # Dominant taraf
            if long_liq > short_liq * 1.5:
                direction = 'SHORT'  # Long'lar likide → fiyat düştü
                confidence = min(80, 50 + (long_liq / short_liq - 1) * 20)
            elif short_liq > long_liq * 1.5:
                direction = 'LONG'   # Short'lar likide → fiyat yükseldi
                confidence = min(80, 50 + (short_liq / long_liq - 1) * 20)
            else:
                direction = 'NEUTRAL'
                confidence = 45
            
            return {
                'available': True,
                'total_liquidation_24h': total_liq_24h,
                'long_liquidation': long_liq,
                'short_liquidation': short_liq,
                'direction': direction,
                'confidence': confidence,
                'source': 'coinglass_api'
            }
        except:
            return self._fallback_liquidation()
    
    def _fallback_liquidation(self) -> Dict:
        """Fallback veri."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_funding_rate(self, symbol: str = 'BTC') -> Dict:
        """
        Funding rate verisi.
        CGFundingExtreme modülü için.
        """
        cache_key = f"funding_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Binance futures API kullan (daha güvenilir)
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}USDT&limit=1"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    rate = float(data[0].get('fundingRate', 0))
                    rate_pct = rate * 100
                    
                    # Extreme funding tespiti
                    if rate_pct > 0.1:
                        direction = 'SHORT'  # Aşırı long = short fırsatı
                        confidence = min(75, 50 + rate_pct * 100)
                        extreme = True
                    elif rate_pct < -0.1:
                        direction = 'LONG'   # Aşırı short = long fırsatı
                        confidence = min(75, 50 + abs(rate_pct) * 100)
                        extreme = True
                    else:
                        direction = 'NEUTRAL'
                        confidence = 45
                        extreme = False
                    
                    result = {
                        'available': True,
                        'funding_rate': rate_pct,
                        'extreme': extreme,
                        'direction': direction,
                        'confidence': confidence,
                        'source': 'binance'
                    }
                    self._set_cache(cache_key, result)
                    return result
            
            return self._fallback_funding()
            
        except Exception as e:
            logger.debug(f"Funding scrape failed: {e}")
            return self._fallback_funding()
    
    def _fallback_funding(self) -> Dict:
        """Fallback funding."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_open_interest_delta(self, symbol: str = 'BTC') -> Dict:
        """
        OI değişim verisi.
        CGOIDelta modülü için.
        """
        cache_key = f"oi_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Binance OI
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}USDT"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                current_oi = float(data.get('openInterest', 0))
                
                # OI history için
                hist_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}USDT&period=5m&limit=12"
                hist_resp = requests.get(hist_url, timeout=10)
                
                if hist_resp.status_code == 200:
                    hist_data = hist_resp.json()
                    if hist_data:
                        old_oi = float(hist_data[0].get('sumOpenInterest', current_oi))
                        oi_change = ((current_oi - old_oi) / old_oi) * 100 if old_oi > 0 else 0
                        
                        # OI artışı = pozisyon açılıyor
                        if oi_change > 2:
                            direction = 'VOLATILE'  # Her iki yönde hareket olabilir
                            confidence = min(70, 50 + oi_change * 5)
                        elif oi_change < -2:
                            direction = 'NEUTRAL'   # Pozisyon kapanıyor
                            confidence = 45
                        else:
                            direction = 'NEUTRAL'
                            confidence = 40
                        
                        result = {
                            'available': True,
                            'current_oi': current_oi,
                            'oi_change_pct': round(oi_change, 2),
                            'direction': direction,
                            'confidence': confidence,
                            'source': 'binance'
                        }
                        self._set_cache(cache_key, result)
                        return result
            
            return self._fallback_oi()
            
        except Exception as e:
            logger.debug(f"OI scrape failed: {e}")
            return self._fallback_oi()
    
    def _fallback_oi(self) -> Dict:
        """Fallback OI."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_exchange_balance(self, symbol: str = 'BTC') -> Dict:
        """
        Borsa bakiye verisi.
        CGExchangeBalance modülü için.
        """
        cache_key = f"balance_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # CryptoQuant benzeri veri için CoinGlass alternatif
            # Public endpoint kullan
            url = "https://api.coinglass.com/api/index/exchange-balance"
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                # Parse logic
                pass
            
            # Fallback: Sabit değer (trend belirsiz)
            return self._fallback_balance()
            
        except Exception as e:
            logger.debug(f"Balance scrape failed: {e}")
            return self._fallback_balance()
    
    def _fallback_balance(self) -> Dict:
        """Fallback balance."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_whale_orders(self) -> Dict:
        """
        Whale emir verisi.
        CGWhaleOrders modülü için.
        """
        cache_key = "whale_orders"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Binance büyük emirler
            url = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=20"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                # Büyük emirler ($1M+)
                big_bid_volume = sum(float(b[0]) * float(b[1]) for b in bids[:10])
                big_ask_volume = sum(float(a[0]) * float(a[1]) for a in asks[:10])
                
                ratio = big_bid_volume / big_ask_volume if big_ask_volume > 0 else 1
                
                if ratio > 1.3:
                    direction = 'LONG'
                    confidence = min(70, 50 + (ratio - 1) * 30)
                elif ratio < 0.7:
                    direction = 'SHORT'
                    confidence = min(70, 50 + (1 - ratio) * 30)
                else:
                    direction = 'NEUTRAL'
                    confidence = 45
                
                result = {
                    'available': True,
                    'bid_volume': big_bid_volume,
                    'ask_volume': big_ask_volume,
                    'ratio': round(ratio, 2),
                    'direction': direction,
                    'confidence': confidence,
                    'source': 'binance_depth'
                }
                self._set_cache(cache_key, result)
                return result
            
            return self._fallback_whale()
            
        except Exception as e:
            logger.debug(f"Whale orders scrape failed: {e}")
            return self._fallback_whale()
    
    def _fallback_whale(self) -> Dict:
        """Fallback whale."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_all_data(self, symbol: str = 'BTC') -> Dict:
        """Tüm verileri topla."""
        return {
            'liquidation': self.get_liquidation_data(symbol),
            'funding': self.get_funding_rate(symbol),
            'oi_delta': self.get_open_interest_delta(symbol),
            'exchange_balance': self.get_exchange_balance(symbol),
            'whale_orders': self.get_whale_orders(),
            'timestamp': datetime.now().isoformat()
        }


# Global instance
_scraper = None

def get_cg_scraper() -> CoinGlassWebScraper:
    """Get or create scraper instance."""
    global _scraper
    if _scraper is None:
        _scraper = CoinGlassWebScraper()
    return _scraper
