# -*- coding: utf-8 -*-
"""
DEMIR AI - Exchange Premium Scraper
Borsa fiyat farkı tespiti (Coinbase Premium).

PHASE 116: Web Scraping for Rate-Limited APIs
- Coinbase vs Binance fiyat farkı
- Kurumsal alım/satım göstergesi
"""
import logging
import requests
from datetime import datetime
from typing import Dict, Optional
import time

logger = logging.getLogger("EXCHANGE_PREMIUM")


class ExchangePremiumScraper:
    """
    Exchange Premium Scraper
    
    Coinbase ve Binance arasındaki fiyat farkını tespit eder.
    Pozitif premium = ABD kurumsal alımı
    Negatif premium = ABD satış baskısı
    """
    
    CACHE_DURATION = 60  # 1 dakika
    
    # Premium thresholds
    PREMIUM_THRESHOLD = 0.3  # %0.3 premium = anlamlı fark
    
    def __init__(self):
        self.cache: Dict[str, tuple] = {}
        logger.info("✅ Exchange Premium Scraper initialized")
    
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
    
    def get_coinbase_premium(self, symbol: str = 'BTC') -> Dict:
        """
        Coinbase premium hesapla.
        
        Coinbase fiyatı > Binance fiyatı = ABD alımı (bullish)
        Coinbase fiyatı < Binance fiyatı = ABD satışı (bearish)
        """
        cache_key = f"premium_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Binance fiyatı
            binance_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
            binance_resp = requests.get(binance_url, timeout=5)
            
            if binance_resp.status_code != 200:
                return self._fallback()
            
            binance_price = float(binance_resp.json()['price'])
            
            # Coinbase fiyatı (public API)
            coinbase_url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
            coinbase_resp = requests.get(coinbase_url, timeout=5)
            
            if coinbase_resp.status_code != 200:
                # Coinbase API başarısız - Kraken dene
                kraken_url = f"https://api.kraken.com/0/public/Ticker?pair={symbol}USD"
                kraken_resp = requests.get(kraken_url, timeout=5)
                
                if kraken_resp.status_code == 200:
                    kraken_data = kraken_resp.json()
                    result = kraken_data.get('result', {})
                    pair_key = list(result.keys())[0] if result else None
                    if pair_key:
                        alt_price = float(result[pair_key]['c'][0])
                    else:
                        return self._fallback()
                else:
                    return self._fallback()
            else:
                coinbase_data = coinbase_resp.json()
                alt_price = float(coinbase_data['data']['amount'])
            
            # Premium hesapla
            premium_pct = ((alt_price - binance_price) / binance_price) * 100
            
            # Sinyal
            if premium_pct > self.PREMIUM_THRESHOLD:
                direction = 'LONG'
                confidence = min(75, 50 + abs(premium_pct) * 20)
                signal_text = "ABD kurumsal alımı"
            elif premium_pct < -self.PREMIUM_THRESHOLD:
                direction = 'SHORT'
                confidence = min(75, 50 + abs(premium_pct) * 20)
                signal_text = "ABD satış baskısı"
            else:
                direction = 'NEUTRAL'
                confidence = 45
                signal_text = "Fark yok"
            
            result = {
                'available': True,
                'binance_price': binance_price,
                'alt_price': alt_price,
                'premium_pct': round(premium_pct, 3),
                'direction': direction,
                'confidence': confidence,
                'signal_text': signal_text,
                'source': 'coinbase_binance'
            }
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.debug(f"Exchange premium failed: {e}")
            return self._fallback()
    
    def _fallback(self) -> Dict:
        """Fallback veri."""
        return {
            'available': False,
            'direction': 'NEUTRAL',
            'confidence': 40,
            'source': 'fallback'
        }
    
    def get_signal(self, symbol: str = 'BTC') -> Dict:
        """Modül sinyali formatında döndür."""
        premium = self.get_coinbase_premium(symbol)
        return {
            'module': 'ExchangeDivergence',
            'direction': premium.get('direction', 'NEUTRAL'),
            'confidence': premium.get('confidence', 40),
            'data': premium
        }


# Global instance
_ep = None

def get_exchange_premium() -> ExchangePremiumScraper:
    """Get or create Exchange Premium scraper."""
    global _ep
    if _ep is None:
        _ep = ExchangePremiumScraper()
    return _ep
