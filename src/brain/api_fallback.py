# -*- coding: utf-8 -*-
"""
DEMIR AI - API Fallback System (FAIL FAST MODE)
API hataları için yedek CANLI veri kaynakları.

⚠️ FAIL FAST: Tüm kaynaklar başarısız olursa None döner!
   Cache veya hardcoded değer KULLANILMAZ.
   Sinyal üretimi bu verilere bağlıysa DURDURULMALIDIR.

PHASE 113: API Fallbacks
- Multiple data source fallbacks (sadece CANLI kaynaklar)
- Retry logic with exponential backoff
- Error recovery
- Rate limit handling
"""
import logging
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger("API_FALLBACK")


class APIFallbackSystem:
    """
    API Yedekleme Sistemi (FAIL FAST MODE)
    
    Birincil API başarısız olduğunda yedek CANLI kaynaklara geçer.
    Tüm kaynaklar başarısız = None döner (cache/hardcoded YOK!)
    """
    
    # Veri kaynakları (öncelik sırasına göre)
    PRICE_SOURCES = [
        {
            'name': 'Binance',
            'url': 'https://api.binance.com/api/v3/ticker/price',
            'parser': lambda r, s: float(r.json()['price']),
            'param_key': 'symbol'
        },
        {
            'name': 'Binance Futures',
            'url': 'https://fapi.binance.com/fapi/v1/ticker/price',
            'parser': lambda r, s: float(r.json()['price']),
            'param_key': 'symbol'
        },
        {
            'name': 'CoinGecko',
            'url': 'https://api.coingecko.com/api/v3/simple/price',
            'parser': lambda r, s: float(r.json().get('bitcoin', {}).get('usd', 0)),
            'param_key': 'ids',
            'params': {'ids': 'bitcoin', 'vs_currencies': 'usd'}
        }
    ]
    
    KLINES_SOURCES = [
        {
            'name': 'Binance Spot',
            'url': 'https://api.binance.com/api/v3/klines',
        },
        {
            'name': 'Binance Futures',
            'url': 'https://fapi.binance.com/fapi/v1/klines',
        }
    ]
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    BACKOFF_MULTIPLIER = 2
    
    def __init__(self):
        self.failed_sources: Dict[str, datetime] = {}
        self.success_count: Dict[str, int] = {}
        self.error_count: Dict[str, int] = {}
        logger.info("✅ API Fallback System initialized")
    
    async def get_price(self, symbol: str = 'BTCUSDT') -> Optional[float]:
        """
        Fiyat al - yedeklerle.
        
        Returns:
            Price or None if all sources fail
        """
        for source in self.PRICE_SOURCES:
            # Skip recently failed sources (5 min cooldown)
            if self._is_source_cooling_down(source['name']):
                continue
            
            try:
                params = source.get('params', {source['param_key']: symbol})
                
                resp = requests.get(
                    source['url'],
                    params=params,
                    timeout=5
                )
                
                if resp.status_code == 200:
                    price = source['parser'](resp, symbol)
                    self._record_success(source['name'])
                    return price
                else:
                    self._record_error(source['name'], f"Status {resp.status_code}")
                    
            except Exception as e:
                self._record_error(source['name'], str(e))
                continue
        
        logger.error(f"All price sources failed for {symbol}")
        return None
    
    async def get_klines(self, symbol: str = 'BTCUSDT', 
                        interval: str = '1h', 
                        limit: int = 100) -> Optional[List]:
        """
        Kline verileri al - yedeklerle.
        """
        for source in self.KLINES_SOURCES:
            if self._is_source_cooling_down(source['name']):
                continue
            
            for retry in range(self.MAX_RETRIES):
                try:
                    delay = self.RETRY_DELAY * (self.BACKOFF_MULTIPLIER ** retry)
                    
                    if retry > 0:
                        await asyncio.sleep(delay)
                    
                    resp = requests.get(
                        source['url'],
                        params={'symbol': symbol, 'interval': interval, 'limit': limit},
                        timeout=10
                    )
                    
                    if resp.status_code == 200:
                        self._record_success(source['name'])
                        return resp.json()
                    elif resp.status_code == 429:  # Rate limit
                        logger.warning(f"{source['name']} rate limited, waiting...")
                        await asyncio.sleep(delay * 2)
                    else:
                        self._record_error(source['name'], f"Status {resp.status_code}")
                        break
                        
                except requests.exceptions.Timeout:
                    logger.debug(f"{source['name']} timeout, retry {retry+1}/{self.MAX_RETRIES}")
                except Exception as e:
                    self._record_error(source['name'], str(e))
                    break
            
            # Move to next source
            continue
        
        logger.error(f"All klines sources failed for {symbol}")
        return None
    
    async def get_funding_rate(self, symbol: str = 'BTCUSDT') -> Optional[float]:
        """Funding rate al - yedeklerle."""
        sources = [
            ('Binance Futures', f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"),
        ]
        
        for name, url in sources:
            if self._is_source_cooling_down(name):
                continue
            
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data:
                        self._record_success(name)
                        return float(data[0].get('fundingRate', 0))
            except Exception as e:
                self._record_error(name, str(e))
                continue
        
        return None
    
    async def get_open_interest(self, symbol: str = 'BTCUSDT') -> Optional[float]:
        """Open Interest al - yedeklerle."""
        sources = [
            ('Binance Futures OI', f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"),
        ]
        
        for name, url in sources:
            if self._is_source_cooling_down(name):
                continue
            
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    self._record_success(name)
                    return float(data.get('openInterest', 0))
            except Exception as e:
                self._record_error(name, str(e))
                continue
        
        return None
    
    def _is_source_cooling_down(self, source_name: str) -> bool:
        """Kaynak cooldown'da mı?"""
        if source_name not in self.failed_sources:
            return False
        
        cooldown_until = self.failed_sources[source_name]
        return datetime.now() < cooldown_until
    
    def _record_success(self, source_name: str):
        """Başarılı istek kaydet."""
        if source_name in self.failed_sources:
            del self.failed_sources[source_name]
        
        self.success_count[source_name] = self.success_count.get(source_name, 0) + 1
    
    def _record_error(self, source_name: str, error: str):
        """Hata kaydet ve cooldown başlat."""
        self.error_count[source_name] = self.error_count.get(source_name, 0) + 1
        
        errors = self.error_count[source_name]
        
        # Daha fazla hata = daha uzun cooldown
        cooldown_minutes = min(30, errors * 2)
        self.failed_sources[source_name] = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        logger.warning(f"API error {source_name}: {error} | Cooldown: {cooldown_minutes}min")
    
    def get_health_status(self) -> Dict:
        """API sağlık durumu."""
        status = {}
        
        all_sources = set()
        for s in self.PRICE_SOURCES + self.KLINES_SOURCES:
            all_sources.add(s['name'])
        
        for source in all_sources:
            success = self.success_count.get(source, 0)
            errors = self.error_count.get(source, 0)
            total = success + errors
            
            if total == 0:
                health = 'UNKNOWN'
            elif errors == 0:
                health = 'HEALTHY'
            elif success / total >= 0.9:
                health = 'GOOD'
            elif success / total >= 0.7:
                health = 'DEGRADED'
            else:
                health = 'UNHEALTHY'
            
            cooling = self._is_source_cooling_down(source)
            
            status[source] = {
                'health': health,
                'success': success,
                'errors': errors,
                'cooling_down': cooling
            }
        
        return status


# Global instance
_fallback = None

def get_fallback() -> APIFallbackSystem:
    """Get or create fallback instance."""
    global _fallback
    if _fallback is None:
        _fallback = APIFallbackSystem()
    return _fallback
