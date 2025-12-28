# -*- coding: utf-8 -*-
"""
DEMIR AI - MULTI-EXCHANGE CONNECTOR
===================================
Birden fazla borsayı destekler: Binance, Bybit, OKX

Kullanım:
- Arbitraj fırsatları
- Fiyat karşılaştırması
- Fallback data source
"""
import logging
import aiohttp
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger("MULTI_EXCHANGE")


class BybitConnector:
    """Bybit Futures API connector"""
    
    BASE_URL = "https://api.bybit.com"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "User-Agent": "DEMIR-AI/1.0",
            "Accept": "application/json"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Get current ticker data
        
        Returns:
            {'symbol': 'BTCUSDT', 'price': 87500, 'change_24h': 2.5, ...}
        """
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        ticker = data['result']['list'][0]
                        return {
                            'symbol': symbol,
                            'price': float(ticker.get('lastPrice', 0)),
                            'change_24h': float(ticker.get('price24hPcnt', 0)) * 100,
                            'volume_24h': float(ticker.get('volume24h', 0)),
                            'high_24h': float(ticker.get('highPrice24h', 0)),
                            'low_24h': float(ticker.get('lowPrice24h', 0)),
                            'source': 'bybit'
                        }
        except Exception as e:
            logger.error(f"Bybit ticker error: {e}")
        return {}
    
    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """Get current funding rate"""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/v5/market/funding/history"
            params = {"category": "linear", "symbol": symbol, "limit": 1}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        return float(data['result']['list'][0].get('fundingRate', 0))
        except Exception as e:
            logger.error(f"Bybit funding error: {e}")
        return 0
    
    async def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """Get open interest"""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/v5/market/open-interest"
            params = {"category": "linear", "symbol": symbol, "intervalTime": "5min", "limit": 1}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        return float(data['result']['list'][0].get('openInterest', 0))
        except Exception as e:
            logger.error(f"Bybit OI error: {e}")
        return 0


class OKXConnector:
    """OKX API connector"""
    
    BASE_URL = "https://www.okx.com"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "User-Agent": "DEMIR-AI/1.0",
            "Accept": "application/json"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_ticker(self, symbol: str = "BTC-USDT-SWAP") -> Dict:
        """Get current ticker"""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/api/v5/market/ticker"
            params = {"instId": symbol}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        ticker = data['data'][0]
                        return {
                            'symbol': symbol,
                            'price': float(ticker.get('last', 0)),
                            'change_24h': float(ticker.get('sodUtc8', 0)),
                            'volume_24h': float(ticker.get('vol24h', 0)),
                            'source': 'okx'
                        }
        except Exception as e:
            logger.error(f"OKX ticker error: {e}")
        return {}


class MultiExchangeManager:
    """Manages multiple exchange connections"""
    
    SYMBOL_MAPPING = {
        'BTCUSDT': {
            'binance': 'BTCUSDT',
            'bybit': 'BTCUSDT',
            'okx': 'BTC-USDT-SWAP'
        },
        'ETHUSDT': {
            'binance': 'ETHUSDT',
            'bybit': 'ETHUSDT',
            'okx': 'ETH-USDT-SWAP'
        },
        'SOLUSDT': {
            'binance': 'SOLUSDT',
            'bybit': 'SOLUSDT',
            'okx': 'SOL-USDT-SWAP'
        }
    }
    
    def __init__(self):
        self.bybit = BybitConnector()
        self.okx = OKXConnector()
    
    async def close(self):
        await self.bybit.close()
        await self.okx.close()
    
    async def get_all_prices(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Get price from all exchanges
        
        Returns:
            {'binance': 87500, 'bybit': 87510, 'okx': 87495}
        """
        mapping = self.SYMBOL_MAPPING.get(symbol, {})
        prices = {}
        
        # Bybit
        bybit_symbol = mapping.get('bybit', symbol)
        bybit_data = await self.bybit.get_ticker(bybit_symbol)
        if bybit_data.get('price'):
            prices['bybit'] = bybit_data['price']
        
        # OKX
        okx_symbol = mapping.get('okx', f"{symbol[:3]}-{symbol[3:]}-SWAP")
        okx_data = await self.okx.get_ticker(okx_symbol)
        if okx_data.get('price'):
            prices['okx'] = okx_data['price']
        
        return prices
    
    async def find_arbitrage(self, symbol: str = "BTCUSDT", binance_price: float = 0) -> Dict:
        """
        Find arbitrage opportunities
        
        Returns:
            {
                'has_opportunity': True/False,
                'best_buy': 'bybit',
                'best_sell': 'binance',
                'spread_pct': 0.15,
                'prices': {...}
            }
        """
        prices = await self.get_all_prices(symbol)
        
        if binance_price > 0:
            prices['binance'] = binance_price
        
        if len(prices) < 2:
            return {'has_opportunity': False, 'reason': 'Not enough exchange data'}
        
        min_exchange = min(prices, key=prices.get)
        max_exchange = max(prices, key=prices.get)
        
        spread = prices[max_exchange] - prices[min_exchange]
        spread_pct = (spread / prices[min_exchange]) * 100
        
        # Arbitrage threshold: 0.1%
        has_opportunity = spread_pct >= 0.1
        
        return {
            'has_opportunity': has_opportunity,
            'best_buy': min_exchange,
            'best_sell': max_exchange,
            'spread_pct': spread_pct,
            'spread_usd': spread,
            'prices': prices
        }


# Singleton
_multi_exchange: Optional[MultiExchangeManager] = None


def get_multi_exchange() -> MultiExchangeManager:
    """Get or create MultiExchangeManager singleton"""
    global _multi_exchange
    if _multi_exchange is None:
        _multi_exchange = MultiExchangeManager()
    return _multi_exchange
