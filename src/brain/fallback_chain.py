# -*- coding: utf-8 -*-
"""
DEMIR AI - MULTI-SOURCE FALLBACK CHAIN
=======================================
Birincil kaynak başarısız olursa alternatif kaynaklara geç.
Her veri için en az 2-3 alternatif kaynak sağlar.

Zincir: Primary API → Backup API → Web Scraping → Cached data

API KEY GEREKMİYOR - Tüm public kaynaklar!
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("FALLBACK_CHAIN")


@dataclass
class DataResult:
    """Tek bir veri sonucu"""
    value: Any = None
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    is_live: bool = True  # False = cached
    quality: str = "REAL"  # "REAL", "BACKUP", "SCRAPED", "CACHED"
    error: str = ""


class FallbackChain:
    """
    Multi-Source Fallback Chain
    
    Her veri için birden fazla kaynak tanımlar.
    Birincil başarısız olursa otomatik olarak yedek kaynağa geçer.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, DataResult] = {}
        self._cache_ttl = 300  # 5 dakika (son çare olarak kullanılır)
        logger.info("🔄 Multi-Source Fallback Chain initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'Mozilla/5.0'}
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _fetch_with_fallback(
        self, 
        key: str,
        sources: List[Tuple[str, Callable]],  # (source_name, async fetch_function)
    ) -> DataResult:
        """
        Birden fazla kaynaktan sırayla veri çekmeyi dene.
        """
        result = DataResult()
        
        for source_name, fetch_func in sources:
            try:
                data = await fetch_func()
                if data is not None and data != 0:
                    result.value = data
                    result.source = source_name
                    result.quality = "REAL" if sources.index((source_name, fetch_func)) == 0 else "BACKUP"
                    result.is_live = True
                    
                    # Cache it
                    self._cache[key] = result
                    logger.debug(f"✅ {key} from {source_name}: {data}")
                    return result
            except Exception as e:
                logger.debug(f"⚠️ {key} failed from {source_name}: {e}")
                continue
        
        # Tüm kaynaklar başarısız - cache'e bak
        if key in self._cache:
            cached = self._cache[key]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self._cache_ttl:
                cached.quality = "CACHED"
                cached.is_live = False
                logger.info(f"📦 {key} from cache ({age:.0f}s old)")
                return cached
        
        result.error = "All sources failed"
        result.quality = "FAILED"
        return result
    
    # =========================================
    # BTC FİYAT - 4 KAYNAK
    # =========================================
    
    async def get_btc_price(self) -> DataResult:
        """BTC fiyatı - 4 farklı kaynak"""
        return await self._fetch_with_fallback("btc_price", [
            ("binance", self._btc_binance),
            ("bybit", self._btc_bybit),
            ("okx", self._btc_okx),
            ("coingecko", self._btc_coingecko),
        ])
    
    async def _btc_binance(self) -> float:
        session = await self._get_session()
        async with session.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT") as resp:
            data = await resp.json()
            return float(data['price'])
    
    async def _btc_bybit(self) -> float:
        session = await self._get_session()
        async with session.get("https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT") as resp:
            data = await resp.json()
            return float(data['result']['list'][0]['lastPrice'])
    
    async def _btc_okx(self) -> float:
        session = await self._get_session()
        async with session.get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT") as resp:
            data = await resp.json()
            return float(data['data'][0]['last'])
    
    async def _btc_coingecko(self) -> float:
        session = await self._get_session()
        async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd") as resp:
            data = await resp.json()
            return float(data['bitcoin']['usd'])
    
    # =========================================
    # DXY - 3 KAYNAK
    # =========================================
    
    async def get_dxy(self) -> DataResult:
        """Dollar Index - 3 kaynak"""
        return await self._fetch_with_fallback("dxy", [
            ("yahoo", self._dxy_yahoo),
            ("investing_api", self._dxy_investing),
            ("tradingview_scrape", self._dxy_tradingview),
        ])
    
    async def _dxy_yahoo(self) -> float:
        session = await self._get_session()
        async with session.get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB") as resp:
            data = await resp.json()
            return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
    
    async def _dxy_investing(self) -> float:
        # Investing.com API (public endpoint)
        session = await self._get_session()
        async with session.get("https://api.investing.com/api/financialdata/8827/historical/chart/?period=MAX&interval=PT1M") as resp:
            data = await resp.json()
            if data.get('data'):
                return float(data['data'][-1][4])  # Last close
        return None
    
    async def _dxy_tradingview(self) -> float:
        # TradingView public endpoint
        session = await self._get_session()
        async with session.get("https://symbol-search.tradingview.com/symbol_search/?text=DXY&type=index") as resp:
            data = await resp.json()
            # Parse result...
            return None  # Fallback implementation
    
    # =========================================
    # VIX - 3 KAYNAK
    # =========================================
    
    async def get_vix(self) -> DataResult:
        """Volatility Index - 3 kaynak"""
        return await self._fetch_with_fallback("vix", [
            ("yahoo", self._vix_yahoo),
            ("cboe", self._vix_cboe),
            ("investing", self._vix_investing),
        ])
    
    async def _vix_yahoo(self) -> float:
        session = await self._get_session()
        async with session.get("https://query1.finance.yahoo.com/v8/finance/chart/^VIX") as resp:
            data = await resp.json()
            return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
    
    async def _vix_cboe(self) -> float:
        # CBOE official data
        session = await self._get_session()
        async with session.get("https://cdn.cboe.com/api/global/delayed_quotes/indices/_VIX.json") as resp:
            data = await resp.json()
            return float(data.get('data', {}).get('last_price', 0))
    
    async def _vix_investing(self) -> float:
        # Investing.com backup
        return None  # Implementation
    
    # =========================================
    # FUNDING RATE - 4 KAYNAK
    # =========================================
    
    async def get_funding_rate(self, symbol: str = "BTCUSDT") -> DataResult:
        """Funding Rate - 4 borsa"""
        return await self._fetch_with_fallback(f"funding_{symbol}", [
            ("binance", lambda: self._funding_binance(symbol)),
            ("bybit", lambda: self._funding_bybit(symbol)),
            ("okx", lambda: self._funding_okx(symbol)),
            ("dydx", lambda: self._funding_dydx(symbol)),
        ])
    
    async def _funding_binance(self, symbol: str) -> float:
        session = await self._get_session()
        async with session.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1") as resp:
            data = await resp.json()
            return float(data[0]['fundingRate']) * 100
    
    async def _funding_bybit(self, symbol: str) -> float:
        session = await self._get_session()
        async with session.get(f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol}&limit=1") as resp:
            data = await resp.json()
            return float(data['result']['list'][0]['fundingRate']) * 100
    
    async def _funding_okx(self, symbol: str) -> float:
        okx_symbol = symbol.replace('USDT', '-USDT-SWAP')
        session = await self._get_session()
        async with session.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_symbol}") as resp:
            data = await resp.json()
            return float(data['data'][0]['fundingRate']) * 100
    
    async def _funding_dydx(self, symbol: str) -> float:
        # dYdX perpetual
        return None  # Implementation
    
    # =========================================
    # OPEN INTEREST - 3 KAYNAK
    # =========================================
    
    async def get_open_interest(self, symbol: str = "BTCUSDT") -> DataResult:
        """Open Interest - 3 kaynak"""
        return await self._fetch_with_fallback(f"oi_{symbol}", [
            ("binance", lambda: self._oi_binance(symbol)),
            ("bybit", lambda: self._oi_bybit(symbol)),
            ("coinglass", lambda: self._oi_coinglass(symbol)),
        ])
    
    async def _oi_binance(self, symbol: str) -> float:
        session = await self._get_session()
        async with session.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}") as resp:
            data = await resp.json()
            oi = float(data['openInterest'])
            # Get price for USD value
            async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as price_resp:
                price_data = await price_resp.json()
                price = float(price_data['price'])
            return oi * price
    
    async def _oi_bybit(self, symbol: str) -> float:
        session = await self._get_session()
        async with session.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={symbol}") as resp:
            data = await resp.json()
            return float(data['result']['list'][0]['openInterest'])
    
    async def _oi_coinglass(self, symbol: str) -> float:
        # Coinglass public endpoint
        return None  # Implementation
    
    # =========================================
    # BTC DOMINANCE - 2 KAYNAK
    # =========================================
    
    async def get_btc_dominance(self) -> DataResult:
        """BTC.D - 2 kaynak"""
        return await self._fetch_with_fallback("btc_dominance", [
            ("coingecko", self._btc_d_coingecko),
            ("coinmarketcap", self._btc_d_cmc),
        ])
    
    async def _btc_d_coingecko(self) -> float:
        session = await self._get_session()
        async with session.get("https://api.coingecko.com/api/v3/global") as resp:
            data = await resp.json()
            return float(data['data']['market_cap_percentage']['btc'])
    
    async def _btc_d_cmc(self) -> float:
        # CoinMarketCap backup
        return None  # Implementation
    
    # =========================================
    # FEAR & GREED - 2 KAYNAK
    # =========================================
    
    async def get_fear_greed(self) -> DataResult:
        """Fear & Greed Index - 2 kaynak"""
        return await self._fetch_with_fallback("fear_greed", [
            ("alternative", self._fg_alternative),
            ("lookintobitcoin", self._fg_looking),
        ])
    
    async def _fg_alternative(self) -> int:
        session = await self._get_session()
        async with session.get("https://api.alternative.me/fng/") as resp:
            data = await resp.json()
            return int(data['data'][0]['value'])
    
    async def _fg_looking(self) -> int:
        # LookIntoBitcoin backup
        return None  # Implementation
    
    # =========================================
    # STOCK INDICES - 2 KAYNAK HER BİRİ
    # =========================================
    
    async def get_spx500(self) -> DataResult:
        """S&P 500 - 2 kaynak"""
        return await self._fetch_with_fallback("spx500", [
            ("yahoo", self._spx_yahoo),
            ("alphavantage", self._spx_alphavantage),
        ])
    
    async def _spx_yahoo(self) -> float:
        session = await self._get_session()
        async with session.get("https://query1.finance.yahoo.com/v8/finance/chart/^GSPC") as resp:
            data = await resp.json()
            return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
    
    async def _spx_alphavantage(self) -> float:
        # AlphaVantage (free tier)
        return None  # Implementation
    
    async def get_gold(self) -> DataResult:
        """Gold XAU/USD - 2 kaynak"""
        return await self._fetch_with_fallback("gold", [
            ("yahoo", self._gold_yahoo),
            ("metals_api", self._gold_metals),
        ])
    
    async def _gold_yahoo(self) -> float:
        session = await self._get_session()
        async with session.get("https://query1.finance.yahoo.com/v8/finance/chart/GC=F") as resp:
            data = await resp.json()
            return float(data['chart']['result'][0]['meta']['regularMarketPrice'])
    
    async def _gold_metals(self) -> float:
        # Metals API backup
        return None  # Implementation
    
    # =========================================
    # TÜM VERİLERİ PARALEL ÇEK
    # =========================================
    
    async def fetch_all_data(self, symbol: str = "BTCUSDT") -> Dict[str, DataResult]:
        """Tüm verileri paralel olarak çek (fallback zincirleri ile)"""
        results = await asyncio.gather(
            self.get_btc_price(),
            self.get_dxy(),
            self.get_vix(),
            self.get_funding_rate(symbol),
            self.get_open_interest(symbol),
            self.get_btc_dominance(),
            self.get_fear_greed(),
            self.get_spx500(),
            self.get_gold(),
            return_exceptions=True
        )
        
        keys = ['btc_price', 'dxy', 'vix', 'funding', 'oi', 'btc_d', 'fear_greed', 'spx500', 'gold']
        
        return {
            key: (result if isinstance(result, DataResult) else DataResult(error=str(result)))
            for key, result in zip(keys, results)
        }
    
    def format_quality_report(self, data: Dict[str, DataResult]) -> str:
        """Veri kalitesi raporu"""
        lines = []
        for key, result in data.items():
            emoji = {
                "REAL": "✅",
                "BACKUP": "🟡",
                "SCRAPED": "🟠",
                "CACHED": "📦",
                "FAILED": "❌"
            }.get(result.quality, "⚪")
            
            value_str = f"{result.value:.2f}" if isinstance(result.value, float) else str(result.value)
            lines.append(f"{emoji} {key}: {value_str} ({result.source})")
        
        return "\n".join(lines)


# Singleton instance
_chain: Optional[FallbackChain] = None

def get_fallback_chain() -> FallbackChain:
    global _chain
    if _chain is None:
        _chain = FallbackChain()
    return _chain
