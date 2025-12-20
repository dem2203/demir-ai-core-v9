# -*- coding: utf-8 -*-
"""
DEMIR AI - MULTI-EXCHANGE ORDER BOOK AGGREGATOR
================================================
Binance, Bybit, OKX, Kraken order book derinliği
API Key gereksiz - Public endpoints kullanılır

4 Coin: BTCUSDT, ETHUSDT, LTCUSDT, SOLUSDT
"""
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("MULTI_EXCHANGE_ORDERBOOK")


@dataclass
class ExchangeOrderBook:
    """Tek bir borsanın order book verisi"""
    exchange: str
    symbol: str
    bid_volume: float = 0
    ask_volume: float = 0
    best_bid: float = 0
    best_ask: float = 0
    spread_pct: float = 0
    imbalance: float = 1.0  # bid/ask ratio
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedOrderBook:
    """Tüm borsaların birleşik order book analizi"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Exchange-specific data
    exchanges: List[ExchangeOrderBook] = field(default_factory=list)
    
    # Aggregated metrics
    total_bid_volume: float = 0
    total_ask_volume: float = 0
    overall_imbalance: float = 1.0  # >1 = bid heavy (bullish), <1 = ask heavy (bearish)
    
    # Cross-exchange analysis
    price_divergence: float = 0  # Max price diff across exchanges (%)
    dominant_exchange: str = ""  # Which exchange has most volume
    
    # Signals
    bid_wall_detected: bool = False
    ask_wall_detected: bool = False
    wall_price: float = 0
    signal_strength: str = "NEUTRAL"  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"


class MultiExchangeOrderBook:
    """
    Multi-Exchange Order Book Aggregator
    
    Public API'ler kullanarak 4 borsadan order book çeker.
    API Key gereksiz!
    """
    
    # Symbol mapping for different exchanges
    SYMBOL_MAP = {
        'BTCUSDT': {
            'binance': 'BTCUSDT',
            'bybit': 'BTCUSDT',
            'okx': 'BTC-USDT',
            'kraken': 'XBTUSDT'
        },
        'ETHUSDT': {
            'binance': 'ETHUSDT',
            'bybit': 'ETHUSDT',
            'okx': 'ETH-USDT',
            'kraken': 'ETHUSDT'
        },
        'LTCUSDT': {
            'binance': 'LTCUSDT',
            'bybit': 'LTCUSDT',
            'okx': 'LTC-USDT',
            'kraken': 'LTCUSDT'
        },
        'SOLUSDT': {
            'binance': 'SOLUSDT',
            'bybit': 'SOLUSDT',
            'okx': 'SOL-USDT',
            'kraken': 'SOLUSDT'
        }
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, AggregatedOrderBook] = {}
        self._cache_ttl = 10  # seconds
        logger.info("📊 Multi-Exchange OrderBook initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    # =========================================
    # EXCHANGE API CALLS (No API Key Required)
    # =========================================
    
    async def _fetch_binance(self, symbol: str) -> ExchangeOrderBook:
        """Binance order book - Public API"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=50"
            
            async with session.get(url) as resp:
                data = await resp.json()
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            bid_volume = sum(float(b[1]) * float(b[0]) for b in bids)  # USD value
            ask_volume = sum(float(a[1]) * float(a[0]) for a in asks)
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
            imbalance = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            return ExchangeOrderBook(
                exchange="binance",
                symbol=symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread_pct,
                imbalance=imbalance
            )
        except Exception as e:
            logger.debug(f"Binance orderbook error: {e}")
            return ExchangeOrderBook(exchange="binance", symbol=symbol)
    
    async def _fetch_bybit(self, symbol: str) -> ExchangeOrderBook:
        """Bybit order book - Public API"""
        try:
            session = await self._get_session()
            url = f"https://api.bybit.com/v5/market/orderbook?category=spot&symbol={symbol}&limit=50"
            
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('result', {})
            bids = result.get('b', [])  # [price, qty]
            asks = result.get('a', [])
            
            bid_volume = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_volume = sum(float(a[1]) * float(a[0]) for a in asks)
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
            imbalance = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            return ExchangeOrderBook(
                exchange="bybit",
                symbol=symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread_pct,
                imbalance=imbalance
            )
        except Exception as e:
            logger.debug(f"Bybit orderbook error: {e}")
            return ExchangeOrderBook(exchange="bybit", symbol=symbol)
    
    async def _fetch_okx(self, symbol: str) -> ExchangeOrderBook:
        """OKX order book - Public API"""
        try:
            session = await self._get_session()
            # OKX uses different symbol format: BTC-USDT
            okx_symbol = self.SYMBOL_MAP.get(symbol, {}).get('okx', symbol.replace('USDT', '-USDT'))
            url = f"https://www.okx.com/api/v5/market/books?instId={okx_symbol}&sz=50"
            
            async with session.get(url) as resp:
                data = await resp.json()
            
            books = data.get('data', [{}])[0]
            bids = books.get('bids', [])  # [price, qty, ?, ?]
            asks = books.get('asks', [])
            
            bid_volume = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_volume = sum(float(a[1]) * float(a[0]) for a in asks)
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
            imbalance = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            return ExchangeOrderBook(
                exchange="okx",
                symbol=symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread_pct,
                imbalance=imbalance
            )
        except Exception as e:
            logger.debug(f"OKX orderbook error: {e}")
            return ExchangeOrderBook(exchange="okx", symbol=symbol)
    
    async def _fetch_kraken(self, symbol: str) -> ExchangeOrderBook:
        """Kraken order book - Public API"""
        try:
            session = await self._get_session()
            # Kraken uses XBT for BTC
            kraken_symbol = self.SYMBOL_MAP.get(symbol, {}).get('kraken', symbol)
            url = f"https://api.kraken.com/0/public/Depth?pair={kraken_symbol}&count=50"
            
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('result', {})
            # Kraken returns data under a dynamic key
            book_data = list(result.values())[0] if result else {}
            
            bids = book_data.get('bids', [])  # [price, volume, timestamp]
            asks = book_data.get('asks', [])
            
            bid_volume = sum(float(b[1]) * float(b[0]) for b in bids)
            ask_volume = sum(float(a[1]) * float(a[0]) for a in asks)
            
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
            imbalance = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            return ExchangeOrderBook(
                exchange="kraken",
                symbol=symbol,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread_pct,
                imbalance=imbalance
            )
        except Exception as e:
            logger.debug(f"Kraken orderbook error: {e}")
            return ExchangeOrderBook(exchange="kraken", symbol=symbol)
    
    # =========================================
    # AGGREGATION
    # =========================================
    
    async def get_aggregated_orderbook(self, symbol: str = "BTCUSDT") -> AggregatedOrderBook:
        """
        4 borsadan order book topla ve analiz et.
        """
        # Check cache
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')[:12]}"
        if symbol in self._cache:
            cached = self._cache[symbol]
            age = (datetime.now() - cached.timestamp).total_seconds()
            if age < self._cache_ttl:
                return cached
        
        # Fetch from all exchanges in parallel
        results = await asyncio.gather(
            self._fetch_binance(symbol),
            self._fetch_bybit(symbol),
            self._fetch_okx(symbol),
            self._fetch_kraken(symbol),
            return_exceptions=True
        )
        
        exchanges = []
        for r in results:
            if isinstance(r, ExchangeOrderBook) and r.bid_volume > 0:
                exchanges.append(r)
        
        if not exchanges:
            return AggregatedOrderBook(symbol=symbol)
        
        # Calculate aggregated metrics
        total_bid = sum(e.bid_volume for e in exchanges)
        total_ask = sum(e.ask_volume for e in exchanges)
        overall_imbalance = total_bid / total_ask if total_ask > 0 else 1.0
        
        # Find price divergence
        prices = [e.best_bid for e in exchanges if e.best_bid > 0]
        if prices:
            max_price = max(prices)
            min_price = min(prices)
            price_divergence = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0
        else:
            price_divergence = 0
        
        # Find dominant exchange (most volume)
        dominant = max(exchanges, key=lambda e: e.bid_volume + e.ask_volume)
        
        # Detect walls
        bid_wall_detected = overall_imbalance > 2.5
        ask_wall_detected = overall_imbalance < 0.4
        
        # Determine signal strength
        if overall_imbalance > 3.0:
            signal_strength = "STRONG_BUY"
        elif overall_imbalance > 2.0:
            signal_strength = "BUY"
        elif overall_imbalance < 0.33:
            signal_strength = "STRONG_SELL"
        elif overall_imbalance < 0.5:
            signal_strength = "SELL"
        else:
            signal_strength = "NEUTRAL"
        
        aggregated = AggregatedOrderBook(
            symbol=symbol,
            exchanges=exchanges,
            total_bid_volume=total_bid,
            total_ask_volume=total_ask,
            overall_imbalance=overall_imbalance,
            price_divergence=price_divergence,
            dominant_exchange=dominant.exchange,
            bid_wall_detected=bid_wall_detected,
            ask_wall_detected=ask_wall_detected,
            wall_price=dominant.best_bid if bid_wall_detected else dominant.best_ask if ask_wall_detected else 0,
            signal_strength=signal_strength
        )
        
        self._cache[symbol] = aggregated
        return aggregated
    
    async def get_all_coins(self) -> Dict[str, AggregatedOrderBook]:
        """4 coin için tüm order book verilerini al"""
        results = await asyncio.gather(
            self.get_aggregated_orderbook("BTCUSDT"),
            self.get_aggregated_orderbook("ETHUSDT"),
            self.get_aggregated_orderbook("LTCUSDT"),
            self.get_aggregated_orderbook("SOLUSDT"),
            return_exceptions=True
        )
        
        coins = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
        return {
            coin: r for coin, r in zip(coins, results)
            if isinstance(r, AggregatedOrderBook)
        }
    
    def format_for_telegram(self, data: AggregatedOrderBook) -> str:
        """Order book verisini Telegram mesajı olarak formatla"""
        exchanges_text = ""
        for ex in data.exchanges:
            imb_emoji = "🟢" if ex.imbalance > 1.2 else "🔴" if ex.imbalance < 0.8 else "⚪"
            exchanges_text += f"  • {ex.exchange.upper()}: {imb_emoji} {ex.imbalance:.2f}x (${ex.bid_volume/1e6:.1f}M/${ ex.ask_volume/1e6:.1f}M)\n"
        
        signal_emoji = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "NEUTRAL": "⚪",
            "SELL": "🔴",
            "STRONG_SELL": "💀"
        }.get(data.signal_strength, "⚪")
        
        wall_text = ""
        if data.bid_wall_detected:
            wall_text = f"\n🧱 *BID WALL*: ${data.wall_price:,.0f}"
        elif data.ask_wall_detected:
            wall_text = f"\n🧱 *ASK WALL*: ${data.wall_price:,.0f}"
        
        return f"""📊 *MULTI-EXCHANGE ORDER BOOK - {data.symbol}*
━━━━━━━━━━━━━━━━━━
{signal_emoji} *Sinyal: {data.signal_strength}*
📈 Genel Imbalance: *{data.overall_imbalance:.2f}x*
💰 Toplam Bid: ${data.total_bid_volume/1e6:.1f}M
💰 Toplam Ask: ${data.total_ask_volume/1e6:.1f}M
🔄 Fiyat Farkı: %{data.price_divergence:.2f}
👑 Dominant: {data.dominant_exchange.upper()}
{wall_text}
━━━ BORSALAR ━━━
{exchanges_text}━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""


# Singleton instance
_instance: Optional[MultiExchangeOrderBook] = None

def get_multi_exchange_orderbook() -> MultiExchangeOrderBook:
    global _instance
    if _instance is None:
        _instance = MultiExchangeOrderBook()
    return _instance
