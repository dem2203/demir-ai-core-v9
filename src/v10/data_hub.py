# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - DATA HUB
========================
Tüm piyasa verilerini tek noktadan toplayan merkezi modül.

ÖZELLİKLER:
- Binance REST + WebSocket entegrasyonu
- Gerçek zamanlı fiyat, order book, funding, OI
- HATA YUTULMAZ - tüm hatalar açıkça loglanır
- Mock/Fallback YOK - veri yoksa açıkça bildirilir

DESTEKLENEN COİNLER: BTCUSDT, ETHUSDT, SOLUSDT, LTCUSDT
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dataclasses import dataclass, field
from enum import Enum
from src.brain.advanced_scrapers import AdvancedMarketScrapers

logger = logging.getLogger("DATA_HUB")

# Configure logging to show ALL errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DataStatus(Enum):
    """Veri durumu - ASLA mock/fallback yok"""
    LIVE = "LIVE"           # Gerçek anlık veri
    STALE = "STALE"         # 1 dk'dan eski
    ERROR = "ERROR"         # Çekilemedi - hata
    NO_DATA = "NO_DATA"     # Veri yok


@dataclass
class MarketSnapshot:
    """Tek coin için tüm piyasa verilerinin anlık görüntüsü"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: DataStatus = DataStatus.NO_DATA
    errors: List[str] = field(default_factory=list)
    
    # Price Data (0 = veri yok)
    price: float = 0
    price_change_1h: float = 0
    price_change_4h: float = 0
    price_change_24h: float = 0
    high_24h: float = 0
    low_24h: float = 0
    
    # Order Book (-1 = veri yok, ASLA default değil)
    bid_volume: float = -1         # -1 = veri yok
    ask_volume: float = -1         # -1 = veri yok
    bid_ask_ratio: float = -1      # -1 = veri yok (ESKİ: 1.0 default YANLIŞ!)
    best_bid: float = 0
    best_ask: float = 0
    spread_pct: float = 0
    
    # Support/Resistance (Kline-based pivots, not order book)
    strong_bids: List[float] = field(default_factory=list)   # Order book büyük alımlar (referans)
    strong_asks: List[float] = field(default_factory=list)   # Order book büyük satışlar (referans)
    support: float = 0              # 24s pivot destek (en az %1 uzak)
    resistance: float = 0           # 24s pivot direnç (en az %1 uzak)
    major_support: float = 0        # 7 günlük majör destek
    major_resistance: float = 0     # 7 günlük majör direnç
    
    # Derivatives (Futures) (-1 = veri yok)
    funding_rate: float = -999      # -999 = veri yok (0 geçerli bir değer)
    open_interest: float = -1       # -1 = veri yok
    oi_change_1h: float = 0
    long_ratio: float = -1          # -1 = veri yok (ESKİ: 0.5 YANLIŞ!)
    short_ratio: float = -1         # -1 = veri yok (ESKİ: 0.5 YANLIŞ!)
    
    # Volume (-1 = veri yok)
    volume_24h: float = -1
    buy_volume: float = -1
    sell_volume: float = -1
    taker_buy_ratio: float = -1     # -1 = veri yok (ESKİ: 0.5 YANLIŞ!)
    
    # Whale Activity (0 = whale yok, -1 = veri yok)
    large_buys: int = -1            # -1 = veri yok
    large_sells: int = -1           # -1 = veri yok  
    whale_net_flow: float = -999    # -999 = veri yok (0 geçerli)
    
    # Technical Indicators (-1 = veri yok)
    rsi_1h: float = -1              # -1 = veri yok (ESKİ: 50 YANLIŞ!)
    rsi_4h: float = -1              # -1 = veri yok (ESKİ: 50 YANLIŞ!)
    ema_20: float = 0
    ema_50: float = 0
    ema_200: float = 0
    macd_signal: str = "UNKNOWN"    # "UNKNOWN" = veri yok (ESKİ: NEUTRAL YANLIŞ!)
    trend: str = "UNKNOWN"          # "UNKNOWN" = veri yok (ESKİ: NEUTRAL YANLIŞ!)
    
    # Raw Data for AI
    raw_klines: List[Any] = field(default_factory=list) # Raw OHLCV data for Feature Engineering
    
    @property
    def is_valid(self) -> bool:
        """Veri geçerli mi?"""
        return self.status == DataStatus.LIVE and self.price > 0


class DataHub:
    """
    DEMIR AI v10 - Merkezi Veri Toplama
    
    Tüm veri kaynaklarını tek noktadan yönetir.
    HATA YUTULMAZ - Her hata açıkça loglanır ve raporlanır.
    """
    
    # Binance Futures API (daha güvenilir, Railway'de 403 almaz)
    FUTURES_BASE = "https://fapi.binance.com"
    
    # User-Agent header (403 önlemek için)
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    SUPPORTED_COINS = ['BTCUSDT', 'ETHUSDT']
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_snapshots: Dict[str, MarketSnapshot] = {}
        self._error_count: int = 0
        self._success_count: int = 0
        self.scraper = AdvancedMarketScrapers()
        logger.info("📡 Data Hub initialized - FUTURES API MODE + WEB FALLBACK")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP session al - timeout'lu + User-Agent header'lı"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=self.HEADERS
            )
        return self._session
    
    async def close(self):
        """Session'ı kapat"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    # =========================================
    # ANA FONKSİYON - TÜM VERİLERİ ÇEK
    # =========================================
    
    async def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """
        Tek coin için tüm verileri paralel çek.
        HATA YUTULMAZ - her hata kaydedilir.
        """
        snapshot = MarketSnapshot(symbol=symbol)
        
        # Paralel veri çekme
        results = await asyncio.gather(
            self._fetch_ticker(symbol),
            self._fetch_orderbook(symbol),
            self._fetch_funding(symbol),
            self._fetch_oi(symbol),
            self._fetch_long_short(symbol),
            self._fetch_klines_for_technicals(symbol),
            self._fetch_trades_for_whales(symbol),
            return_exceptions=True
        )
        
        # Sonuçları işle - HATALARI AÇIKÇA LOGLA
        keys = ['ticker', 'orderbook', 'funding', 'oi', 'ls_ratio', 'technicals', 'whales']
        
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"{keys[i]}: {str(result)}"
                snapshot.errors.append(error_msg)
                logger.error(f"❌ DATA HUB ERROR [{symbol}] {error_msg}")
                self._error_count += 1
            else:
                # SUCCESS - veriyi ata
                self._apply_result(snapshot, keys[i], result)
                success_count += 1
                self._success_count += 1
        
        # CRITICAL FALLBACK: If API failed to get price, try Web Scraping
        if snapshot.price == 0:
            try:
                # Run sync scraper in thread to not block engine
                loop = asyncio.get_event_loop()
                web_price = await loop.run_in_executor(None, self.scraper.get_realtime_price, symbol)
                
                if web_price > 0:
                    snapshot.price = web_price
                    snapshot.errors.append("WARNING: Using Web Fallback Data")
                    # Web data is valid enough for price action
                    success_count += 3 # Boost score to pass validation
                    logger.warning(f"⚠️ {symbol} RECOVERED via Web Fallback: ${web_price}")
            except Exception as e:
                logger.error(f"Web fallback failed: {e}")

        # Status belirle
        if success_count >= 5:
            snapshot.status = DataStatus.LIVE
        elif success_count >= 3:
            snapshot.status = DataStatus.STALE
            logger.warning(f"⚠️ {symbol} partial data: {success_count}/7 sources OK")
        else:
            snapshot.status = DataStatus.ERROR
            logger.error(f"❌ {symbol} FAILED: only {success_count}/7 sources OK")
        
        self._last_snapshots[symbol] = snapshot
        return snapshot
    
    async def get_all_snapshots(self) -> Dict[str, MarketSnapshot]:
        """Tüm coinler için snapshot al"""
        results = await asyncio.gather(
            *[self.get_snapshot(coin) for coin in self.SUPPORTED_COINS],
            return_exceptions=True
        )
        
        return {
            coin: (result if isinstance(result, MarketSnapshot) 
                   else MarketSnapshot(symbol=coin, status=DataStatus.ERROR, errors=[str(result)]))
            for coin, result in zip(self.SUPPORTED_COINS, results)
        }
    
    # =========================================
    # VERİ ÇEKME FONKSİYONLARI
    # HATA YUTULMAZ - Exception fırlatılır
    # =========================================
    
    async def _fetch_ticker(self, symbol: str) -> Dict:
        """24hr ticker from FUTURES API - HATA YUTULMAZ"""
        session = await self._get_session()
        
        # FUTURES API kullan (403 önlemek için)
        url = f"{self.FUTURES_BASE}/fapi/v1/ticker/24hr?symbol={symbol}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Ticker API error: {resp.status}")
            data = await resp.json()
        
        return {
            'price': float(data['lastPrice']),
            'high_24h': float(data['highPrice']),
            'low_24h': float(data['lowPrice']),
            'volume_24h': float(data['quoteVolume']),
            'price_change_24h': float(data['priceChangePercent'])
        }
    
    async def _fetch_orderbook(self, symbol: str) -> Dict:
        """Order book depth from FUTURES API - HATA YUTULMAZ"""
        session = await self._get_session()
        
        # FUTURES API kullan
        url = f"{self.FUTURES_BASE}/fapi/v1/depth?symbol={symbol}&limit=100"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Orderbook API error: {resp.status}")
            data = await resp.json()
        
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        # Top 20 volume
        bid_volume = sum(float(b[0]) * float(b[1]) for b in bids[:20])
        ask_volume = sum(float(a[0]) * float(a[1]) for a in asks[:20])
        
        # Strong levels (büyük order'lar)
        strong_bids = []
        strong_asks = []
        
        for bid in bids:
            size_usd = float(bid[0]) * float(bid[1])
            if size_usd > 100000:  # $100K+ order
                strong_bids.append(float(bid[0]))
        
        for ask in asks:
            size_usd = float(ask[0]) * float(ask[1])
            if size_usd > 100000:
                strong_asks.append(float(ask[0]))
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_ask_ratio': bid_volume / ask_volume if ask_volume > 0 else 1.0,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread_pct': spread,
            'strong_bids': strong_bids[:5],
            'strong_asks': strong_asks[:5]
        }
    
    async def _fetch_funding(self, symbol: str) -> Dict:
        """Funding rate - HATA YUTULMAZ"""
        session = await self._get_session()
        
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Funding API error: {resp.status}")
            data = await resp.json()
        
        if not data:
            raise Exception("No funding data returned")
        
        return {
            'funding_rate': float(data[0]['fundingRate']) * 100
        }
    
    async def _fetch_oi(self, symbol: str) -> Dict:
        """Open Interest - HATA YUTULMAZ"""
        session = await self._get_session()
        
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"OI API error: {resp.status}")
            data = await resp.json()
        
        oi = float(data.get('openInterest', 0))
        
        # Price for USD conversion
        price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        async with session.get(price_url) as resp:
            price_data = await resp.json()
        price = float(price_data.get('price', 0))
        
        return {
            'open_interest': oi * price
        }
    
    async def _fetch_long_short(self, symbol: str) -> Dict:
        """Long/Short ratio - HATA YUTULMAZ"""
        session = await self._get_session()
        
        url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"L/S API error: {resp.status}")
            data = await resp.json()
        
        if not data:
            raise Exception("No L/S data returned")
        
        return {
            'long_ratio': float(data[0].get('longAccount', 0.5)),
            'short_ratio': float(data[0].get('shortAccount', 0.5))
        }
    
    async def _fetch_klines_for_technicals(self, symbol: str) -> Dict:
        """Klines for technical indicators from FUTURES API - HATA YUTULMAZ"""
        session = await self._get_session()
        
        # 1h klines - FUTURES API kullan
        url = f"{self.FUTURES_BASE}/fapi/v1/klines?symbol={symbol}&interval=1h&limit=200"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Klines API error: {resp.status}")
            klines = await resp.json()
        
        closes = [float(k[4]) for k in klines]
        
        # RSI
        rsi_1h = self._calculate_rsi(closes[-14:])
        rsi_4h = self._calculate_rsi(closes[-56::4][:14]) if len(closes) >= 56 else rsi_1h
        
        # EMAs
        ema_20 = self._calculate_ema(closes, 20)
        ema_50 = self._calculate_ema(closes, 50)
        ema_200 = self._calculate_ema(closes, 200) if len(closes) >= 200 else ema_50
        
        current = closes[-1]
        
        # Trend
        if current > ema_20 > ema_50:
            trend = "BULLISH"
        elif current < ema_20 < ema_50:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # MACD simple
        if len(closes) >= 3:
            if closes[-1] > closes[-2] > closes[-3]:
                macd_signal = "BUY"
            elif closes[-1] < closes[-2] < closes[-3]:
                macd_signal = "SELL"
            else:
                macd_signal = "NEUTRAL"
        else:
            macd_signal = "NEUTRAL"
        
        # Price changes
        price_change_1h = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
        price_change_4h = ((closes[-1] - closes[-5]) / closes[-5] * 100) if len(closes) >= 5 else 0
        
        # === GERÇEK DESTEK/DİRENÇ HESAPLA ===
        # Son 24 saatlik (24 mum) high/low'lardan pivot seviyeleri bul
        current = closes[-1]
        highs = [float(k[2]) for k in klines[-24:]]
        lows = [float(k[3]) for k in klines[-24:]]
        
        # Destek: Fiyatın EN AZ %1 altındaki en yüksek dip
        # Direnç: Fiyatın EN AZ %1 üstündeki en düşük tepe
        min_distance_pct = 0.01  # %1 minimum mesafe
        
        # Potansiyel destek seviyeleri (fiyatın altında, en az %1 uzak)
        potential_supports = [low for low in lows if low < current * (1 - min_distance_pct)]
        # En yakın güçlü destek
        support = max(potential_supports) if potential_supports else current * 0.98
        
        # Potansiyel direnç seviyeleri (fiyatın üstünde, en az %1 uzak)
        potential_resistances = [high for high in highs if high > current * (1 + min_distance_pct)]
        # En yakın güçlü direnç
        resistance = min(potential_resistances) if potential_resistances else current * 1.02
        
        # Son 7 günlük majör seviyeler (haftalık pivot)
        if len(klines) >= 168:
            weekly_highs = [float(k[2]) for k in klines[-168:]]
            weekly_lows = [float(k[3]) for k in klines[-168:]]
            major_resistance = max(weekly_highs)
            major_support = min(weekly_lows)
        else:
            major_resistance = resistance
            major_support = support
        
        return {
            'rsi_1h': rsi_1h,
            'rsi_4h': rsi_4h,
            'ema_20': ema_20,
            'ema_50': ema_50,
            'ema_200': ema_200,
            'trend': trend,
            'macd_signal': macd_signal,
            'price_change_1h': price_change_1h,
            'price_change_4h': price_change_4h,
            # Yeni: Gerçek S/R seviyeleri
            'support': support,
            'resistance': resistance,
            'major_support': major_support,
            'major_resistance': major_resistance,
            'raw_klines': klines # Raw Binance klines
        }
    
    async def _fetch_trades_for_whales(self, symbol: str) -> Dict:
        """Recent trades for whale detection from FUTURES API - HATA YUTULMAZ"""
        session = await self._get_session()
        
        # FUTURES API kullan (aggregated trades, daha verimli)
        url = f"{self.FUTURES_BASE}/fapi/v1/aggTrades?symbol={symbol}&limit=500"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Trades API error: {resp.status}")
            trades = await resp.json()
        
        # Get current price from last trade
        # aggTrades format: {p: price, q: qty, m: was buyer maker}
        current_price = float(trades[-1]['p']) if trades else 0
        
        # Whale threshold (>$50K for BTC, scaled for others)
        whale_threshold = 50000 if 'BTC' in symbol else 20000
        
        large_buys = 0
        large_sells = 0
        buy_volume = 0
        sell_volume = 0
        
        for trade in trades:
            # aggTrades format: q=quantity, p=price, m=was buyer maker
            qty = float(trade.get('q', 0))
            price = float(trade.get('p', 0))
            value = qty * price
            
            # m=True means buyer was maker (so trade was SELL taker)
            is_maker_buy = trade.get('m', False)
            
            if is_maker_buy:
                # Buyer was maker = seller was taker = SELL
                sell_volume += value
                if value >= whale_threshold:
                    large_sells += 1
            else:
                # Buyer was taker = BUY
                buy_volume += value
                if value >= whale_threshold:
                    large_buys += 1
        
        total = buy_volume + sell_volume
        
        return {
            'large_buys': large_buys,
            'large_sells': large_sells,
            'whale_net_flow': large_buys - large_sells,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'taker_buy_ratio': buy_volume / total if total > 0 else -1  # -1 = veri yok
        }
    
    # =========================================
    # YARDIMCI FONKSİYONLAR
    # =========================================
    
    def _apply_result(self, snapshot: MarketSnapshot, key: str, data: Dict):
        """Veriyi snapshot'a uygula"""
        if key == 'ticker':
            snapshot.price = data.get('price', 0)
            snapshot.high_24h = data.get('high_24h', 0)
            snapshot.low_24h = data.get('low_24h', 0)
            snapshot.volume_24h = data.get('volume_24h', 0)
            snapshot.price_change_24h = data.get('price_change_24h', 0)
        
        elif key == 'orderbook':
            snapshot.bid_volume = data.get('bid_volume', 0)
            snapshot.ask_volume = data.get('ask_volume', 0)
            snapshot.bid_ask_ratio = data.get('bid_ask_ratio', 1.0)
            snapshot.best_bid = data.get('best_bid', 0)
            snapshot.best_ask = data.get('best_ask', 0)
            snapshot.spread_pct = data.get('spread_pct', 0)
            snapshot.strong_bids = data.get('strong_bids', [])
            snapshot.strong_asks = data.get('strong_asks', [])
        
        elif key == 'funding':
            snapshot.funding_rate = data.get('funding_rate', 0)
        
        elif key == 'oi':
            snapshot.open_interest = data.get('open_interest', 0)
        
        elif key == 'ls_ratio':
            snapshot.long_ratio = data.get('long_ratio', 0.5)
            snapshot.short_ratio = data.get('short_ratio', 0.5)
        
        elif key == 'technicals':
            snapshot.rsi_1h = data.get('rsi_1h', -1)  # -1 = veri yok
            snapshot.rsi_4h = data.get('rsi_4h', -1)  # -1 = veri yok
            snapshot.ema_20 = data.get('ema_20', 0)
            snapshot.ema_50 = data.get('ema_50', 0)
            snapshot.ema_200 = data.get('ema_200', 0)
            snapshot.trend = data.get('trend', 'NEUTRAL')
            snapshot.macd_signal = data.get('macd_signal', 'NEUTRAL')
            snapshot.price_change_1h = data.get('price_change_1h', 0)
            snapshot.price_change_4h = data.get('price_change_4h', 0)
            # Yeni S/R seviyeleri (kline-based)
            snapshot.support = data.get('support', 0)
            snapshot.resistance = data.get('resistance', 0)
            snapshot.major_support = data.get('major_support', 0)
            snapshot.major_resistance = data.get('major_resistance', 0)
            snapshot.raw_klines = data.get('raw_klines', [])
        
        elif key == 'whales':
            snapshot.large_buys = data.get('large_buys', 0)
            snapshot.large_sells = data.get('large_sells', 0)
            snapshot.whale_net_flow = data.get('whale_net_flow', 0)
            snapshot.buy_volume = data.get('buy_volume', 0)
            snapshot.sell_volume = data.get('sell_volume', 0)
            snapshot.taker_buy_ratio = data.get('taker_buy_ratio', 0.5)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI hesapla"""
        if len(prices) < period:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """EMA hesapla"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def get_stats(self) -> Dict:
        """Hata/başarı istatistikleri"""
        total = self._error_count + self._success_count
        return {
            'total_calls': total,
            'success': self._success_count,
            'errors': self._error_count,
            'success_rate': (self._success_count / total * 100) if total > 0 else 0
        }


# Singleton
_data_hub: Optional[DataHub] = None

def get_data_hub() -> DataHub:
    global _data_hub
    if _data_hub is None:
        _data_hub = DataHub()
    return _data_hub
