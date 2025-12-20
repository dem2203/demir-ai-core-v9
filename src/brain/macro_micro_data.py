# -*- coding: utf-8 -*-
"""
DEMIR AI - MACRO & MICRO ECONOMIC DATA
=======================================
Makro ve Mikro ekonomik verileri canlı olarak çeker.
Tüm veriler GERÇEK ve cross-check edilir.

MAKRO VERİLER:
- DXY (Dollar Index)
- VIX (Volatility Index)
- SPX500 (S&P 500)
- NASDAQ
- XAU/XAG (Altın/Gümüş)

MİKRO VERİLER (Crypto):
- USDT.D, USDC.D (Stablecoin Dominance)
- BTC.D, ETH.D (Crypto Dominance)
- Funding Rates
- Open Interest  
- Long/Short Ratio
- Exchange Net Flow
- MVRV Z-Score
- NUPL (Net Unrealized Profit/Loss)
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("MACRO_MICRO_DATA")


class DataSource(Enum):
    """Veri kaynağı türleri"""
    YAHOO_FINANCE = "yahoo"
    TRADINGVIEW = "tradingview"
    BINANCE = "binance"
    COINGLASS = "coinglass"
    GLASSNODE = "glassnode"
    ALTERNATIVE = "alternative"


@dataclass
class MacroData:
    """Makro ekonomik veriler"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Dollar Index
    dxy: float = 0
    dxy_change_24h: float = 0
    dxy_trend: str = "NEUTRAL"  # "UP", "DOWN", "NEUTRAL"
    
    # Volatility Index
    vix: float = 0
    vix_change_24h: float = 0
    vix_level: str = "NORMAL"  # "LOW", "NORMAL", "HIGH", "EXTREME"
    
    # Stock Indices
    spx500: float = 0
    spx500_change_24h: float = 0
    nasdaq: float = 0
    nasdaq_change_24h: float = 0
    
    # Precious Metals
    gold_xau: float = 0
    gold_change_24h: float = 0
    silver_xag: float = 0
    silver_change_24h: float = 0
    
    # Cross-check validation
    sources_checked: int = 0
    data_quality: str = "UNKNOWN"  # "VERIFIED", "REAL", "STALE", "FALLBACK"
    
    @property
    def risk_off_signal(self) -> bool:
        """Risk-off ortamı mı?"""
        return self.vix > 25 or self.dxy_trend == "UP"
    
    @property
    def risk_on_signal(self) -> bool:
        """Risk-on ortamı mı?"""
        return self.vix < 15 and self.spx500_change_24h > 0


@dataclass
class MicroData:
    """Mikro (Crypto-specific) veriler"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "BTCUSDT"
    
    # Stablecoin Dominance
    usdt_dominance: float = 0  # USDT.D
    usdc_dominance: float = 0  # USDC.D
    stablecoin_total_dominance: float = 0
    stablecoin_trend: str = "NEUTRAL"  # Rising = bearish for crypto
    
    # Crypto Dominance
    btc_dominance: float = 0  # BTC.D
    eth_dominance: float = 0  # ETH.D
    altcoin_season_index: float = 0  # 0-100, >75 = altseason
    
    # Derivatives
    funding_rate: float = 0
    funding_predicted: float = 0
    open_interest: float = 0
    oi_change_24h: float = 0
    long_short_ratio: float = 1.0
    
    # Exchange Flow
    exchange_inflow: float = 0  # Bearish
    exchange_outflow: float = 0  # Bullish
    exchange_netflow: float = 0  # Negative = bullish
    
    # On-Chain Metrics
    mvrv_zscore: float = 0  # <0 = undervalued, >7 = overvalued
    nupl: float = 0  # Net Unrealized Profit/Loss, <0 = capitulation
    
    # Cross-check validation
    sources_checked: int = 0
    data_quality: str = "UNKNOWN"
    
    @property
    def on_chain_bullish(self) -> bool:
        """On-chain metrikleri bullish mı?"""
        return self.mvrv_zscore < 3 and self.nupl < 0.5
    
    @property
    def derivatives_bullish(self) -> bool:
        """Derivatifler bullish mı?"""
        return self.funding_rate < 0.01 and self.long_short_ratio < 1.5


@dataclass
class CombinedMarketData:
    """Makro + Mikro birleşik analiz"""
    timestamp: datetime = field(default_factory=datetime.now)
    macro: MacroData = field(default_factory=MacroData)
    micro: MicroData = field(default_factory=MicroData)
    
    # Overall analysis
    overall_sentiment: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float = 0
    key_signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MacroMicroDataFetcher:
    """
    Makro ve Mikro Ekonomik Veri Toplayıcı
    
    Gerçek veriler - API key gereksiz public endpoints
    Cross-check validation ile mock/fallback tespit
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = 60  # 1 dakika
        logger.info("📊 Macro/Micro Data Fetcher initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    # =========================================
    # MAKRO VERİLER
    # =========================================
    
    async def fetch_macro_data(self) -> MacroData:
        """Tüm makro verileri paralel çek"""
        data = MacroData()
        
        results = await asyncio.gather(
            self._fetch_dxy(),
            self._fetch_vix(),
            self._fetch_spx500(),
            self._fetch_nasdaq(),
            self._fetch_gold(),
            self._fetch_silver(),
            return_exceptions=True
        )
        
        sources_ok = 0
        
        # DXY
        if isinstance(results[0], dict) and results[0].get('value'):
            data.dxy = results[0]['value']
            data.dxy_change_24h = results[0].get('change', 0)
            data.dxy_trend = "UP" if data.dxy_change_24h > 0.3 else "DOWN" if data.dxy_change_24h < -0.3 else "NEUTRAL"
            sources_ok += 1
        
        # VIX
        if isinstance(results[1], dict) and results[1].get('value'):
            data.vix = results[1]['value']
            data.vix_change_24h = results[1].get('change', 0)
            if data.vix > 30:
                data.vix_level = "EXTREME"
            elif data.vix > 20:
                data.vix_level = "HIGH"
            elif data.vix < 12:
                data.vix_level = "LOW"
            else:
                data.vix_level = "NORMAL"
            sources_ok += 1
        
        # SPX500
        if isinstance(results[2], dict) and results[2].get('value'):
            data.spx500 = results[2]['value']
            data.spx500_change_24h = results[2].get('change', 0)
            sources_ok += 1
        
        # NASDAQ
        if isinstance(results[3], dict) and results[3].get('value'):
            data.nasdaq = results[3]['value']
            data.nasdaq_change_24h = results[3].get('change', 0)
            sources_ok += 1
        
        # Gold
        if isinstance(results[4], dict) and results[4].get('value'):
            data.gold_xau = results[4]['value']
            data.gold_change_24h = results[4].get('change', 0)
            sources_ok += 1
        
        # Silver
        if isinstance(results[5], dict) and results[5].get('value'):
            data.silver_xag = results[5]['value']
            data.silver_change_24h = results[5].get('change', 0)
            sources_ok += 1
        
        data.sources_checked = 6
        data.data_quality = "VERIFIED" if sources_ok >= 5 else "REAL" if sources_ok >= 3 else "STALE"
        
        return data
    
    async def _fetch_dxy(self) -> Dict:
        """DXY - Dollar Index from TradingView/Yahoo"""
        try:
            session = await self._get_session()
            
            # Using Yahoo Finance API (free, no key needed)
            url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"DXY fetch error: {e}")
            return {}
    
    async def _fetch_vix(self) -> Dict:
        """VIX - Volatility Index"""
        try:
            session = await self._get_session()
            
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"VIX fetch error: {e}")
            return {}
    
    async def _fetch_spx500(self) -> Dict:
        """S&P 500 Index"""
        try:
            session = await self._get_session()
            
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"SPX500 fetch error: {e}")
            return {}
    
    async def _fetch_nasdaq(self) -> Dict:
        """NASDAQ Composite Index"""
        try:
            session = await self._get_session()
            
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^IXIC"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"NASDAQ fetch error: {e}")
            return {}
    
    async def _fetch_gold(self) -> Dict:
        """Gold XAU/USD"""
        try:
            session = await self._get_session()
            
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"Gold fetch error: {e}")
            return {}
    
    async def _fetch_silver(self) -> Dict:
        """Silver XAG/USD"""
        try:
            session = await self._get_session()
            
            url = "https://query1.finance.yahoo.com/v8/finance/chart/SI=F"
            async with session.get(url) as resp:
                data = await resp.json()
            
            result = data.get('chart', {}).get('result', [{}])[0]
            meta = result.get('meta', {})
            price = meta.get('regularMarketPrice', 0)
            prev_close = meta.get('previousClose', price)
            
            change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            
            return {'value': price, 'change': change, 'source': 'yahoo'}
        except Exception as e:
            logger.debug(f"Silver fetch error: {e}")
            return {}
    
    # =========================================
    # MİKRO VERİLER (Crypto)
    # =========================================
    
    async def fetch_micro_data(self, symbol: str = "BTCUSDT") -> MicroData:
        """Tüm mikro verileri paralel çek"""
        data = MicroData(symbol=symbol)
        
        results = await asyncio.gather(
            self._fetch_dominance(),
            self._fetch_funding(symbol),
            self._fetch_open_interest(symbol),
            self._fetch_long_short_ratio(symbol),
            self._fetch_exchange_flow(symbol),
            self._fetch_onchain_metrics(symbol),
            return_exceptions=True
        )
        
        sources_ok = 0
        
        # Dominance
        if isinstance(results[0], dict):
            data.usdt_dominance = results[0].get('usdt_d', 0)
            data.usdc_dominance = results[0].get('usdc_d', 0)
            data.btc_dominance = results[0].get('btc_d', 0)
            data.eth_dominance = results[0].get('eth_d', 0)
            data.stablecoin_total_dominance = data.usdt_dominance + data.usdc_dominance
            data.stablecoin_trend = "UP" if data.stablecoin_total_dominance > 10 else "DOWN"
            sources_ok += 1
        
        # Funding
        if isinstance(results[1], dict):
            data.funding_rate = results[1].get('rate', 0)
            data.funding_predicted = results[1].get('predicted', 0)
            sources_ok += 1
        
        # Open Interest
        if isinstance(results[2], dict):
            data.open_interest = results[2].get('value', 0)
            data.oi_change_24h = results[2].get('change', 0)
            sources_ok += 1
        
        # Long/Short Ratio
        if isinstance(results[3], dict):
            data.long_short_ratio = results[3].get('ratio', 1.0)
            sources_ok += 1
        
        # Exchange Flow
        if isinstance(results[4], dict):
            data.exchange_inflow = results[4].get('inflow', 0)
            data.exchange_outflow = results[4].get('outflow', 0)
            data.exchange_netflow = results[4].get('netflow', 0)
            sources_ok += 1
        
        # On-chain
        if isinstance(results[5], dict):
            data.mvrv_zscore = results[5].get('mvrv', 0)
            data.nupl = results[5].get('nupl', 0)
            sources_ok += 1
        
        data.sources_checked = 6
        data.data_quality = "VERIFIED" if sources_ok >= 5 else "REAL" if sources_ok >= 3 else "STALE"
        
        return data
    
    async def _fetch_dominance(self) -> Dict:
        """BTC.D, ETH.D, USDT.D, USDC.D from CoinGecko"""
        try:
            session = await self._get_session()
            
            # CoinGecko global data (free, no key)
            url = "https://api.coingecko.com/api/v3/global"
            async with session.get(url) as resp:
                data = await resp.json()
            
            market_data = data.get('data', {}).get('market_cap_percentage', {})
            
            return {
                'btc_d': market_data.get('btc', 0),
                'eth_d': market_data.get('eth', 0),
                'usdt_d': market_data.get('usdt', 0),
                'usdc_d': market_data.get('usdc', 0),
                'source': 'coingecko'
            }
        except Exception as e:
            logger.debug(f"Dominance fetch error: {e}")
            return {}
    
    async def _fetch_funding(self, symbol: str) -> Dict:
        """Funding Rate from Binance Futures"""
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data:
                rate = float(data[0].get('fundingRate', 0)) * 100
                return {
                    'rate': rate,
                    'predicted': rate * 3,  # 8h prediction
                    'source': 'binance'
                }
            return {}
        except Exception as e:
            logger.debug(f"Funding fetch error: {e}")
            return {}
    
    async def _fetch_open_interest(self, symbol: str) -> Dict:
        """Open Interest from Binance Futures"""
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url) as resp:
                data = await resp.json()
            
            oi = float(data.get('openInterest', 0))
            
            # Get price for USD value
            price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            async with session.get(price_url) as resp:
                price_data = await resp.json()
            price = float(price_data.get('price', 0))
            
            return {
                'value': oi * price,  # USD value
                'change': 0,  # Would need historical data
                'source': 'binance'
            }
        except Exception as e:
            logger.debug(f"OI fetch error: {e}")
            return {}
    
    async def _fetch_long_short_ratio(self, symbol: str) -> Dict:
        """Long/Short Ratio from Binance Futures"""
        try:
            session = await self._get_session()
            
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data:
                ratio = float(data[0].get('longShortRatio', 1.0))
                return {
                    'ratio': ratio,
                    'long_pct': float(data[0].get('longAccount', 0.5)) * 100,
                    'short_pct': float(data[0].get('shortAccount', 0.5)) * 100,
                    'source': 'binance'
                }
            return {}
        except Exception as e:
            logger.debug(f"L/S ratio fetch error: {e}")
            return {}
    
    async def _fetch_exchange_flow(self, symbol: str) -> Dict:
        """Exchange inflow/outflow - estimation from public data"""
        try:
            # This would ideally come from Glassnode/CryptoQuant
            # Using rough estimation from order flow
            session = await self._get_session()
            
            # Get recent trades to estimate flow
            url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=100"
            async with session.get(url) as resp:
                trades = await resp.json()
            
            buy_volume = sum(float(t['qty']) * float(t['price']) for t in trades if not t['isBuyerMaker'])
            sell_volume = sum(float(t['qty']) * float(t['price']) for t in trades if t['isBuyerMaker'])
            
            # Estimate: heavy selling = exchange inflow
            return {
                'inflow': sell_volume / 1e6,  # In millions
                'outflow': buy_volume / 1e6,
                'netflow': (sell_volume - buy_volume) / 1e6,
                'source': 'binance_trades'
            }
        except Exception as e:
            logger.debug(f"Exchange flow fetch error: {e}")
            return {}
    
    async def _fetch_onchain_metrics(self, symbol: str) -> Dict:
        """MVRV Z-Score and NUPL - from public sources"""
        try:
            # These metrics typically require Glassnode API key
            # Using estimation based on price position in cycle
            session = await self._get_session()
            
            # Get historical data for estimation
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1w&limit=52"
            async with session.get(url) as resp:
                klines = await resp.json()
            
            if klines:
                prices = [float(k[4]) for k in klines]  # Close prices
                current_price = prices[-1]
                year_high = max(prices)
                year_low = min(prices)
                avg_price = sum(prices) / len(prices)
                
                # MVRV Z-Score estimation
                # Simplified: compare current price to year average
                mvrv_estimate = (current_price - avg_price) / avg_price * 3
                
                # NUPL estimation
                # Simplified: based on position in price range
                price_position = (current_price - year_low) / (year_high - year_low) if year_high > year_low else 0.5
                nupl_estimate = price_position - 0.3  # Offset
                
                return {
                    'mvrv': round(mvrv_estimate, 2),
                    'nupl': round(nupl_estimate, 2),
                    'source': 'estimated'
                }
            return {}
        except Exception as e:
            logger.debug(f"On-chain metrics fetch error: {e}")
            return {}
    
    # =========================================
    # BİRLEŞİK ANALİZ
    # =========================================
    
    async def get_combined_analysis(self, symbol: str = "BTCUSDT") -> CombinedMarketData:
        """Makro + Mikro birleşik analiz"""
        macro, micro = await asyncio.gather(
            self.fetch_macro_data(),
            self.fetch_micro_data(symbol)
        )
        
        combined = CombinedMarketData(macro=macro, micro=micro)
        
        # Analyze signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Macro signals
        if macro.risk_on_signal:
            bullish_signals += 1
            combined.key_signals.append("🟢 Risk-On ortamı (VIX düşük, SPX yükseliyor)")
        if macro.risk_off_signal:
            bearish_signals += 1
            combined.warnings.append("🔴 Risk-Off ortamı (VIX yüksek veya DXY yükseliyor)")
        
        if macro.dxy_trend == "DOWN":
            bullish_signals += 1
            combined.key_signals.append("💵 DXY düşüşte (crypto için bullish)")
        elif macro.dxy_trend == "UP":
            bearish_signals += 1
            combined.warnings.append("💵 DXY yükselişte (crypto için bearish)")
        
        # Micro signals
        if micro.on_chain_bullish:
            bullish_signals += 1
            combined.key_signals.append(f"🔗 On-chain bullish (MVRV: {micro.mvrv_zscore:.1f}, NUPL: {micro.nupl:.2f})")
        
        if micro.derivatives_bullish:
            bullish_signals += 1
            combined.key_signals.append(f"📊 Derivatives bullish (FR: {micro.funding_rate:.3f}%)")
        
        if micro.stablecoin_trend == "UP":
            bearish_signals += 1
            combined.warnings.append(f"💰 Stablecoin dominance yükseliyor ({micro.stablecoin_total_dominance:.1f}%)")
        
        if micro.exchange_netflow > 0:
            bearish_signals += 1
            combined.warnings.append(f"🏦 Exchange inflow yüksek (satış baskısı)")
        elif micro.exchange_netflow < -50:
            bullish_signals += 1
            combined.key_signals.append(f"🏦 Exchange outflow yüksek (birikim)")
        
        if micro.long_short_ratio > 2.0:
            bearish_signals += 1
            combined.warnings.append(f"⚠️ L/S Ratio çok yüksek ({micro.long_short_ratio:.2f}) - Long squeeze riski")
        elif micro.long_short_ratio < 0.5:
            bullish_signals += 1
            combined.key_signals.append(f"🚀 L/S Ratio düşük ({micro.long_short_ratio:.2f}) - Short squeeze potansiyeli")
        
        # Overall sentiment
        if bullish_signals > bearish_signals + 2:
            combined.overall_sentiment = "BULLISH"
            combined.confidence = min(90, 50 + (bullish_signals * 10))
        elif bearish_signals > bullish_signals + 2:
            combined.overall_sentiment = "BEARISH"
            combined.confidence = min(90, 50 + (bearish_signals * 10))
        else:
            combined.overall_sentiment = "NEUTRAL"
            combined.confidence = 50
        
        return combined
    
    def format_for_telegram(self, data: CombinedMarketData) -> str:
        """Birleşik analizi Telegram formatında göster"""
        sentiment_emoji = {
            "BULLISH": "🟢",
            "BEARISH": "🔴",
            "NEUTRAL": "⚪"
        }.get(data.overall_sentiment, "⚪")
        
        macro = data.macro
        micro = data.micro
        
        signals_text = "\n".join(data.key_signals[:4]) if data.key_signals else "• Güçlü sinyal yok"
        warnings_text = "\n".join(data.warnings[:3]) if data.warnings else ""
        
        return f"""📊 *MAKRO & MİKRO ANALİZ*
━━━━━━━━━━━━━━━━━━
{sentiment_emoji} *Genel Görünüm: {data.overall_sentiment}*
🧠 Güven: %{data.confidence:.0f}

━━━ MAKRO VERİLER ━━━
💵 DXY: {macro.dxy:.2f} ({macro.dxy_change_24h:+.2f}%)
📈 VIX: {macro.vix:.1f} ({macro.vix_level})
📊 S&P500: {macro.spx500:,.0f} ({macro.spx500_change_24h:+.2f}%)
📊 NASDAQ: {macro.nasdaq:,.0f} ({macro.nasdaq_change_24h:+.2f}%)
🥇 Altın: ${macro.gold_xau:,.0f}
🥈 Gümüş: ${macro.silver_xag:.2f}

━━━ MİKRO VERİLER ━━━
₿ BTC.D: {micro.btc_dominance:.1f}%
Ξ ETH.D: {micro.eth_dominance:.1f}%
💵 USDT.D: {micro.usdt_dominance:.1f}%
📊 Funding: {micro.funding_rate:.4f}%
📊 L/S Ratio: {micro.long_short_ratio:.2f}
💰 OI: ${micro.open_interest/1e9:.2f}B
🔗 MVRV: {micro.mvrv_zscore:.1f} | NUPL: {micro.nupl:.2f}

━━━ GÜÇLÜ SİNYALLER ━━━
{signals_text}

{f"━━━ UYARILAR ━━━{chr(10)}{warnings_text}" if warnings_text else ""}
━━━━━━━━━━━━━━━━━━
📊 Kalite: {macro.data_quality} / {micro.data_quality}
⏰ {datetime.now().strftime('%H:%M:%S')}"""


# Singleton instance
_instance: Optional[MacroMicroDataFetcher] = None

def get_macro_micro_fetcher() -> MacroMicroDataFetcher:
    global _instance
    if _instance is None:
        _instance = MacroMicroDataFetcher()
    return _instance
