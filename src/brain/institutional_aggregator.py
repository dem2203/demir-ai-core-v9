# -*- coding: utf-8 -*-
"""
DEMIR AI - INSTITUTIONAL GRADE DATA AGGREGATOR
===============================================
17 Canlı Veri Kaynağı + 13 Ani Hareket Tetikleyicisi

Bu modül TÜM veri kaynaklarını tek çatı altında toplar.
Notifications.py bu modülü kullanarak kapsamlı bildirimler gönderir.
"""
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("INSTITUTIONAL_AGGREGATOR")


@dataclass
class LiveDataSnapshot:
    """Tüm canlı verilerin anlık görüntüsü"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "BTCUSDT"
    
    # === 17 VERİ KAYNAĞI ===
    
    # 1. Whale Activity
    whale_net_flow: float = 0  # Pozitif = alım, negatif = satım
    whale_trade_count: int = 0
    
    # 2. Order Book
    orderbook_imbalance: float = 1.0  # >1 bid heavy, <1 ask heavy
    orderbook_bid_volume: float = 0
    orderbook_ask_volume: float = 0
    
    # 3. Liquidation Zones
    liq_long_total: float = 0
    liq_short_total: float = 0
    liq_nearest_level: float = 0
    liq_nearest_direction: str = ""  # "LONG" or "SHORT"
    
    # 4. Funding Rate
    funding_rate: float = 0
    funding_predicted: float = 0
    
    # 5. Open Interest
    open_interest: float = 0
    oi_change_1h: float = 0  # Percentage
    oi_change_24h: float = 0
    
    # 6. Long/Short Ratio
    long_short_ratio: float = 1.0
    long_account_pct: float = 50
    short_account_pct: float = 50
    
    # 7. CVD (Cumulative Volume Delta)
    cvd_value: float = 0
    cvd_trend: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    
    # 8. Exchange Inflow/Outflow
    exchange_inflow: float = 0
    exchange_outflow: float = 0
    exchange_netflow: float = 0  # Positive = inflow, negative = outflow
    
    # 9. Stablecoin Supply
    usdt_supply_change: float = 0
    usdc_supply_change: float = 0
    
    # 10. DeFi TVL
    defi_tvl: float = 0
    defi_tvl_change_24h: float = 0
    
    # 11. Options Market
    put_call_ratio: float = 1.0
    max_pain_price: float = 0
    
    # 12. CME Gap
    cme_gap_price: float = 0
    cme_gap_filled: bool = True
    cme_gap_direction: str = ""  # "UP" or "DOWN"
    
    # 13. Cross-Exchange Premium
    binance_price: float = 0
    coinbase_premium: float = 0  # Percentage vs Binance
    bybit_premium: float = 0
    
    # 14. ETF/Grayscale
    etf_flow_daily: float = 0  # In millions USD
    grayscale_premium: float = 0
    
    # 15. Fear & Greed
    fear_greed_index: int = 50
    fear_greed_label: str = "Neutral"
    
    # 16. Network Metrics
    active_addresses: int = 0
    hash_rate: float = 0
    hash_rate_change: float = 0
    
    # 17. Taker Buy/Sell
    taker_buy_ratio: float = 0.5  # 0-1, >0.5 = buyers dominant
    taker_buy_volume: float = 0
    taker_sell_volume: float = 0


@dataclass
class AlertTrigger:
    """Tek bir uyarı tetikleyicisi"""
    name: str
    active: bool = False
    value: str = ""
    severity: str = "LOW"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    direction: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    message: str = ""


@dataclass 
class SuddenAlertSnapshot:
    """13 Ani Hareket Tetikleyicisi"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = "BTCUSDT"
    
    # Alert Triggers
    triggers: List[AlertTrigger] = field(default_factory=list)
    
    # Summary
    active_trigger_count: int = 0
    dominant_direction: str = "NEUTRAL"
    overall_severity: str = "LOW"
    should_alert: bool = False


class InstitutionalAggregator:
    """
    Kurumsal Seviye Veri Toplayıcı
    
    17 veri kaynağını tek çağrıda toplar.
    13 ani hareket tetikleyicisini kontrol eder.
    """
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Tuple[datetime, any]] = {}
        self._cache_ttl = 30  # seconds
        logger.info("🏦 Institutional Aggregator initialized")
    
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
    # ANA VERİ TOPLAMA
    # =========================================
    
    async def get_live_snapshot(self, symbol: str = "BTCUSDT") -> LiveDataSnapshot:
        """
        Tüm 17 veri kaynağından anlık görüntü al.
        Paralel çağrılarla hızlandırılmış.
        """
        snapshot = LiveDataSnapshot(symbol=symbol)
        
        # Paralel veri çekme
        results = await asyncio.gather(
            self._fetch_whale_data(symbol),
            self._fetch_orderbook(symbol),
            self._fetch_liquidation(symbol),
            self._fetch_funding(symbol),
            self._fetch_open_interest(symbol),
            self._fetch_long_short_ratio(symbol),
            self._fetch_cvd(symbol),
            self._fetch_exchange_flow(symbol),
            self._fetch_stablecoin_supply(),
            self._fetch_defi_tvl(),
            self._fetch_options_data(symbol),
            self._fetch_cme_gap(symbol),
            self._fetch_cross_exchange(symbol),
            self._fetch_etf_flow(),
            self._fetch_fear_greed(),
            self._fetch_network_metrics(symbol),
            self._fetch_taker_volume(symbol),
            return_exceptions=True
        )
        
        # Sonuçları snapshot'a yaz
        keys = [
            'whale', 'orderbook', 'liquidation', 'funding', 'oi', 'ls_ratio',
            'cvd', 'exchange_flow', 'stablecoin', 'defi', 'options', 'cme',
            'cross_exchange', 'etf', 'fear_greed', 'network', 'taker'
        ]
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Data fetch error ({keys[i]}): {result}")
                continue
            self._apply_result(snapshot, keys[i], result)
        
        return snapshot
    
    def _apply_result(self, snapshot: LiveDataSnapshot, key: str, data: Dict):
        """Sonuçları snapshot'a uygula."""
        if not data:
            return
        
        if key == 'whale':
            snapshot.whale_net_flow = data.get('net_flow', 0)
            snapshot.whale_trade_count = data.get('trade_count', 0)
        
        elif key == 'orderbook':
            snapshot.orderbook_imbalance = data.get('imbalance', 1.0)
            snapshot.orderbook_bid_volume = data.get('bid_volume', 0)
            snapshot.orderbook_ask_volume = data.get('ask_volume', 0)
        
        elif key == 'liquidation':
            snapshot.liq_long_total = data.get('long_total', 0)
            snapshot.liq_short_total = data.get('short_total', 0)
            snapshot.liq_nearest_level = data.get('nearest_level', 0)
            snapshot.liq_nearest_direction = data.get('nearest_direction', '')
        
        elif key == 'funding':
            snapshot.funding_rate = data.get('rate', 0)
            snapshot.funding_predicted = data.get('predicted', 0)
        
        elif key == 'oi':
            snapshot.open_interest = data.get('value', 0)
            snapshot.oi_change_1h = data.get('change_1h', 0)
            snapshot.oi_change_24h = data.get('change_24h', 0)
        
        elif key == 'ls_ratio':
            snapshot.long_short_ratio = data.get('ratio', 1.0)
            snapshot.long_account_pct = data.get('long_pct', 50)
            snapshot.short_account_pct = data.get('short_pct', 50)
        
        elif key == 'cvd':
            snapshot.cvd_value = data.get('value', 0)
            snapshot.cvd_trend = data.get('trend', 'NEUTRAL')
        
        elif key == 'exchange_flow':
            snapshot.exchange_inflow = data.get('inflow', 0)
            snapshot.exchange_outflow = data.get('outflow', 0)
            snapshot.exchange_netflow = data.get('netflow', 0)
        
        elif key == 'stablecoin':
            snapshot.usdt_supply_change = data.get('usdt_change', 0)
            snapshot.usdc_supply_change = data.get('usdc_change', 0)
        
        elif key == 'defi':
            snapshot.defi_tvl = data.get('tvl', 0)
            snapshot.defi_tvl_change_24h = data.get('change_24h', 0)
        
        elif key == 'options':
            snapshot.put_call_ratio = data.get('put_call', 1.0)
            snapshot.max_pain_price = data.get('max_pain', 0)
        
        elif key == 'cme':
            snapshot.cme_gap_price = data.get('gap_price', 0)
            snapshot.cme_gap_filled = data.get('filled', True)
            snapshot.cme_gap_direction = data.get('direction', '')
        
        elif key == 'cross_exchange':
            snapshot.binance_price = data.get('binance', 0)
            snapshot.coinbase_premium = data.get('coinbase_premium', 0)
            snapshot.bybit_premium = data.get('bybit_premium', 0)
        
        elif key == 'etf':
            snapshot.etf_flow_daily = data.get('daily_flow', 0)
            snapshot.grayscale_premium = data.get('gbtc_premium', 0)
        
        elif key == 'fear_greed':
            snapshot.fear_greed_index = data.get('value', 50)
            snapshot.fear_greed_label = data.get('label', 'Neutral')
        
        elif key == 'network':
            snapshot.active_addresses = data.get('active_addresses', 0)
            snapshot.hash_rate = data.get('hash_rate', 0)
            snapshot.hash_rate_change = data.get('hash_change', 0)
        
        elif key == 'taker':
            snapshot.taker_buy_ratio = data.get('buy_ratio', 0.5)
            snapshot.taker_buy_volume = data.get('buy_volume', 0)
            snapshot.taker_sell_volume = data.get('sell_volume', 0)

    # =========================================
    # ANİ HAREKET TETİKLEYİCİLERİ (13)
    # =========================================
    
    async def check_sudden_triggers(self, symbol: str = "BTCUSDT") -> SuddenAlertSnapshot:
        """
        13 ani hareket tetikleyicisini kontrol et.
        """
        alert = SuddenAlertSnapshot(symbol=symbol)
        
        # Paralel kontroller (15 trigger - 13 mevcut + 2 yeni)
        checks = await asyncio.gather(
            self._check_bollinger_squeeze(symbol),
            self._check_volume_anomaly(symbol),
            self._check_liquidation_cascade(symbol),
            self._check_exchange_divergence(symbol),
            self._check_oi_spike(symbol),
            self._check_funding_extreme(symbol),
            self._check_taker_anomaly(symbol),
            self._check_flash_move(symbol),
            self._check_large_wallet(symbol),
            self._check_smart_money(symbol),
            self._check_market_structure(symbol),
            self._check_volatility_regime(symbol),
            self._check_correlation_breakdown(symbol),
            # NEW: Multi-exchange ve Whale tracker
            self._check_multi_exchange_imbalance(symbol),
            self._check_whale_activity(symbol),
            return_exceptions=True
        )
        
        triggers = []
        for check in checks:
            if isinstance(check, Exception):
                continue
            if check and check.active:
                triggers.append(check)
        
        alert.triggers = triggers
        alert.active_trigger_count = len(triggers)
        
        # Dominant yön hesapla
        bullish = sum(1 for t in triggers if t.direction == "BULLISH")
        bearish = sum(1 for t in triggers if t.direction == "BEARISH")
        
        if bullish > bearish:
            alert.dominant_direction = "BULLISH"
        elif bearish > bullish:
            alert.dominant_direction = "BEARISH"
        else:
            alert.dominant_direction = "NEUTRAL"
        
        # Severity hesapla
        critical = any(t.severity == "CRITICAL" for t in triggers)
        high = any(t.severity == "HIGH" for t in triggers)
        
        if critical:
            alert.overall_severity = "CRITICAL"
        elif high:
            alert.overall_severity = "HIGH"
        elif len(triggers) >= 3:
            alert.overall_severity = "MEDIUM"
        else:
            alert.overall_severity = "LOW"
        
        # Alert gönderilmeli mi? (1 trigger yeterli - daha hassas)
        alert.should_alert = alert.active_trigger_count >= 1 or critical
        
        return alert

    # =========================================
    # VERİ ÇEKME FONKSİYONLARI (17)
    # =========================================
    
    async def _fetch_whale_data(self, symbol: str) -> Dict:
        """1. Whale Tracker"""
        try:
            from src.brain.whale_tracker import get_whale_tracker
            tracker = get_whale_tracker()
            summary = tracker.get_whale_summary()
            return {
                'net_flow': summary.get('net_flow_usd', 0),
                'trade_count': summary.get('whale_trade_count', 0)
            }
        except Exception as e:
            logger.debug(f"Whale fetch error: {e}")
            return {}
    
    async def _fetch_orderbook(self, symbol: str) -> Dict:
        """2. Order Book Depth"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
            async with session.get(url) as resp:
                data = await resp.json()
                
            bids = sum(float(b[1]) for b in data.get('bids', []))
            asks = sum(float(a[1]) for a in data.get('asks', []))
            
            return {
                'bid_volume': bids,
                'ask_volume': asks,
                'imbalance': bids / asks if asks > 0 else 1.0
            }
        except Exception as e:
            logger.debug(f"Orderbook fetch error: {e}")
            return {}
    
    async def _fetch_liquidation(self, symbol: str) -> Dict:
        """3. Liquidation Zones"""
        try:
            from src.brain.coinglass_scraper import get_cg_scraper
            scraper = get_cg_scraper()
            base = symbol.replace('USDT', '')
            data = await scraper.get_liquidation_data(base)
            
            return {
                'long_total': data.get('long_liquidation', 0),
                'short_total': data.get('short_liquidation', 0),
                'nearest_level': data.get('magnet_price', 0),
                'nearest_direction': data.get('direction', '')
            }
        except Exception as e:
            logger.debug(f"Liquidation fetch error: {e}")
            return {}
    
    async def _fetch_funding(self, symbol: str) -> Dict:
        """4. Funding Rate"""
        try:
            session = await self._get_session()
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            async with session.get(url) as resp:
                data = await resp.json()
            
            rate = float(data[0]['fundingRate']) if data else 0
            return {
                'rate': rate * 100,  # Convert to percentage
                'predicted': rate * 100 * 3  # 8h prediction
            }
        except Exception as e:
            logger.debug(f"Funding fetch error: {e}")
            return {}
    
    async def _fetch_open_interest(self, symbol: str) -> Dict:
        """5. Open Interest"""
        try:
            session = await self._get_session()
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url) as resp:
                data = await resp.json()
            
            oi = float(data.get('openInterest', 0))
            return {
                'value': oi,
                'change_1h': 0,  # Would need historical data
                'change_24h': 0
            }
        except Exception as e:
            logger.debug(f"OI fetch error: {e}")
            return {}
    
    async def _fetch_long_short_ratio(self, symbol: str) -> Dict:
        """6. Long/Short Ratio"""
        try:
            session = await self._get_session()
            url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data:
                ratio = float(data[0].get('longShortRatio', 1))
                long_pct = float(data[0].get('longAccount', 50)) * 100
                short_pct = float(data[0].get('shortAccount', 50)) * 100
                return {
                    'ratio': ratio,
                    'long_pct': long_pct,
                    'short_pct': short_pct
                }
            return {}
        except Exception as e:
            logger.debug(f"L/S ratio fetch error: {e}")
            return {}
    
    async def _fetch_cvd(self, symbol: str) -> Dict:
        """7. CVD (Cumulative Volume Delta)"""
        try:
            # CVD hesaplaması için taker buy/sell kullanıyoruz
            session = await self._get_session()
            url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=12"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data:
                # Son 1 saatlik CVD trendi
                buy_total = sum(float(d.get('buyVol', 0)) for d in data)
                sell_total = sum(float(d.get('sellVol', 0)) for d in data)
                cvd = buy_total - sell_total
                
                trend = "BULLISH" if cvd > 0 else "BEARISH" if cvd < 0 else "NEUTRAL"
                
                return {
                    'value': cvd,
                    'trend': trend
                }
            return {}
        except Exception as e:
            logger.debug(f"CVD fetch error: {e}")
            return {}
    
    async def _fetch_exchange_flow(self, symbol: str) -> Dict:
        """8. Exchange Inflow/Outflow - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            base = symbol.replace('USDT', '')
            data = await scraper.scrape_exchange_flow(base)
            return {
                'inflow': data.get('inflow', 0),
                'outflow': data.get('outflow', 0),
                'netflow': data.get('netflow', 0)
            }
        except Exception as e:
            logger.debug(f"Exchange flow scrape error: {e}")
            return {'inflow': 0, 'outflow': 0, 'netflow': 0}
    
    async def _fetch_stablecoin_supply(self) -> Dict:
        """9. Stablecoin Supply Changes - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            data = await scraper.scrape_stablecoin_supply()
            return {
                'usdt_change': data.get('usdt_change', 0),
                'usdc_change': data.get('usdc_change', 0)
            }
        except Exception as e:
            logger.debug(f"Stablecoin scrape error: {e}")
            return {'usdt_change': 0, 'usdc_change': 0}
    
    async def _fetch_defi_tvl(self) -> Dict:
        """10. DeFi TVL"""
        try:
            session = await self._get_session()
            url = "https://api.llama.fi/protocols"
            async with session.get(url) as resp:
                data = await resp.json()
            
            total_tvl = sum(p.get('tvl', 0) for p in data[:50])  # Top 50
            return {
                'tvl': total_tvl,
                'change_24h': 0
            }
        except Exception as e:
            logger.debug(f"DeFi TVL fetch error: {e}")
            return {}
    
    async def _fetch_options_data(self, symbol: str) -> Dict:
        """11. Options Market - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            base = symbol.replace('USDT', '')
            data = await scraper.scrape_options_data(base)
            return {
                'put_call': data.get('put_call', 1.0),
                'max_pain': data.get('max_pain', 0)
            }
        except Exception as e:
            logger.debug(f"Options scrape error: {e}")
            return {'put_call': 1.0, 'max_pain': 0}
    
    async def _fetch_cme_gap(self, symbol: str) -> Dict:
        """12. CME Gap - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            data = await scraper.scrape_cme_gap()
            return {
                'gap_price': data.get('gap_price', 0),
                'filled': data.get('filled', True),
                'direction': data.get('direction', '')
            }
        except Exception as e:
            logger.debug(f"CME gap scrape error: {e}")
            return {'gap_price': 0, 'filled': True, 'direction': ''}
    
    async def _fetch_cross_exchange(self, symbol: str) -> Dict:
        """13. Cross-Exchange Premium"""
        try:
            session = await self._get_session()
            
            # Binance
            async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as resp:
                bin_data = await resp.json()
                binance_price = float(bin_data.get('price', 0))
            
            # Bybit
            try:
                async with session.get(f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}") as resp:
                    bybit_data = await resp.json()
                    bybit_price = float(bybit_data.get('result', {}).get('list', [{}])[0].get('lastPrice', 0))
            except:
                bybit_price = binance_price
            
            bybit_premium = ((bybit_price - binance_price) / binance_price) * 100 if binance_price > 0 else 0
            
            return {
                'binance': binance_price,
                'coinbase_premium': 0,  # Would need Coinbase API
                'bybit_premium': bybit_premium
            }
        except Exception as e:
            logger.debug(f"Cross-exchange fetch error: {e}")
            return {}
    
    async def _fetch_etf_flow(self) -> Dict:
        """14. ETF/Grayscale Flow - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            etf_data = await scraper.scrape_etf_flow()
            gbtc_data = await scraper.scrape_grayscale_premium()
            return {
                'daily_flow': etf_data.get('daily_flow', 0),
                'gbtc_premium': gbtc_data.get('premium', 0)
            }
        except Exception as e:
            logger.debug(f"ETF flow scrape error: {e}")
            return {'daily_flow': 0, 'gbtc_premium': 0}
    
    async def _fetch_fear_greed(self) -> Dict:
        """15. Fear & Greed Index"""
        try:
            session = await self._get_session()
            url = "https://api.alternative.me/fng/"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data.get('data'):
                val = int(data['data'][0].get('value', 50))
                label = data['data'][0].get('value_classification', 'Neutral')
                return {
                    'value': val,
                    'label': label
                }
            return {}
        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")
            return {}
    
    async def _fetch_network_metrics(self, symbol: str) -> Dict:
        """16. Network Metrics - WEB SCRAPING"""
        try:
            from src.brain.web_scrapers import get_web_scraper
            scraper = get_web_scraper()
            data = await scraper.scrape_network_metrics()
            return {
                'active_addresses': data.get('active_addresses', 0),
                'hash_rate': data.get('hash_rate', 0),
                'hash_change': data.get('hash_change', 0)
            }
        except Exception as e:
            logger.debug(f"Network metrics scrape error: {e}")
            return {'active_addresses': 0, 'hash_rate': 0, 'hash_change': 0}
    
    async def _fetch_taker_volume(self, symbol: str) -> Dict:
        """17. Taker Buy/Sell Ratio"""
        try:
            session = await self._get_session()
            url = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={symbol}&period=5m&limit=1"
            async with session.get(url) as resp:
                data = await resp.json()
            
            if data:
                buy_vol = float(data[0].get('buyVol', 0))
                sell_vol = float(data[0].get('sellVol', 0))
                total = buy_vol + sell_vol
                buy_ratio = buy_vol / total if total > 0 else 0.5
                
                return {
                    'buy_ratio': buy_ratio,
                    'buy_volume': buy_vol,
                    'sell_volume': sell_vol
                }
            return {}
        except Exception as e:
            logger.debug(f"Taker volume fetch error: {e}")
            return {}

    # =========================================
    # TETİKLEYİCİ KONTROL FONKSİYONLARI (13)
    # =========================================
    
    async def _check_bollinger_squeeze(self, symbol: str) -> AlertTrigger:
        """1. Bollinger Squeeze"""
        try:
            from src.brain.bollinger_squeeze import BollingerSqueezeDetector
            detector = BollingerSqueezeDetector()
            result = await detector.check_squeeze(symbol)
            
            if result.get('squeeze_active'):
                return AlertTrigger(
                    name="Bollinger Squeeze",
                    active=True,
                    value=f"{result.get('bandwidth_pct', 0):.1f}%",
                    severity="HIGH" if result.get('breakout_imminent') else "MEDIUM",
                    direction="NEUTRAL",
                    message="Sıkışma tespit edildi - Patlama yaklaşıyor!"
                )
        except:
            pass
        return AlertTrigger(name="Bollinger Squeeze", active=False)
    
    async def _check_volume_anomaly(self, symbol: str) -> AlertTrigger:
        """2. Volume Anomaly"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=20"
            async with session.get(url) as resp:
                klines = await resp.json()
            
            volumes = [float(k[5]) for k in klines[:-1]]
            avg_vol = sum(volumes) / len(volumes)
            current_vol = float(klines[-1][5])
            
            ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            if ratio > 1.5:
                return AlertTrigger(
                    name="Volume Spike",
                    active=True,
                    value=f"{ratio:.1f}x",
                    severity="HIGH" if ratio > 2.5 else "MEDIUM",
                    direction="NEUTRAL",
                    message=f"Hacim normal seviyenin {ratio:.1f} katı!"
                )
        except:
            pass
        return AlertTrigger(name="Volume Anomaly", active=False)
    
    async def _check_liquidation_cascade(self, symbol: str) -> AlertTrigger:
        """3. Liquidation Cascade Risk"""
        try:
            data = await self._fetch_liquidation(symbol)
            funding_data = await self._fetch_funding(symbol)
            
            long_liq = data.get('long_total', 0)
            short_liq = data.get('short_total', 0)
            funding = funding_data.get('rate', 0)
            
            # Long squeeze riski
            if funding > 0.05 and long_liq > short_liq * 1.5:
                return AlertTrigger(
                    name="Long Squeeze Risk",
                    active=True,
                    value=f"FR: {funding:.3f}%",
                    severity="HIGH",
                    direction="BEARISH",
                    message="Yüksek long likidasyonu + pozitif funding = Dump riski!"
                )
            
            # Short squeeze riski
            if funding < -0.02 and short_liq > long_liq * 1.5:
                return AlertTrigger(
                    name="Short Squeeze Risk",
                    active=True,
                    value=f"FR: {funding:.3f}%",
                    severity="HIGH",
                    direction="BULLISH",
                    message="Yüksek short likidasyonu + negatif funding = Pump riski!"
                )
        except:
            pass
        return AlertTrigger(name="Liquidation Cascade", active=False)
    
    async def _check_exchange_divergence(self, symbol: str) -> AlertTrigger:
        """4. Exchange Price Divergence"""
        try:
            data = await self._fetch_cross_exchange(symbol)
            bybit_prem = abs(data.get('bybit_premium', 0))
            
            if bybit_prem > 0.3:
                direction = "BULLISH" if data.get('bybit_premium', 0) > 0 else "BEARISH"
                return AlertTrigger(
                    name="Exchange Divergence",
                    active=True,
                    value=f"{bybit_prem:.2f}%",
                    severity="MEDIUM",
                    direction=direction,
                    message=f"Bybit {'premium' if direction == 'BULLISH' else 'discount'} tespit edildi"
                )
        except:
            pass
        return AlertTrigger(name="Exchange Divergence", active=False)
    
    async def _check_oi_spike(self, symbol: str) -> AlertTrigger:
        """5. Sudden OI Spike/Drop - AKTIF"""
        try:
            session = await self._get_session()
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            async with session.get(url) as resp:
                data = await resp.json()
            oi_current = float(data.get('openInterest', 0))
            
            # Get price
            async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as resp:
                price_data = await resp.json()
            price = float(price_data.get('price', 0))
            oi_usd = oi_current * price
            
            # Store for comparison
            if not hasattr(self, '_last_oi'):
                self._last_oi = {}
            
            if symbol in self._last_oi:
                last_oi = self._last_oi[symbol]
                change_pct = ((oi_usd - last_oi) / last_oi * 100) if last_oi > 0 else 0
                
                if abs(change_pct) > 5:  # 5%+ OI change
                    direction = "BULLISH" if change_pct > 0 else "BEARISH"
                    self._last_oi[symbol] = oi_usd
                    return AlertTrigger(
                        name="OI Spike" if change_pct > 0 else "OI Drop",
                        active=True,
                        value=f"{change_pct:+.1f}%",
                        severity="HIGH" if abs(change_pct) > 10 else "MEDIUM",
                        direction=direction,
                        message=f"Open Interest {change_pct:+.1f}% değişti! (${oi_usd/1e9:.1f}B)"
                    )
            
            self._last_oi[symbol] = oi_usd
        except Exception as e:
            logger.debug(f"OI spike check error: {e}")
        return AlertTrigger(name="OI Spike", active=False)
    
    async def _check_funding_extreme(self, symbol: str) -> AlertTrigger:
        """6. Funding Rate Extreme"""
        try:
            data = await self._fetch_funding(symbol)
            rate = data.get('rate', 0)
            
            if abs(rate) > 0.05:  # Eşik düşürüldü (0.1 → 0.05)
                direction = "BEARISH" if rate > 0 else "BULLISH"
                squeeze_type = "Long" if rate > 0 else "Short"
                return AlertTrigger(
                    name="Extreme Funding",
                    active=True,
                    value=f"{rate:.3f}%",
                    severity="HIGH",
                    direction=direction,
                    message=f"Aşırı funding! {squeeze_type} squeeze riski yüksek."
                )
        except:
            pass
        return AlertTrigger(name="Funding Extreme", active=False)
    
    async def _check_taker_anomaly(self, symbol: str) -> AlertTrigger:
        """7. Abnormal Taker Flow"""
        try:
            data = await self._fetch_taker_volume(symbol)
            ratio = data.get('buy_ratio', 0.5)
            
            if ratio > 0.7:
                return AlertTrigger(
                    name="Taker Buy Dominance",
                    active=True,
                    value=f"{ratio*100:.0f}%",
                    severity="MEDIUM",
                    direction="BULLISH",
                    message="Agresif alıcılar dominant!"
                )
            elif ratio < 0.3:
                return AlertTrigger(
                    name="Taker Sell Dominance",
                    active=True,
                    value=f"{ratio*100:.0f}%",
                    severity="MEDIUM",
                    direction="BEARISH",
                    message="Agresif satıcılar dominant!"
                )
        except:
            pass
        return AlertTrigger(name="Taker Anomaly", active=False)
    
    async def _check_flash_move(self, symbol: str) -> AlertTrigger:
        """8. Flash Crash/Pump"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=5"
            async with session.get(url) as resp:
                klines = await resp.json()
            
            if len(klines) >= 2:
                prev_close = float(klines[-2][4])
                curr_close = float(klines[-1][4])
                change_pct = ((curr_close - prev_close) / prev_close) * 100
                
                if abs(change_pct) > 1.5:
                    direction = "BULLISH" if change_pct > 0 else "BEARISH"
                    return AlertTrigger(
                        name="Flash Move",
                        active=True,
                        value=f"{change_pct:+.2f}%",
                        severity="CRITICAL",
                        direction=direction,
                        message=f"1 dakikada {change_pct:+.2f}% hareket!"
                    )
        except:
            pass
        return AlertTrigger(name="Flash Move", active=False)
    
    async def _check_large_wallet(self, symbol: str) -> AlertTrigger:
        """9. Large Wallet Movement"""
        # Would need blockchain API
        return AlertTrigger(name="Large Wallet", active=False)
    
    async def _check_smart_money(self, symbol: str) -> AlertTrigger:
        """10. Smart Money Pattern"""
        # Would need advanced pattern detection
        return AlertTrigger(name="Smart Money", active=False)
    
    async def _check_market_structure(self, symbol: str) -> AlertTrigger:
        """11. Market Structure Break"""
        # Would need swing high/low analysis
        return AlertTrigger(name="Market Structure", active=False)
    
    async def _check_volatility_regime(self, symbol: str) -> AlertTrigger:
        """12. Volatility Regime Change"""
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=24"
            async with session.get(url) as resp:
                klines = await resp.json()
            
            ranges = [(float(k[2]) - float(k[3])) / float(k[4]) * 100 for k in klines]
            avg_range = sum(ranges[:-1]) / len(ranges[:-1])
            current_range = ranges[-1]
            
            if current_range > avg_range * 2:
                return AlertTrigger(
                    name="Volatility Spike",
                    active=True,
                    value=f"{current_range:.1f}%",
                    severity="MEDIUM",
                    direction="NEUTRAL",
                    message="Volatilite normal seviyenin 2 katı üzerinde!"
                )
        except:
            pass
        return AlertTrigger(name="Volatility Regime", active=False)
    
    async def _check_correlation_breakdown(self, symbol: str) -> AlertTrigger:
        """13. Correlation Breakdown (BTC/ETH)"""
        # Would need multi-asset analysis
        return AlertTrigger(name="Correlation Breakdown", active=False)
    
    async def _check_multi_exchange_imbalance(self, symbol: str) -> AlertTrigger:
        """14. Multi-Exchange Order Book Imbalance (NEW)"""
        try:
            from src.brain.multi_exchange_orderbook import get_multi_exchange_orderbook
            orderbook = get_multi_exchange_orderbook()
            data = await orderbook.get_aggregated_orderbook(symbol)
            
            # Check for significant cross-exchange imbalance
            if data.overall_imbalance > 2.5 or data.overall_imbalance < 0.4:
                direction = "BULLISH" if data.overall_imbalance > 1 else "BEARISH"
                wall_type = "BID WALL" if data.bid_wall_detected else "ASK WALL" if data.ask_wall_detected else "IMBALANCE"
                
                severity = "HIGH" if data.overall_imbalance > 3 or data.overall_imbalance < 0.33 else "MEDIUM"
                
                return AlertTrigger(
                    name="Multi-Exchange Imbalance",
                    active=True,
                    value=f"{data.overall_imbalance:.2f}x ({len(data.exchanges)} borsa)",
                    severity=severity,
                    direction=direction,
                    message=f"{wall_type} tespit edildi! Dominant: {data.dominant_exchange.upper()}"
                )
            
            # Check for significant price divergence across exchanges
            if data.price_divergence > 0.5:
                return AlertTrigger(
                    name="Cross-Exchange Divergence",
                    active=True,
                    value=f"%{data.price_divergence:.2f}",
                    severity="MEDIUM",
                    direction="NEUTRAL",
                    message=f"Borsalar arası fiyat farkı! Arbitraj fırsatı olabilir."
                )
        except Exception as e:
            logger.debug(f"Multi-exchange imbalance check error: {e}")
        
        return AlertTrigger(name="Multi-Exchange Imbalance", active=False)
    
    async def _check_whale_activity(self, symbol: str) -> AlertTrigger:
        """15. Whale Wallet Activity (NEW)"""
        try:
            from src.brain.whale_wallet_tracker import get_whale_wallet_tracker
            tracker = get_whale_wallet_tracker()
            activity = await tracker.get_whale_activity(symbol)
            
            # Check for significant whale movement
            if activity.signal in ["STRONG_BUY", "STRONG_SELL"]:
                direction = "BULLISH" if activity.signal == "STRONG_BUY" else "BEARISH"
                
                return AlertTrigger(
                    name="Whale Activity",
                    active=True,
                    value=f"${activity.total_volume_24h/1e6:.1f}M",
                    severity="HIGH",
                    direction=direction,
                    message=activity.signal_reason
                )
            
            # Check for large individual transactions
            if activity.large_tx_count >= 3:
                # Multiple large transactions indicate institutional activity
                net_flow = activity.exchange_outflow_24h - activity.exchange_inflow_24h
                direction = "BULLISH" if net_flow > 0 else "BEARISH" if net_flow < 0 else "NEUTRAL"
                
                return AlertTrigger(
                    name="Large Wallet Activity",
                    active=True,
                    value=f"{activity.large_tx_count} büyük TX",
                    severity="MEDIUM",
                    direction=direction,
                    message=f"Net flow: ${net_flow/1e6:.1f}M"
                )
        except Exception as e:
            logger.debug(f"Whale activity check error: {e}")
        
        return AlertTrigger(name="Whale Activity", active=False)


# =========================================
# GLOBAL INSTANCE
# =========================================

_aggregator: Optional[InstitutionalAggregator] = None

def get_aggregator() -> InstitutionalAggregator:
    """Get or create aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = InstitutionalAggregator()
    return _aggregator
