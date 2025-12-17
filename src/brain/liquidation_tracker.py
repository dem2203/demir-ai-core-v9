"""
DEMIR AI - Liquidation Tracker
Likidasyon seviyelerini hesaplar ve cascade riskini tespit eder.

PHASE 42: Critical Scraper - No API Key Required
- Binance public API kullanır (auth gerektirmez)
- Open Interest, Funding Rate, Long/Short ratio
- Approximate liquidation zones (10x, 25x, 50x, 100x leverage)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("LIQUIDATION_TRACKER")


@dataclass
class LiquidationZone:
    """Likidasyon bölgesi"""
    symbol: str
    price_level: float
    size_usd: float  # Estimated liquidation size
    leverage: int
    side: str  # LONG or SHORT
    distance_pct: float  # Distance from current price


class LiquidationTracker:
    """
    Likidasyon Seviyesi Takipçisi
    
    API key gerektirmez - Binance public endpoints kullanır.
    
    Nasıl çalışır:
    1. Open Interest + Long/Short ratio alır
    2. Leverage dağılımını tahmin eder (10x, 25x, 50x, 100x)
    3. Her leverage için liquidation price hesaplar
    4. Büyük cluster'ları ($100M+) tespit eder
    """
    
    # Binance public endpoints (no auth required)
    BASE_URL = "https://fapi.binance.com"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    # Leverage assumptions (market distribution estimate)
    LEVERAGE_DISTRIBUTION = {
        10: 0.15,   # 15% of traders
        25: 0.30,   # 30% of traders
        50: 0.35,   # 35% of traders
        100: 0.20,  # 20% of traders
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch = {}
    
    def get_liquidation_zones(self, symbol: str = 'BTCUSDT') -> List[LiquidationZone]:
        """
        Likidasyon bölgelerini hesapla.
        
        Returns:
            List of significant liquidation zones
        """
        cache_key = f'liq_zones_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # 1. Get current market data
            market_data = self._get_market_data(symbol)
            if not market_data:
                return []
            
            current_price = market_data['price']
            oi = market_data['open_interest']
            ls_ratio = market_data['long_short_ratio']
            
            # 2. Calculate OI distribution between longs and shorts
            # L/S ratio = 2.0 means 67% long, 33% short
            long_pct = ls_ratio / (1 + ls_ratio) if ls_ratio > 0 else 0.5
            short_pct = 1 - long_pct
            
            long_oi_usd = oi * long_pct
            short_oi_usd = oi * short_pct
            
            # 3. Calculate liquidation levels for each leverage
            zones = []
            
            for leverage, distribution in self.LEVERAGE_DISTRIBUTION.items():
                # Long liquidations (price goes DOWN)
                long_liq_price = current_price * (1 - (0.95 / leverage))  # 95% margin before liquidation
                long_liq_size = long_oi_usd * distribution
                
                if long_liq_size > 10_000_000:  # $10M+ clusters
                    distance_pct = ((current_price - long_liq_price) / current_price) * 100
                    
                    zones.append(LiquidationZone(
                        symbol=symbol,
                        price_level=long_liq_price,
                        size_usd=long_liq_size,
                        leverage=leverage,
                        side='LONG',
                        distance_pct=distance_pct
                    ))
                
                # Short liquidations (price goes UP)
                short_liq_price = current_price * (1 + (0.95 / leverage))
                short_liq_size = short_oi_usd * distribution
                
                if short_liq_size > 10_000_000:  # $10M+ clusters
                    distance_pct = ((short_liq_price - current_price) / current_price) * 100
                    
                    zones.append(LiquidationZone(
                        symbol=symbol,
                        price_level=short_liq_price,
                        size_usd=short_liq_size,
                        leverage=leverage,
                        side='SHORT',
                        distance_pct=distance_pct
                    ))
            
            # 4. Sort by proximity to current price
            zones.sort(key=lambda z: abs(z.distance_pct))
            
            # 5. Cache and return
            self._set_cache(cache_key, zones)
            logger.info(f"Calculated {len(zones)} liquidation zones for {symbol}")
            
            return zones
            
        except Exception as e:
            logger.warning(f"Liquidation zones calculation failed: {e}")
            return []
    
    def get_liquidation_summary(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Likidasyon özeti oluştur (Dashboard için).
        """
        zones = self.get_liquidation_zones(symbol)
        
        if not zones:
            return {
                'symbol': symbol,
                'zone_count': 0,
                'nearest_long_liq': None,
                'nearest_short_liq': None,
                'cascade_risk': 'LOW',
                'summary': 'Likidasyon verisi hesaplanamadı'
            }
        
        # Find nearest liquidations
        long_zones = [z for z in zones if z.side == 'LONG']
        short_zones = [z for z in zones if z.side == 'SHORT']
        
        nearest_long = min(long_zones, key=lambda z: abs(z.distance_pct)) if long_zones else None
        nearest_short = min(short_zones, key=lambda z: abs(z.distance_pct)) if short_zones else None
        
        # Calculate cascade risk
        # High risk if $500M+ within 3% of current price
        nearby_size = sum(z.size_usd for z in zones if abs(z.distance_pct) < 3)
        
        if nearby_size > 500_000_000:
            cascade_risk = 'HIGH'
            risk_emoji = '🔴'
        elif nearby_size > 200_000_000:
            cascade_risk = 'MEDIUM'
            risk_emoji = '🟡'
        else:
            cascade_risk = 'LOW'
            risk_emoji = '🟢'
        
        # Generate summary
        summary = f"{risk_emoji} Cascade Risk: {cascade_risk} "
        if nearest_long:
            summary += f"| Nearest Long Liq: ${nearest_long.price_level:,.0f} ({nearest_long.distance_pct:.1f}% away)"
        if nearest_short:
            summary += f" | Nearest Short Liq: ${nearest_short.price_level:,.0f} (+{nearest_short.distance_pct:.1f}%)"
        
        return {
            'symbol': symbol,
            'zone_count': len(zones),
            'nearest_long_liq': nearest_long,
            'nearest_short_liq': nearest_short,
            'cascade_risk': cascade_risk,
            'nearby_liq_size': nearby_size,
            'summary': summary,
            'zones': zones[:10],  # Top 10 zones
            'timestamp': datetime.now()
        }
    
    def format_for_telegram(self, symbol: str = 'BTCUSDT') -> str:
        """Telegram formatı"""
        summary = self.get_liquidation_summary(symbol)
        
        msg = "⚡ *Liquidation Heatmap*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        msg += f"🪙 **{symbol}**\n"
        msg += f"📊 Cascade Risk: **{summary['cascade_risk']}**\n"
        msg += f"💥 Nearby Liquidations: ${summary['nearby_liq_size']/1e9:.2f}B\n\n"
        
        # Nearest long liq
        if summary['nearest_long_liq']:
            nl = summary['nearest_long_liq']
            msg += f"🔴 **Long Liquidation:**\n"
            msg += f"  Price: `${nl.price_level:,.0f}` ({nl.distance_pct:.1f}% below)\n"
            msg += f"  Size: ${nl.size_usd/1e6:.0f}M ({nl.leverage}x)\n\n"
        
        # Nearest short liq
        if summary['nearest_short_liq']:
            ns = summary['nearest_short_liq']
            msg += f"🟢 **Short Liquidation:**\n"
            msg += f"  Price: `${ns.price_level:,.0f}` (+{ns.distance_pct:.1f}% above)\n"
            msg += f"  Size: ${ns.size_usd/1e6:.0f}M ({ns.leverage}x)\n\n"
        
        msg += f"⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
    
    # =========================================
    # PRIVATE HELPERS
    # =========================================
    
    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Piyasa verilerini al (Binance public API).
        """
        try:
            # 1. Current price
            ticker_url = f"{self.BASE_URL}/fapi/v1/ticker/price?symbol={symbol}"
            ticker_resp = requests.get(ticker_url, timeout=10, headers=self.HEADERS)
            price = float(ticker_resp.json()['price']) if ticker_resp.status_code == 200 else 0
            
            # 2. Open Interest
            oi_url = f"{self.BASE_URL}/fapi/v1/openInterest?symbol={symbol}"
            oi_resp = requests.get(oi_url, timeout=10, headers=self.HEADERS)
            oi_btc = float(oi_resp.json()['openInterest']) if oi_resp.status_code == 200 else 0
            oi_usd = oi_btc * price  # Convert to USD
            
            # 3. Long/Short ratio (global account ratio)
            ls_url = f"{self.BASE_URL}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1"
            ls_resp = requests.get(ls_url, timeout=10, headers=self.HEADERS)
            ls_ratio = float(ls_resp.json()[0]['longShortRatio']) if ls_resp.status_code == 200 and ls_resp.json() else 1.0
            
            if price > 0 and oi_usd > 0:
                return {
                    'price': price,
                    'open_interest': oi_usd,
                    'long_short_ratio': ls_ratio,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logger.warning(f"Market data fetch failed: {e}")
        
        return None
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and fresh"""
        if key not in self.cache or key not in self.last_fetch:
            return False
        
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Cache data"""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()
