# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - MACRO CONTEXT
============================
BTC Dominance, Fear & Greed Index, Total Market Cap gibi
makro verileri toplayan modül.

Tüm API'ler ücretsiz ve key gerektirmez.
"""
import logging
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict

logger = logging.getLogger("MACRO_CONTEXT")


@dataclass
class MacroContext:
    """Makro piyasa bağlamı"""
    # BTC Dominance
    btc_dominance: float = 0.0
    btc_dominance_change_24h: float = 0.0
    
    # Market Cap
    total_market_cap: float = 0.0
    total_market_cap_change_24h: float = 0.0
    
    # Fear & Greed
    fear_greed_index: int = 50
    fear_greed_label: str = "Neutral"
    
    # Altcoin Season
    altcoin_season_index: int = 50  # 0-100, >75 = altcoin season
    
    # Timestamps
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'btc_dominance': self.btc_dominance,
            'btc_dominance_change_24h': self.btc_dominance_change_24h,
            'total_market_cap': self.total_market_cap,
            'total_market_cap_change_24h': self.total_market_cap_change_24h,
            'fear_greed_index': self.fear_greed_index,
            'fear_greed_label': self.fear_greed_label,
            'altcoin_season_index': self.altcoin_season_index,
            'last_updated': self.last_updated
        }
    
    def get_market_sentiment(self) -> str:
        """Genel piyasa sentiment özeti"""
        signals = []
        
        # Fear & Greed
        if self.fear_greed_index < 25:
            signals.append("🔴 Extreme Fear (Contrarian BUY)")
        elif self.fear_greed_index < 40:
            signals.append("🟠 Fear")
        elif self.fear_greed_index > 75:
            signals.append("🔴 Extreme Greed (Contrarian SELL)")
        elif self.fear_greed_index > 60:
            signals.append("🟢 Greed")
        else:
            signals.append("⚪ Neutral")
        
        # BTC Dominance
        if self.btc_dominance_change_24h < -1:
            signals.append("📉 BTC.D düşüyor (Altcoin sezonu?)")
        elif self.btc_dominance_change_24h > 1:
            signals.append("📈 BTC.D yükseliyor (Risk-off)")
        
        return " | ".join(signals)


class MacroContextCollector:
    """Makro veri toplayıcı"""
    
    COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
    FEAR_GREED_API = "https://api.alternative.me/fng/"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    
    def __init__(self):
        self._cache: Optional[MacroContext] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)  # 5 dakika cache
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info("📊 Macro Context Collector initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.HEADERS)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_context(self, force_refresh: bool = False) -> MacroContext:
        """Makro context al (cached)"""
        
        # Cache check
        if not force_refresh and self._cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache
        
        context = MacroContext()
        
        try:
            session = await self._get_session()
            
            # 1. CoinGecko Global Data
            try:
                async with session.get(self.COINGECKO_GLOBAL, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        global_data = data.get('data', {})
                        
                        # BTC Dominance
                        context.btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                        
                        # Total Market Cap
                        context.total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                        context.total_market_cap_change_24h = global_data.get('market_cap_change_percentage_24h_usd', 0)
                        
                        logger.debug(f"CoinGecko: BTC.D={context.btc_dominance:.1f}%, MCap=${context.total_market_cap/1e12:.2f}T")
            except Exception as e:
                logger.warning(f"CoinGecko error: {e}")
            
            # 2. Fear & Greed Index
            try:
                async with session.get(self.FEAR_GREED_API, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fng_data = data.get('data', [{}])[0]
                        
                        context.fear_greed_index = int(fng_data.get('value', 50))
                        context.fear_greed_label = fng_data.get('value_classification', 'Neutral')
                        
                        logger.debug(f"Fear & Greed: {context.fear_greed_index} ({context.fear_greed_label})")
            except Exception as e:
                logger.warning(f"Fear & Greed error: {e}")
            
            # 3. Altcoin Season Index (calculated)
            # BTC.D < 40% and dropping = Altcoin season
            if context.btc_dominance > 0:
                if context.btc_dominance < 40:
                    context.altcoin_season_index = 90
                elif context.btc_dominance < 45:
                    context.altcoin_season_index = 75
                elif context.btc_dominance < 50:
                    context.altcoin_season_index = 60
                elif context.btc_dominance < 55:
                    context.altcoin_season_index = 50
                else:
                    context.altcoin_season_index = 30
                
                # Adjust for trend
                if context.btc_dominance_change_24h < -1:
                    context.altcoin_season_index += 10
                elif context.btc_dominance_change_24h > 1:
                    context.altcoin_season_index -= 10
                
                context.altcoin_season_index = max(0, min(100, context.altcoin_season_index))
            
            context.last_updated = datetime.now().isoformat()
            
            # Update cache
            self._cache = context
            self._cache_time = datetime.now()
            
            logger.info(f"📊 Macro Context updated: BTC.D={context.btc_dominance:.1f}%, Fear={context.fear_greed_index}")
            
        except Exception as e:
            logger.error(f"Macro context collection error: {e}")
        
        return context


# Singleton
_macro_collector: Optional[MacroContextCollector] = None


def get_macro_context_collector() -> MacroContextCollector:
    global _macro_collector
    if _macro_collector is None:
        _macro_collector = MacroContextCollector()
    return _macro_collector


async def get_macro_context() -> MacroContext:
    """Quick access to macro context"""
    collector = get_macro_context_collector()
    return await collector.get_context()
