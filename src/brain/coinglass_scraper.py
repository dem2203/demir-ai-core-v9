# -*- coding: utf-8 -*-
"""
DEMIR AI - Coinglass Hyperliquid Scraper
Whale pozisyonları, likidasyon seviyeleri ve sentiment analizi.

PHASE 48: Advanced Whale Intelligence
- Whale pozisyonları (büyük trader'lar)
- Likidasyon fiyatları (hunt seviyeleri)
- Long/Short ratio (sentiment)
- Real-time whale hareketleri

Uses Playwright for JavaScript-rendered content.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("COINGLASS_SCRAPER")

# Try to import Playwright
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - Coinglass scraping disabled")


@dataclass
class WhalePosition:
    """Whale pozisyon bilgisi"""
    wallet: str
    symbol: str
    side: str  # LONG / SHORT
    size_usd: float
    leverage: float
    entry_price: float
    liq_price: float
    pnl_pct: float
    timestamp: datetime


class CoinglassScraper:
    """
    Coinglass Hyperliquid Whale Tracker
    
    Scrapes:
    1. Aggregate whale positions (long/short)
    2. Individual whale trades
    3. Liquidation heatmap levels
    4. Symbol-wise L/S ratios
    """
    
    BASE_URL = "https://www.coinglass.com/hyperliquid"
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 120  # 2 minutes (whale data changes fast)
        self.last_fetch = {}
    
    async def get_whale_data(self) -> Dict:
        """
        Ana whale veri toplama fonksiyonu.
        
        Returns:
            {
                'total_long_usd': 500000000,
                'total_short_usd': 450000000,
                'long_ratio': 52.5,
                'short_ratio': 47.5,
                'net_sentiment': 'LONG_HEAVY',
                'top_whales': [...],
                'recent_activity': [...],
                'timestamp': datetime
            }
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available")
            return self._empty_result()
        
        cache_key = 'coinglass_whale'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Go to Coinglass Hyperliquid
                await page.goto(self.BASE_URL, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(3000)  # Wait for JS rendering
                
                # Extract data
                result = await self._extract_whale_data(page)
                
                await browser.close()
                
                self._set_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.error(f"Coinglass scraping error: {e}")
            return self._empty_result()
    
    async def _extract_whale_data(self, page) -> Dict:
        """Extract whale data from page."""
        result = {
            'total_long_usd': 0,
            'total_short_usd': 0,
            'long_ratio': 50,
            'short_ratio': 50,
            'net_sentiment': 'NEUTRAL',
            'top_whales': [],
            'recent_activity': [],
            'liquidation_zones': [],
            'timestamp': datetime.now()
        }
        
        try:
            # Try to get aggregate stats from page
            content = await page.content()
            
            # Look for position data in page
            # The actual selectors depend on page structure
            long_elements = await page.query_selector_all('[class*="long"]')
            short_elements = await page.query_selector_all('[class*="short"]')
            
            # Get table rows for whale positions
            rows = await page.query_selector_all('table tbody tr')
            
            for i, row in enumerate(rows[:10]):  # Top 10 whales
                try:
                    cells = await row.query_selector_all('td')
                    if len(cells) >= 5:
                        whale = {
                            'rank': i + 1,
                            'wallet': await cells[0].inner_text() if len(cells) > 0 else '',
                            'symbol': await cells[1].inner_text() if len(cells) > 1 else '',
                            'side': await cells[2].inner_text() if len(cells) > 2 else '',
                            'size': await cells[3].inner_text() if len(cells) > 3 else '',
                            'pnl': await cells[4].inner_text() if len(cells) > 4 else '',
                        }
                        result['top_whales'].append(whale)
                except:
                    continue
            
            # Determine sentiment
            if result['long_ratio'] > 55:
                result['net_sentiment'] = 'LONG_HEAVY'
            elif result['short_ratio'] > 55:
                result['net_sentiment'] = 'SHORT_HEAVY'
            else:
                result['net_sentiment'] = 'BALANCED'
            
            logger.info(f"Coinglass: {len(result['top_whales'])} whales found, sentiment: {result['net_sentiment']}")
            
        except Exception as e:
            logger.warning(f"Data extraction error: {e}")
        
        return result
    
    def get_whale_data_sync(self) -> Dict:
        """Synchronous wrapper for async function."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.get_whale_data())
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Sync whale data error: {e}")
            return self._empty_result()
    
    def get_liquidation_zones(self, symbol: str = 'BTC') -> Dict:
        """
        Likidasyon bölgelerini al.
        
        Returns:
            {
                'above_price': [88000, 89000, 90000],  # Long liq zones
                'below_price': [82000, 81000, 80000],  # Short liq zones
                'nearest_long_liq': 88000,
                'nearest_short_liq': 82000,
                'strongest_magnet': 88000  # Price likely to move here
            }
        """
        whale_data = self.get_whale_data_sync()
        
        # Extract liquidation prices from whale positions
        long_liqs = []
        short_liqs = []
        
        for whale in whale_data.get('top_whales', []):
            side = whale.get('side', '').upper()
            # Would need to parse liq_price from table
            # This is simplified - real implementation would extract actual prices
        
        return {
            'above_price': sorted(long_liqs)[:5] if long_liqs else [],
            'below_price': sorted(short_liqs, reverse=True)[:5] if short_liqs else [],
            'nearest_long_liq': long_liqs[0] if long_liqs else 0,
            'nearest_short_liq': short_liqs[0] if short_liqs else 0,
            'strongest_magnet': 0,
            'timestamp': datetime.now()
        }
    
    def get_signal_enhancement(self, current_price: float) -> Dict:
        """
        Mevcut sinyali whale verileriyle güçlendir.
        
        Returns:
            {
                'whale_bias': 'LONG' / 'SHORT' / 'NEUTRAL',
                'confidence_boost': 15,  # %15 güven artışı
                'liquidation_warning': 'SHORT_SQUEEZE_RISK',
                'smart_money_direction': 'LONG'
            }
        """
        whale_data = self.get_whale_data_sync()
        
        sentiment = whale_data.get('net_sentiment', 'NEUTRAL')
        
        # Determine bias
        if sentiment == 'LONG_HEAVY':
            whale_bias = 'LONG'
            confidence_boost = 10
        elif sentiment == 'SHORT_HEAVY':
            whale_bias = 'SHORT'
            confidence_boost = 10
        else:
            whale_bias = 'NEUTRAL'
            confidence_boost = 0
        
        # Check liquidation risk
        liq_zones = self.get_liquidation_zones()
        liq_warning = 'NONE'
        
        if liq_zones['nearest_long_liq'] and current_price > liq_zones['nearest_long_liq'] * 0.98:
            liq_warning = 'LONG_LIQ_NEAR'
        elif liq_zones['nearest_short_liq'] and current_price < liq_zones['nearest_short_liq'] * 1.02:
            liq_warning = 'SHORT_SQUEEZE_RISK'
        
        return {
            'whale_bias': whale_bias,
            'confidence_boost': confidence_boost,
            'liquidation_warning': liq_warning,
            'smart_money_direction': whale_bias,
            'whale_count': len(whale_data.get('top_whales', [])),
            'timestamp': datetime.now()
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result."""
        return {
            'total_long_usd': 0,
            'total_short_usd': 0,
            'long_ratio': 50,
            'short_ratio': 50,
            'net_sentiment': 'NEUTRAL',
            'top_whales': [],
            'recent_activity': [],
            'liquidation_zones': [],
            'timestamp': datetime.now()
        }
    
    def _is_cached(self, key: str) -> bool:
        """Check cache freshness."""
        if key not in self.cache or key not in self.last_fetch:
            return False
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Cache data."""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()


# Convenience functions
def get_whale_intel() -> Dict:
    """Quick whale intelligence."""
    scraper = CoinglassScraper()
    return scraper.get_whale_data_sync()


def get_liquidation_hunt_levels() -> Dict:
    """Get liquidation hunt target levels."""
    scraper = CoinglassScraper()
    return scraper.get_liquidation_zones()
