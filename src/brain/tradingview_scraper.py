# -*- coding: utf-8 -*-
"""
DEMIR AI - TradingView Scraper (Playwright Browser Edition)
Gerçek browser ile web sitelerinden insan gibi veri okur!

PHASE 43 UPDATED: Playwright Browser Scraping
- Gerçek Chromium browser açar
- JavaScript render bekler
- Değerleri insan gibi okur
- NO MOCK DATA!
"""
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("TRADINGVIEW_SCRAPER")

# Import browser scraper
try:
    from src.brain.browser_scraper import BrowserScraper, scrape_tradingview
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False
    logger.warning("Browser scraper not available - falling back to HTTP")


class TradingViewScraper:
    """
    TradingView Real-Time Data Scraper using Playwright Browser.
    
    Opens real browser and reads values like a human!
    Works with JavaScript-heavy sites.
    """
    
    # Symbol mappings for browser scraper
    BROWSER_SYMBOLS = {
        'gold': 'GOLD',
        'nasdaq': 'IXIC',
        'dxy': 'DXY',
        'vix': 'VIX',
        'btc_dominance': 'BTC.D',
        'eth_dominance': 'ETH.D',
        'usdt_dominance': 'USDT.D',
        'usdc_dominance': 'USDC.D',
        'spy': 'SPY',
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch = {}
        
        # Initialize browser scraper if available
        if BROWSER_AVAILABLE:
            self.browser_scraper = BrowserScraper()
            logger.info("TradingView Scraper initialized with Playwright browser")
        else:
            self.browser_scraper = None
            logger.warning("Playwright not available - limited functionality")
    
    def get_symbol_data(self, symbol_key: str) -> Dict:
        """
        Get real-time data using Playwright browser.
        Opens real browser and reads TradingView like a human!
        """
        cache_key = f'tv_{symbol_key}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        browser_symbol = self.BROWSER_SYMBOLS.get(symbol_key)
        if not browser_symbol:
            logger.warning(f"Unknown symbol key: {symbol_key}")
            return self._empty_result(symbol_key)
        
        try:
            if BROWSER_AVAILABLE:
                # Use Playwright browser - reads like a human!
                data = self.browser_scraper.get_tradingview_data_sync(browser_symbol)
                
                if data and data.get('price', 0) > 0:
                    result = {
                        'symbol': browser_symbol,
                        'price': data['price'],
                        'change': data.get('change', 0),
                        'change_abs': 0,
                        'timestamp': datetime.now()
                    }
                    self._set_cache(cache_key, result)
                    logger.info(f"Browser read {symbol_key}: {data['price']}")
                    return result
            
            logger.warning(f"Browser scraping failed for {symbol_key}")
            return self._empty_result(symbol_key)
            
        except Exception as e:
            logger.error(f"TradingView browser error for {symbol_key}: {e}")
            return self._empty_result(symbol_key)
    
    def get_all_macro_data(self) -> Dict:
        """Get all macro data using browser."""
        logger.info("Fetching all macro data via browser...")
        
        return {
            'gold': self.get_symbol_data('gold'),
            'nasdaq': self.get_symbol_data('nasdaq'),
            'dxy': self.get_symbol_data('dxy'),
            'vix': self.get_symbol_data('vix'),
            'btc_dominance': self.get_symbol_data('btc_dominance'),
            'eth_dominance': self.get_symbol_data('eth_dominance'),
            'usdt_dominance': self.get_symbol_data('usdt_dominance'),
            'usdc_dominance': self.get_symbol_data('usdc_dominance'),
            'spy': self.get_symbol_data('spy'),
            'timestamp': datetime.now()
        }
    
    def get_stablecoin_summary(self) -> Dict:
        """Stablecoin dominance analysis."""
        usdt = self.get_symbol_data('usdt_dominance')
        usdc = self.get_symbol_data('usdc_dominance')
        
        usdt_d = usdt.get('price', 0)
        usdc_d = usdc.get('price', 0)
        total = usdt_d + usdc_d
        
        # Interpretation based on real data
        if total > 0:
            if total > 8:
                signal = 'EXTREME_FEAR'
                emoji = '🔴'
                interpretation = 'Money fleeing to stablecoins (extreme fear)'
            elif total > 6:
                signal = 'CAUTION'
                emoji = '🟡'
                interpretation = 'Stablecoin inflow increasing (mild fear)'
            elif total < 4:
                signal = 'GREED'
                emoji = '🟢'
                interpretation = 'Stablecoins flowing back to crypto (greed)'
            else:
                signal = 'NEUTRAL'
                emoji = '⚪'
                interpretation = 'Stablecoin dominance normal'
        else:
            signal = 'N/A'
            emoji = '⚪'
            interpretation = 'Stablecoin data not available'
        
        return {
            'usdt_dominance': usdt_d,
            'usdc_dominance': usdc_d,
            'total_stablecoin_dominance': total,
            'signal': signal,
            'emoji': emoji,
            'interpretation': interpretation,
            'timestamp': datetime.now()
        }
    
    def _empty_result(self, symbol_key: str) -> Dict:
        """Return empty result - NO MOCK DATA!"""
        return {
            'symbol': self.BROWSER_SYMBOLS.get(symbol_key, ''),
            'price': 0,  # 0 = N/A
            'change': 0,
            'change_abs': 0,
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
