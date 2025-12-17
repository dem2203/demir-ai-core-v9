# -*- coding: utf-8 -*-
"""
DEMIR AI - Browser-Based Web Scraper (Playwright)
İnsan gibi web sitelerini okuyarak gerçek değerleri alır.

NO MOCK DATA - Gerçek browser ile gerçek data!
"""
import logging
from datetime import datetime
from typing import Dict, Optional
import asyncio

logger = logging.getLogger("BROWSER_SCRAPER")

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed - browser scraping disabled")


class BrowserScraper:
    """
    Browser-based web scraper using Playwright.
    
    Opens real browser (headless) and reads values like a human.
    Works with JavaScript-heavy sites like TradingView.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        self.last_fetch = {}
    
    async def get_tradingview_data(self, symbol: str) -> Dict:
        """
        TradingView'dan gerçek veri al - insan gibi!
        
        Args:
            symbol: 'BTC.D', 'ETH.D', 'GOLD', etc.
        
        Returns:
            {'symbol': 'BTC.D', 'price': 59.63, 'change': 0.35}
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available")
            return self._empty_result(symbol)
        
        # Check cache
        cache_key = f'tv_{symbol}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            async with async_playwright() as p:
                # Launch headless browser
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to TradingView
                url = self._get_tradingview_url(symbol)
                await page.goto(url, timeout=30000)
                
                # Wait for page to fully load
                await page.wait_for_timeout(3000)
                
                # Extract price - different selectors based on symbol type
                price = await self._extract_price(page, symbol)
                change = await self._extract_change(page, symbol)
                
                await browser.close()
                
                if price and price > 0:
                    change_val = change if change is not None else 0
                    result = {
                        'symbol': symbol,
                        'price': price,
                        'change': change_val,
                        'timestamp': datetime.now()
                    }
                    self._set_cache(cache_key, result)
                    logger.info(f"Browser scraped {symbol}: {price} ({change_val:+.2f}%)")
                    return result
                
                return self._empty_result(symbol)
                
        except Exception as e:
            logger.error(f"Browser scraping error for {symbol}: {e}")
            return self._empty_result(symbol)
    
    def get_tradingview_data_sync(self, symbol: str) -> Dict:
        """Synchronous wrapper for async method."""
        try:
            # Create new event loop if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(self.get_tradingview_data(symbol))
        except Exception as e:
            logger.error(f"Sync wrapper error: {e}")
            return self._empty_result(symbol)
    
    async def get_all_dominance_data(self) -> Dict:
        """Get all dominance metrics from TradingView."""
        results = {}
        
        symbols = ['BTC.D', 'ETH.D', 'USDT.D', 'USDC.D']
        
        for symbol in symbols:
            data = await self.get_tradingview_data(symbol)
            key = symbol.lower().replace('.', '_')
            results[key] = data.get('price', 0)
            results[f'{key}_change'] = data.get('change', 0)
        
        results['timestamp'] = datetime.now()
        return results
    
    async def get_macro_data(self) -> Dict:
        """Get macro indicators like Gold, DXY, VIX."""
        results = {}
        
        symbols = {
            'gold': 'GOLD',
            'nasdaq': 'IXIC',
            'dxy': 'DXY',
            'vix': 'VIX',
            'spy': 'SPY'
        }
        
        for key, symbol in symbols.items():
            data = await self.get_tradingview_data(symbol)
            results[key] = data.get('price', 0)
            results[f'{key}_change'] = data.get('change', 0)
        
        results['timestamp'] = datetime.now()
        return results
    
    def _get_tradingview_url(self, symbol: str) -> str:
        """Get TradingView URL for symbol."""
        symbol_map = {
            'BTC.D': 'CRYPTOCAP-BTC.D',
            'ETH.D': 'CRYPTOCAP-ETH.D',
            'USDT.D': 'CRYPTOCAP-USDT.D',
            'USDC.D': 'CRYPTOCAP-USDC.D',
            'GOLD': 'TVC-GOLD',
            'DXY': 'TVC-DXY',
            'VIX': 'TVC-VIX',
            'SPY': 'AMEX-SPY',
            'IXIC': 'NASDAQ-IXIC',
        }
        
        tv_symbol = symbol_map.get(symbol, symbol)
        return f"https://www.tradingview.com/symbols/{tv_symbol}/"
    
    async def _extract_price(self, page, symbol: str) -> Optional[float]:
        """Extract price from TradingView page."""
        try:
            # Multiple selectors to try
            selectors = [
                '.tv-symbol-price-quote__value',
                '[data-symbol-price]',
                '.js-symbol-last',
                '.last-price',
            ]
            
            for selector in selectors:
                try:
                    element = page.locator(selector).first
                    if await element.count() > 0:
                        text = await element.text_content()
                        # Clean the text and parse as float
                        clean = text.replace(',', '').replace('%', '').strip()
                        return float(clean)
                except:
                    continue
            
            # Fallback: extract from page title
            title = await page.title()
            # Pattern: "BTC.D 59.63 +0.35 (0.60%) — TradingView"
            import re
            match = re.search(r'[\d.]+', title)
            if match:
                return float(match.group())
            
            return None
            
        except Exception as e:
            logger.debug(f"Price extraction failed: {e}")
            return None
    
    async def _extract_change(self, page, symbol: str) -> Optional[float]:
        """Extract change % from TradingView page."""
        try:
            # Try title first (most reliable)
            title = await page.title()
            import re
            # Pattern: "+0.35 (0.60%)" or "-0.35 (-0.60%)"
            match = re.search(r'\(([+-]?[\d.]+)%\)', title)
            if match:
                return float(match.group(1))
            
            return None
            
        except Exception as e:
            logger.debug(f"Change extraction failed: {e}")
            return None
    
    def _empty_result(self, symbol: str) -> Dict:
        """Return empty result - NO MOCK DATA!"""
        return {
            'symbol': symbol,
            'price': 0,
            'change': 0,
            'timestamp': datetime.now()
        }
    
    def _is_cached(self, key: str) -> bool:
        """Check cache freshness."""
        if key not in self.cache or key not in self.last_fetch:
            return False
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Set cache."""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()


# Convenience function for synchronous use
def scrape_tradingview(symbol: str) -> Dict:
    """
    Simple function to scrape TradingView.
    
    Usage:
        data = scrape_tradingview('BTC.D')
        print(data['price'])  # 59.63
    """
    scraper = BrowserScraper()
    return scraper.get_tradingview_data_sync(symbol)
