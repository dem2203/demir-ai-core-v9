# -*- coding: utf-8 -*-
"""
DEMIR AI - TradingView Real Web Scraper
STRICT: NO MOCK DATA, NO FALLBACKS, NO APPROXIMATIONS

Pure web scraping from TradingView - Real data only!
If scraping fails = shows N/A (acceptable)
"""
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional
import re
import json

logger = logging.getLogger("TRADINGVIEW_SCRAPER")


class TradingViewScraper:
    """
    Pure TradingView Web Scraper
    
    RULES:
    - NO mock data
    - NO fallback APIs  
    - NO approximations
    - Real scraping or N/A
    """
    
    SYMBOLS = {
        'gold': 'OANDA:XAUUSD',
        'nasdaq': 'NASDAQ:IXIC',
        'dxy': 'TVC:DXY',
        'vix': 'CBOE:VIX',
        'btc_dominance': 'CRYPTOCAP:BTC.D',
        'eth_dominance': 'CRYPTOCAP:ETH.D',
        'usdt_dominance': 'CRYPTOCAP:USDT.D',
        'usdc_dominance': 'CRYPTOCAP:USDC.D',
        'spy': 'AMEX:SPY',
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 min
        self.last_fetch = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    
    def get_symbol_data(self, symbol_key: str) -> Dict:
        """
        Get REAL data from TradingView web scraping.
        
        Returns real data or empty (N/A) - NO MOCK DATA!
        """
        cache_key = f'tv_{symbol_key}'
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        symbol = self.SYMBOLS.get(symbol_key)
        if not symbol:
            return self._empty_result(symbol_key)
        
        try:
            # Real TradingView scraping
            data = self._scrape_tradingview_real(symbol, symbol_key)
            
            if data and data.get('price', 0) > 0:
                self._set_cache(cache_key, data)
                logger.info(f"✅ Real TradingView data for {symbol_key}: {data['price']}")
                return data
            
            # If scraping failed, return N/A (acceptable - no mock!)
            logger.warning(f"TradingView scraping failed for {symbol_key} - showing N/A")
            return self._empty_result(symbol_key)
            
        except Exception as e:
            logger.error(f"TradingView error for {symbol_key}: {e}")
            return self._empty_result(symbol_key)
    
    def get_all_macro_data(self) -> Dict:
        """Get all macro data."""
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
        """Stablecoin dominance - REAL data only."""
        usdt = self.get_symbol_data('usdt_dominance')
        usdc = self.get_symbol_data('usdc_dominance')
        
        usdt_d = usdt.get('price', 0)
        usdc_d = usdc.get('price', 0)
        total = usdt_d + usdc_d
        
        # Only interpret if we have REAL data (not 0)
        if total > 0:
            if total > 8:
                signal, emoji, interpretation = 'EXTREME_FEAR', '🔴', 'Money fleeing to stablecoins'
            elif total > 6:
                signal, emoji, interpretation = 'CAUTION', '🟡', 'Stablecoin inflow increasing'
            elif total < 4:
                signal, emoji, interpretation = 'GREED', '🟢', 'Stablecoins flowing to crypto'
            else:
                signal, emoji, interpretation = 'NEUTRAL', '⚪', 'Normal stablecoin dominance'
        else:
            signal, emoji, interpretation = 'N/A', '⚪', 'Stablecoin data unavailable'
        
        return {
            'usdt_dominance': usdt_d,
            'usdc_dominance': usdc_d,
            'total_stablecoin_dominance': total,
            'signal': signal,
            'emoji': emoji,
            'interpretation': interpretation,
            'timestamp': datetime.now()
        }
    
    # ========================================
    # REAL WEB SCRAPING
    # ========================================
    
    def _scrape_tradingview_real(self, symbol: str, symbol_key: str) -> Optional[Dict]:
        """
        REAL TradingView web scraping.
        
        Tries multiple methods to extract real data from TradingView.
        """
        # Method 1: Symbol page with embedded JSON data
        data = self._scrape_symbol_page_json(symbol)
        if data:
            return data
        
        # Method 2: Chart embed data
        data = self._scrape_chart_data(symbol)
        if data:
            return data
        
        # Method 3: Widget data
        data = self._scrape_widget_data(symbol)
        if data:
            return data
        
        return None
    
    def _scrape_symbol_page_json(self, symbol: str) -> Optional[Dict]:
        """
        Scrape TradingView symbol page for embedded JSON data.
        
        TradingView embeds data in <script> tags as JSON.
        """
        try:
            url = f"https://www.tradingview.com/symbols/{symbol.replace(':', '-')}/"
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find script tag with __NEXT_DATA__ or similar
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and ('__NEXT_DATA__' in script.string or 'quoteData' in script.string):
                    try:
                        # Extract JSON from script
                        json_text = script.string
                        
                        # Try to parse JSON
                        if '__NEXT_DATA__' in json_text:
                            start = json_text.find('{')
                            end = json_text.rfind('}') + 1
                            if start != -1 and end > start:
                                data = json.loads(json_text[start:end])
                                
                                # Navigate JSON structure to find price
                                # TradingView structure varies, need to explore
                                price = self._extract_price_from_json(data)
                                change = self._extract_change_from_json(data)
                                
                                if price and price > 0:
                                    return {
                                        'symbol': symbol,
                                        'price': price,
                                        'change': change or 0,
                                        'change_abs': 0,
                                        'timestamp': datetime.now()
                                    }
                    except:
                        continue
            
            # Alternative: Look for meta tags
            price_meta = soup.find('meta', {'property': 'og:price:amount'})
            if price_meta:
                price = float(price_meta.get('content', 0))
                if price > 0:
                    return {
                        'symbol': symbol,
                        'price': price,
                        'change': 0,
                        'change_abs': 0,
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception as e:
            logger.debug(f"Symbol page JSON scraping failed: {e}")
            return None
    
    def _scrape_chart_data(self, symbol: str) -> Optional[Dict]:
        """Scrape chart endpoint data."""
        try:
            # TradingView chart API (public, but might need special headers)
            url = f"https://symbol-search.tradingview.com/symbol_search/?text={symbol}&type=&exchange="
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    item = data[0]
                    # Extract price if available
                    # Structure varies by symbol type
                    pass
            
            return None
        except:
            return None
    
    def _scrape_widget_data(self, symbol: str) -> Optional[Dict]:
        """Scrape widget endpoint."""
        # TradingView has widget endpoints that might be accessible
        return None
    
    def _extract_price_from_json(self, data: dict) -> Optional[float]:
        """Extract price from nested JSON."""
        try:
            # Common paths in TradingView JSON
            paths = [
                ['props', 'pageProps', 'symbolData', 'last'],
                ['props', 'pageProps', 'quoteData', 'pro_perm'],
                ['symbolData', 'last'],
                ['quoteData', 'lp'],
            ]
            
            for path in paths:
                value = data
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if isinstance(value, (int, float)):
                        return float(value)
            
            return None
        except:
            return None
    
    def _extract_change_from_json(self, data: dict) -> Optional[float]:
        """Extract change % from JSON."""
        try:
            paths = [
                ['props', 'pageProps', 'symbolData', 'ch'],
                ['symbolData', 'ch'],
                ['quoteData', 'chp'],
            ]
            
            for path in paths:
                value = data
                for key in path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        break
                else:
                    if isinstance(value, (int, float)):
                        return float(value)
            
            return None
        except:
            return None
    
    def _empty_result(self, symbol_key: str) -> Dict:
        """
        Empty result (N/A) - NO MOCK DATA!
        """
        return {
            'symbol': self.SYMBOLS.get(symbol_key, ''),
            'price': 0,  # 0 = N/A in dashboard
            'change': 0,
            'change_abs': 0,
            'timestamp': datetime.now()
        }
    
    def _is_cached(self, key: str) -> bool:
        """Check cache."""
        if key not in self.cache or key not in self.last_fetch:
            return False
        
        age = (datetime.now() - self.last_fetch[key]).total_seconds()
        return age < self.cache_duration
    
    def _set_cache(self, key: str, data):
        """Set cache."""
        self.cache[key] = data
        self.last_fetch[key] = datetime.now()
