"""
CORRELATION CONNECTOR - Cross-Asset & Dominance Data
Fetches market correlation data for comprehensive AI analysis.

Data Sources:
- Yahoo Finance: Gold, Silver, Nasdaq, SPX, Oil
- CoinGecko: BTC Dominance, USDT Dominance
- Binance: Cross-pairs (ETH/BTC, LTC/BTC)
"""
import logging
import yfinance as yf
import requests
from typing import Dict, Optional
from datetime import datetime, timedelta
import time

logger = logging.getLogger("CORRELATION_CONNECTOR")

# Cache to reduce API calls
_cache = {}
_cache_duration = 300  # 5 minutes


class CorrelationConnector:
    """
    Fetches cross-asset correlations and market structure data.
    
    Features:
    - Precious metals (Gold, Silver)
    - Stock indices (Nasdaq, S&P 500)
    - Crypto dominance (BTC.D, USDT.D)
    - Cross-pair ratios (ETH/BTC, LTC/BTC)
    """
    
    # Yahoo Finance tickers
    CROSS_ASSETS = {
        'gold': 'GC=F',       # Gold Futures
        'silver': 'SI=F',     # Silver Futures
        'nasdaq': '^IXIC',    # Nasdaq Composite
        'spx': '^GSPC',       # S&P 500
        'oil': 'CL=F',        # Crude Oil
    }
    
    def __init__(self):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached value if still valid."""
        if key in _cache:
            data, timestamp = _cache[key]
            if time.time() - timestamp < _cache_duration:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache data with timestamp."""
        _cache[key] = (data, time.time())
    
    def fetch_cross_assets(self) -> Dict[str, float]:
        """
        Fetch cross-asset prices and changes.
        Returns: {asset_name: price, asset_name_change: % change}
        """
        cache_key = "cross_assets"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {}
        
        for name, ticker in self.CROSS_ASSETS.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period="2d")
                
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    
                    result[name] = round(current, 2)
                    result[f"{name}_change"] = round(change, 2)
                elif len(hist) == 1:
                    result[name] = round(hist['Close'].iloc[-1], 2)
                    result[f"{name}_change"] = 0.0
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                result[name] = 0.0
                result[f"{name}_change"] = 0.0
        
        self._set_cache(cache_key, result)
        logger.info(f"📊 Cross-assets fetched: Gold=${result.get('gold', 0)}, SPX={result.get('spx', 0)}")
        return result
    
    def fetch_dominance(self) -> Dict[str, float]:
        """
        Fetch BTC and USDT dominance from CoinGecko.
        """
        cache_key = "dominance"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {'btc_dominance': 0.0, 'usdt_dominance': 0.0}
        
        try:
            # CoinGecko global endpoint
            url = f"{self.coingecko_url}/global"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('data', {})
                market_cap_pct = data.get('market_cap_percentage', {})
                
                result['btc_dominance'] = round(market_cap_pct.get('btc', 0), 2)
                result['usdt_dominance'] = round(market_cap_pct.get('usdt', 0), 2)
                result['eth_dominance'] = round(market_cap_pct.get('eth', 0), 2)
                result['total_market_cap'] = data.get('total_market_cap', {}).get('usd', 0)
                
                logger.info(f"📊 Dominance: BTC={result['btc_dominance']}%, USDT={result['usdt_dominance']}%")
            else:
                logger.warning(f"CoinGecko API returned {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to fetch dominance: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    def fetch_cross_pairs(self) -> Dict[str, float]:
        """
        Fetch crypto cross-pair ratios from Binance.
        ETH/BTC, LTC/BTC ratios show altcoin strength vs BTC.
        """
        cache_key = "cross_pairs"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {}
        pairs = ['ETHBTC', 'LTCBTC']
        
        try:
            for pair in pairs:
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={pair}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get('lastPrice', 0))
                    change = float(data.get('priceChangePercent', 0))
                    
                    key = pair.lower().replace('btc', '_btc')
                    result[key] = price
                    result[f"{key}_change"] = round(change, 2)
                    
        except Exception as e:
            logger.error(f"Failed to fetch cross-pairs: {e}")
        
        if result:
            logger.info(f"📊 Cross-pairs: ETH/BTC={result.get('eth_btc', 0):.6f}")
        
        self._set_cache(cache_key, result)
        return result
    
    def fetch_all(self) -> Dict[str, float]:
        """
        Fetch all correlation data in one call.
        Returns comprehensive market structure data.
        """
        result = {}
        
        # Cross-assets (Gold, Silver, Nasdaq, SPX, Oil)
        result.update(self.fetch_cross_assets())
        
        # Dominance (BTC.D, USDT.D)
        result.update(self.fetch_dominance())
        
        # Cross-pairs (ETH/BTC, LTC/BTC)
        result.update(self.fetch_cross_pairs())
        
        result['timestamp'] = datetime.now().isoformat()
        
        return result


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = CorrelationConnector()
    data = connector.fetch_all()
    print("\n📊 All Correlation Data:")
    for key, value in data.items():
        print(f"  {key}: {value}")
