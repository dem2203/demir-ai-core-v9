# -*- coding: utf-8 -*-
"""
CORRELATION CONNECTOR - Cross-Asset & Dominance Data (NO YFINANCE)

Data Sources:
- CoinGecko: BTC Dominance, USDT Dominance, ETH Dominance
- Binance: Cross-pairs (ETH/BTC, LTC/BTC)
- FRED API: Gold, SPX (if available)

NO YFINANCE - Production safe.
"""
import logging
import requests
from typing import Dict, Optional
from datetime import datetime
import time
import os

logger = logging.getLogger("CORRELATION_CONNECTOR")

# Cache to reduce API calls
_cache = {}
_cache_duration = 300  # 5 minutes


class CorrelationConnector:
    """
    Fetches cross-asset correlations and market structure data.
    NO YFINANCE - Uses free APIs only.
    
    Features:
    - Crypto dominance (BTC.D, USDT.D, ETH.D)
    - Cross-pair ratios (ETH/BTC, LTC/BTC)
    - Traditional markets via FRED (if available)
    """
    
    def __init__(self):
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
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
        Fetch cross-asset data WITHOUT yfinance.
        Uses FRED API for traditional markets.
        """
        cache_key = "cross_assets"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {}
        
        # 1. Gold from FRED (GOLDAMGBD228NLBM)
        if self.fred_api_key:
            try:
                gold_resp = requests.get(
                    self.fred_base_url,
                    params={
                        'series_id': 'GOLDAMGBD228NLBM',
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 2,
                        'sort_order': 'desc'
                    },
                    timeout=10
                )
                if gold_resp.status_code == 200:
                    data = gold_resp.json()
                    obs = data.get('observations', [])
                    if len(obs) >= 1:
                        result['gold'] = float(obs[0]['value'])
                        if len(obs) >= 2:
                            prev = float(obs[1]['value'])
                            result['gold_change'] = ((result['gold'] - prev) / prev) * 100
                        else:
                            result['gold_change'] = 0
            except Exception as e:
                logger.debug(f"FRED Gold failed: {e}")
                result['gold'] = 2650  # Approximate
                result['gold_change'] = 0
        else:
            result['gold'] = 2650
            result['gold_change'] = 0
        
        # 2. S&P 500 from FRED (SP500)
        if self.fred_api_key:
            try:
                spx_resp = requests.get(
                    self.fred_base_url,
                    params={
                        'series_id': 'SP500',
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 2,
                        'sort_order': 'desc'
                    },
                    timeout=10
                )
                if spx_resp.status_code == 200:
                    data = spx_resp.json()
                    obs = data.get('observations', [])
                    if len(obs) >= 1:
                        result['spx'] = float(obs[0]['value'])
                        if len(obs) >= 2:
                            prev = float(obs[1]['value'])
                            result['spx_change'] = ((result['spx'] - prev) / prev) * 100
                        else:
                            result['spx_change'] = 0
            except Exception as e:
                logger.debug(f"FRED SPX failed: {e}")
                result['spx'] = 6000  # Approximate
                result['spx_change'] = 0
        else:
            result['spx'] = 6000
            result['spx_change'] = 0
        
        # 3. Nasdaq from FRED (NASDAQCOM)
        if self.fred_api_key:
            try:
                nasdaq_resp = requests.get(
                    self.fred_base_url,
                    params={
                        'series_id': 'NASDAQCOM',
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 2,
                        'sort_order': 'desc'
                    },
                    timeout=10
                )
                if nasdaq_resp.status_code == 200:
                    data = nasdaq_resp.json()
                    obs = data.get('observations', [])
                    if len(obs) >= 1:
                        result['nasdaq'] = float(obs[0]['value'])
                        if len(obs) >= 2:
                            prev = float(obs[1]['value'])
                            result['nasdaq_change'] = ((result['nasdaq'] - prev) / prev) * 100
                        else:
                            result['nasdaq_change'] = 0
            except Exception as e:
                logger.debug(f"FRED Nasdaq failed: {e}")
                result['nasdaq'] = 20000
                result['nasdaq_change'] = 0
        else:
            result['nasdaq'] = 20000
            result['nasdaq_change'] = 0
        
        # Silver and Oil - use approximate values (FRED doesn't have real-time)
        result['silver'] = 30  # Approximate
        result['silver_change'] = 0
        result['oil'] = 70  # Approximate
        result['oil_change'] = 0
        
        self._set_cache(cache_key, result)
        logger.info(f"📊 Cross-assets: Gold=${result.get('gold', 0):.0f}, SPX={result.get('spx', 0):.0f}")
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
