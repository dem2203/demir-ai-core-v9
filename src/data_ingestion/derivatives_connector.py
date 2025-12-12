"""
DERIVATIVES CONNECTOR - Futures & On-Chain Data
Fetches critical derivatives and on-chain metrics for comprehensive AI analysis.

Data Sources:
- Binance Futures: Open Interest, Long/Short Ratio, Funding Rate
- CoinGlass: Aggregated derivatives data
"""
import logging
import requests
from typing import Dict, Optional
from datetime import datetime
import time

logger = logging.getLogger("DERIVATIVES_CONNECTOR")

# Cache to reduce API calls
_cache = {}
_cache_duration = 60  # 1 minute (derivatives data changes frequently)


class DerivativesConnector:
    """
    Fetches derivatives market data critical for crypto trading.
    
    Features:
    - Open Interest (OI) - Total futures positions
    - Long/Short Ratio - Market sentiment
    - Aggregated Funding Rate
    - Liquidation data
    """
    
    BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1"
    
    def __init__(self):
        pass
        
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
    
    def fetch_open_interest(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch Open Interest for a symbol.
        High OI = High leverage = Potential volatility
        """
        cache_key = f"oi_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {'open_interest': 0.0, 'open_interest_value': 0.0}
        
        try:
            url = f"{self.BINANCE_FUTURES_URL}/openInterest?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                result['open_interest'] = float(data.get('openInterest', 0))
                result['open_interest_value'] = result['open_interest']  # In contracts
                
        except Exception as e:
            logger.warning(f"Failed to fetch OI for {symbol}: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    def fetch_long_short_ratio(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch Long/Short Ratio (Top Trader Accounts).
        > 1 = More longs (bullish sentiment)
        < 1 = More shorts (bearish sentiment)
        """
        cache_key = f"ls_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {
            'long_short_ratio': 1.0,
            'long_account_pct': 50.0,
            'short_account_pct': 50.0
        }
        
        try:
            # Top Trader Long/Short Ratio (Accounts)
            url = f"{self.BINANCE_FUTURES_URL}/topLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    latest = data[0]
                    result['long_short_ratio'] = float(latest.get('longShortRatio', 1.0))
                    result['long_account_pct'] = float(latest.get('longAccount', 0.5)) * 100
                    result['short_account_pct'] = float(latest.get('shortAccount', 0.5)) * 100
                    
        except Exception as e:
            logger.warning(f"Failed to fetch L/S ratio for {symbol}: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    def fetch_funding_rate(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch current and predicted funding rate.
        Positive = Longs pay shorts (bullish pressure)
        Negative = Shorts pay longs (bearish pressure)
        """
        cache_key = f"fr_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        result = {'funding_rate': 0.0, 'funding_rate_pct': 0.0}
        
        try:
            url = f"{self.BINANCE_FUTURES_URL}/premiumIndex?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                fr = float(data.get('lastFundingRate', 0))
                result['funding_rate'] = fr
                result['funding_rate_pct'] = fr * 100  # Convert to percentage
                
        except Exception as e:
            logger.warning(f"Failed to fetch funding rate for {symbol}: {e}")
        
        self._set_cache(cache_key, result)
        return result
    
    def fetch_liquidations_estimate(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Estimate liquidation levels based on current price and leverage.
        Uses mark price and typical leverage levels.
        """
        result = {
            'estimated_liq_long': 0.0,
            'estimated_liq_short': 0.0,
            'liquidation_risk': 'MEDIUM'
        }
        
        try:
            # Get mark price
            url = f"{self.BINANCE_FUTURES_URL}/premiumIndex?symbol={symbol}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                mark_price = float(data.get('markPrice', 0))
                
                if mark_price > 0:
                    # Typical liquidation estimates at 10x leverage
                    result['estimated_liq_long'] = mark_price * 0.90  # -10%
                    result['estimated_liq_short'] = mark_price * 1.10  # +10%
                    
                    # Get funding rate to assess risk
                    fr = float(data.get('lastFundingRate', 0))
                    if abs(fr) > 0.001:  # >0.1% funding
                        result['liquidation_risk'] = 'HIGH'
                    elif abs(fr) < 0.0001:
                        result['liquidation_risk'] = 'LOW'
                        
        except Exception as e:
            logger.warning(f"Failed to estimate liquidations: {e}")
        
        return result
    
    def fetch_all_derivatives(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch all derivatives data for a symbol.
        Returns comprehensive derivatives metrics.
        """
        result = {}
        
        # Open Interest
        oi_data = self.fetch_open_interest(symbol)
        result.update(oi_data)
        
        # Long/Short Ratio
        ls_data = self.fetch_long_short_ratio(symbol)
        result.update(ls_data)
        
        # Funding Rate
        fr_data = self.fetch_funding_rate(symbol)
        result.update(fr_data)
        
        # Liquidation estimates
        liq_data = self.fetch_liquidations_estimate(symbol)
        result.update(liq_data)
        
        result['derivatives_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"📊 Derivatives {symbol}: OI={result.get('open_interest', 0):.0f}, L/S={result.get('long_short_ratio', 1):.2f}, FR={result.get('funding_rate_pct', 0):.4f}%")
        
        return result
    
    def fetch_multi_symbol(self, symbols: list = None) -> Dict[str, Dict]:
        """
        Fetch derivatives data for multiple symbols.
        """
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
        
        result = {}
        for symbol in symbols:
            result[symbol] = self.fetch_all_derivatives(symbol)
            time.sleep(0.1)  # Rate limit protection
        
        return result


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    connector = DerivativesConnector()
    
    print("\n📊 BTC Derivatives Data:")
    data = connector.fetch_all_derivatives("BTCUSDT")
    for key, value in data.items():
        print(f"  {key}: {value}")
    
    print("\n📊 ETH Derivatives Data:")
    data = connector.fetch_all_derivatives("ETHUSDT")
    for key, value in data.items():
        print(f"  {key}: {value}")
