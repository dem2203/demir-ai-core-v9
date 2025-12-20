# -*- coding: utf-8 -*-
"""
DEMIR AI - Macro Connector (yfinance-free version)
FRED API + CoinGecko + Binance for all market data.
NO YFINANCE - Production safe.
"""
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import os

logger = logging.getLogger("MACRO_CONNECTOR")

class MacroConnector:
    """
    MACRO ECONOMIC INTELLIGENCE - NO YFINANCE
    
    Data Sources:
    - FRED API: Interest rates, CPI, unemployment, VIX, DXY
    - CoinGecko: BTC dominance, market cap
    - Binance: Crypto prices
    """
    
    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        self.cache_duration = 3600 * 24  # 24 hours (FRED data)
        
        if not self.api_key:
            logger.warning("⚠️ FRED_API_KEY not found. Using limited macro data.")
            
    def fetch_data(self) -> Dict[str, float]:
        """Get latest macro indicators - NO YFINANCE"""
        
        result = {}
        errors = []
        
        # 1. FRED Data (Economic indicators)
        fred_indicators = {
            "interest_rate": "FEDFUNDS",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "m2_money_supply": "M2SL",
            "vix": "VIXCLS",       # VIX from FRED
            "dxy": "DTWEXBGS"      # Dollar Index from FRED
        }
        
        for name, series_id in fred_indicators.items():
            value, error = self._get_series_latest(series_id)
            if value is not None:
                result[name] = value
            elif error:
                errors.append(f"{name}: {error}")
        
        # 2. CoinGecko - Crypto market data (free, no limit issues)
        try:
            cg_response = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            )
            if cg_response.status_code == 200:
                global_data = cg_response.json().get('data', {})
                
                result['btc_dominance'] = global_data.get('market_cap_percentage', {}).get('btc', 50)
                result['btc_dominance_change'] = global_data.get('market_cap_change_percentage_24h_usd', 0)
                result['total_market_cap'] = global_data.get('total_market_cap', {}).get('usd', 0)
                
                logger.debug(f"CoinGecko: BTC.D={result['btc_dominance']:.1f}%")
        except Exception as e:
            logger.warning(f"CoinGecko failed: {e}")
            result['btc_dominance'] = 50
            result['btc_dominance_change'] = 0
        
        # 3. Binance - Crypto prices and ETH/BTC ratio
        try:
            # BTC price
            btc_resp = requests.get(
                "https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': 'BTCUSDT'},
                timeout=5
            )
            if btc_resp.status_code == 200:
                btc_data = btc_resp.json()
                btc_price = float(btc_data.get('lastPrice', 100000))
                result['btc_price'] = btc_price
            else:
                btc_price = 100000
            
            # ETH/BTC ratio
            ethbtc_resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': 'ETHBTC'},
                timeout=5
            )
            if ethbtc_resp.status_code == 200:
                result['eth_btc_ratio'] = float(ethbtc_resp.json().get('price', 0.05))
                
        except Exception as e:
            logger.warning(f"Binance price fetch failed: {e}")
            btc_price = 100000
            result['eth_btc_ratio'] = 0.05
        
        # 4. Gold & Traditional Markets - Use FRED if available, otherwise estimate
        # FRED has gold series: GOLDAMGBD228NLBM (London Gold Fixing)
        gold_value, _ = self._get_series_latest("GOLDAMGBD228NLBM")
        if gold_value:
            result['gold'] = gold_value
            result['gold_change'] = 0  # FRED doesn't give 24h change easily
            result['gold_btc_ratio'] = gold_value / btc_price if btc_price > 0 else 0.02
        else:
            # Fallback - approximate gold price
            result['gold'] = 2650  # Approximate current gold price
            result['gold_change'] = 0
            result['gold_btc_ratio'] = 2650 / btc_price if btc_price > 0 else 0.02
        
        # 5. Nasdaq - Use FRED NASDAQCOM series
        nasdaq_value, _ = self._get_series_latest("NASDAQCOM")
        if nasdaq_value:
            result['nasdaq'] = nasdaq_value
            result['nasdaq_change'] = 0
        else:
            result['nasdaq'] = 20000  # Approximate
            result['nasdaq_change'] = 0
        
        # 6. S&P 500 proxy - Use FRED SP500 series
        sp500_value, _ = self._get_series_latest("SP500")
        if sp500_value:
            result['sp500_btc_ratio'] = sp500_value / (btc_price / 1000) if btc_price > 0 else 5.0
        else:
            result['sp500_btc_ratio'] = 5.0
        
        # Store errors in result for debug
        if errors:
            result['debug_errors'] = "; ".join(errors)

        # Calculate Macro Trend Score (-100 to 100)
        score = 0
        rate = result.get("interest_rate", 5.0)
        
        if rate < 3.0: score += 30
        elif rate > 5.0: score -= 30
        
        dxy_val = result.get('dxy')
        if dxy_val:
            if dxy_val > 115: score -= 20  # Strong Dollar = Bad for Crypto
            elif dxy_val < 100: score += 20
        
        vix_val = result.get('vix')
        if vix_val:
            if vix_val > 30: score -= 15  # High fear
            elif vix_val < 15: score += 10  # Low fear
        
        result["macro_score"] = score
        result["timestamp"] = datetime.now().isoformat()
        
        # Log summary
        rate_str = f"{rate:.2f}%" if result.get("interest_rate") else "N/A"
        dxy_str = f"{result.get('dxy', 0):.2f}" if result.get('dxy') else "N/A"
        logger.info(f"🌍 Macro: Rate={rate_str} | DXY={dxy_str} | BTC.D={result.get('btc_dominance', 0):.1f}% | Score={score}")
        
        return result
        
    def _get_series_latest(self, series_id: str) -> tuple[Optional[float], Optional[str]]:
        """Fetch single series from FRED. Returns (value, error_msg)"""
        if not self.api_key:
            return None, "FRED API Key missing"
            
        # Check cache
        if series_id in self.cache:
            ts, val = self.cache[series_id]
            if (datetime.now() - ts).total_seconds() < self.cache_duration:
                return val, None
                
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": 1,
                "sort_order": "desc"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "observations" in data and len(data["observations"]) > 0:
                value = float(data["observations"][0]["value"])
                self.cache[series_id] = (datetime.now(), value)
                return value, None
            else:
                return None, "No observations found"
                
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return None, str(e)
    
    async def fetch_macro_data(self, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """
        Async wrapper for fetch_data() returning DataFrame format.
        """
        # Get current macro snapshot
        data = self.fetch_data()
        
        if not data:
            logger.warning("Failed to fetch macro data")
            return pd.DataFrame()
        
        # Convert to DataFrame format expected by FeatureEngineer
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'macro_score': [data.get('macro_score', 0)],
            'interest_rate': [data.get('interest_rate', 0)],
            'cpi': [data.get('cpi', 0)],
            'unemployment': [data.get('unemployment', 0)],
            'm2_money_supply': [data.get('m2_money_supply', 0)],
            'macro_DXY': [data.get('dxy', 100.0)],
            'macro_VIX': [data.get('vix', 20.0)],
            'macro_debug': [data.get('debug_errors', 'OK')]
        })
        
        logger.info(f"📊 Macro data fetched: Score={data.get('macro_score', 0):.1f}")
        return df
