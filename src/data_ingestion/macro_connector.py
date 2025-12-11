import logging
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from src.config.settings import Config

logger = logging.getLogger("MACRO_CONNECTOR")

class MacroConnector:
    """
    MACRO ECONOMIC INTELLIGENCE
    Fetches key economic indicators from FRED and Market Data from Yahoo Finance.
    
    Indicators:
    1. US Interest Rate (FEDFUNDS)
    2. Inflation (CPIAUCSL)
    3. Unemployment Rate (UNRATE)
    4. M2 Money Supply (M2SL)
    5. DXY Index (DX-Y.NYB)
    6. VIX Index (^VIX)
    """
    
    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        self.cache_duration = 3600 * 24  # 24 hours (FRED data)
        self.market_cache_duration = 300 # 5 minutes (Market data)
        
        if not self.api_key:
            logger.warning("⚠️ FRED_API_KEY not found. Macro analysis will use fallback values.")
            
    def fetch_data(self) -> Dict[str, float]:
        """Get latest macro indicators"""
        
        # 1. FRED Data (Economic + Market Proxy)
        indicators = {
            "interest_rate": "FEDFUNDS",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "m2_money_supply": "M2SL",
            "vix": "VIXCLS",       # CBOE Volatility Index from FRED
            "dxy": "DTWEXBGS"      # Nominal Broad U.S. Dollar Index (Proxy for DXY)
        }
        
        result = {}
        
        for name, series_id in indicators.items():
            value = self._get_series_latest(series_id)
            if value is not None:
                result[name] = value

        # 2. Market Data Fallback (yfinance) - Only if FRED fails or for specific real-time checks
        # Railway often blocks yfinance, so we rely on FRED first.
        if "dxy" not in result or "vix" not in result:
            try:
                # DXY
                if "dxy" not in result:
                    dxy_ticker = yf.Ticker("DX-Y.NYB")
                    if hasattr(dxy_ticker, 'fast_info') and 'last_price' in dxy_ticker.fast_info:
                        result['dxy'] = dxy_ticker.fast_info['last_price']
                
                # VIX
                if "vix" not in result:
                    vix_ticker = yf.Ticker("^VIX")
                    if hasattr(vix_ticker, 'fast_info') and 'last_price' in vix_ticker.fast_info:
                        result['vix'] = vix_ticker.fast_info['last_price']
            except Exception as e:
                logger.warning(f"yfinance fallback failed: {e}")

        # Calculate Macro Trend Score (-100 to 100)
        score = 0
        rate = result.get("interest_rate", 5.0)
        
        if rate < 3.0: score += 30
        elif rate > 5.0: score -= 30
        
        # DTWEXBGS is usually higher than DXY (e.g., 120 vs 104). Adjust logic slightly or just track trend.
        # For simplicity, we assume higher is bearish for crypto.
        dxy_val = result.get('dxy')
        if dxy_val:
            if dxy_val > 115: score -= 20 # Strong Dollar (Broad Index) -> Bad for Crypto
            elif dxy_val < 100: score += 20
        
        result["macro_score"] = score
        result["timestamp"] = datetime.now().isoformat()
        
        # Log summary
        rate_str = f"{rate}%" if result.get("interest_rate") else "N/A"
        dxy_str = f"{result.get('dxy', 0):.2f}" if result.get('dxy') else "N/A"
        logger.info(f"🌍 Macro Data: Rate={rate_str} | DXY(Broad)={dxy_str} | Score={score}")
        
        return result
        
    def _get_series_latest(self, series_id: str) -> Optional[float]:
        """Fetch single series from FRED"""
        if not self.api_key:
            return None
            
        # Check cache
        if series_id in self.cache:
            ts, val = self.cache[series_id]
            if (datetime.now() - ts).total_seconds() < self.cache_duration:
                return val
                
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
                return value
                
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return None
            
        return None
    
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
            'macro_VIX': [data.get('vix', 20.0)]
        })
        
        logger.info(f"📊 Macro data fetched: Score={data.get('macro_score', 0):.1f}")
        return df
