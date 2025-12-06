import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import os
from src.config.settings import Config

logger = logging.getLogger("MACRO_CONNECTOR")

class MacroConnector:
    """
    MACRO ECONOMIC INTELLIGENCE
    Fetches key economic indicators from FRED (Federal Reserve Economic Data).
    
    Indicators:
    1. US Interest Rate (FEDFUNDS)
    2. Inflation (CPIAUCSL)
    3. Unemployment Rate (UNRATE)
    4. M2 Money Supply (M2SL)
    """
    
    def __init__(self):
        self.api_key = os.getenv("FRED_API_KEY")
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        self.cache_duration = 3600 * 24  # 24 hours (Macro data changes slowly)
        
        if not self.api_key:
            logger.warning("⚠️ FRED_API_KEY not found. Macro analysis will use fallback values.")
            
    def fetch_data(self) -> Dict[str, float]:
        """Get latest macro indicators"""
        
        indicators = {
            "interest_rate": "FEDFUNDS",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "m2_money_supply": "M2SL"
        }
        
        result = {}
        
        for name, series_id in indicators.items():
            value = self._get_series_latest(series_id)
            if value is not None:
                result[name] = value
                
        # Calculate Macro Trend Score (-100 to 100)
        # Low Rate, High M2 = Bullish
        # High Rate, High CPI = Bearish
        
        score = 0
        rate = result.get("interest_rate", 5.0)
        
        if rate < 3.0: score += 30
        elif rate > 5.0: score -= 30
        
        result["macro_score"] = score
        result["timestamp"] = datetime.now().isoformat()
        
        # Log summary
        rate_str = f"{rate}%" if result.get("interest_rate") else "N/A"
        logger.info(f"🌍 Macro Data: Rate={rate_str} | Score={score}")
        
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
        
        Args:
            period: Time period (ignored, FRED data is current snapshot)
            interval: Interval (ignored)
        
        Returns:
            DataFrame with macro indicators as columns
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
            'm2_money_supply': [data.get('m2_money_supply', 0)]
        })
        
        logger.info(f"📊 Macro data fetched: Score={data.get('macro_score', 0):.1f}")
        return df
