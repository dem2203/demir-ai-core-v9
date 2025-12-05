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
    4. US Dollar Index (DXY - proxied or fetched)
    5. S&P 500 (SP500)
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
            "ma": "M2SL" # Money Supply
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
        
        # M2 (Money Supply) Growth proxy check - simplified for now
        
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
