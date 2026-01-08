import aiohttp
import asyncio
import logging
from datetime import datetime
from src.config import Config

logger = logging.getLogger("MACRO_BRAIN")

class MacroBrain:
    """
    The World Observer.
    Fetches VIX, DXY, SPX, BTC Dominance to determine the global regime.
    """
    def __init__(self):
        self.fred_key = Config.FRED_API_KEY
        self.base_fred_url = "https://api.stlouisfed.org/fred/series/observations"
        self.cache = {}
        
    async def _fetch_fred(self, series_id: str):
        """Fetch single series from FRED"""
        if not self.fred_key:
            return None
            
        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json",
            "limit": 1,
            "sort_order": "desc"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_fred_url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "observations" in data and data["observations"]:
                            return float(data["observations"][0]["value"])
        except Exception as e:
            logger.warning(f"FRED Fetch Error ({series_id}): {e}")
        return None

    async def _fetch_coingecko_global(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/global") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('data', {})
        except Exception as e:
            logger.warning(f"CoinGecko Error: {e}")
        return {}

    async def analyze_world(self) -> dict:
        """
        Gathers all macro data and returns a structured analysis.
        """
        # Parallel Fetch
        vix_task = self._fetch_fred("VIXCLS")      # Volatility Index
        dxy_task = self._fetch_fred("DTWEXBGS")    # Dollar Index
        spx_task = self._fetch_fred("SP500")       # S&P 500
        cg_task = self._fetch_coingecko_global()
        
        vix, dxy, spx, cg_data = await asyncio.gather(vix_task, dxy_task, spx_task, cg_task)
        
        btc_dominance = cg_data.get('market_cap_percentage', {}).get('btc', 50.0)
        
        # Scoring Logic
        score = 0
        reasons = []
        
        # VIX Analysis
        if vix:
            if vix > 30: 
                score -= 30
                reasons.append(f"VIX High ({vix}): Extreme Fear")
            elif vix < 15:
                score += 10
                reasons.append(f"VIX Low ({vix}): Stable Market")
            else:
                reasons.append(f"VIX Neutral ({vix})")
        
        # DXY Analysis
        if dxy:
            if dxy > 106:
                score -= 20
                reasons.append(f"DXY Strong ({dxy}): Dollar Squeezing Assets")
            elif dxy < 100:
                score += 20
                reasons.append(f"DXY Weak ({dxy}): Good for Crypto")
                
        # BTC Dominance
        if btc_dominance > 60:
            reasons.append(f"BTC Dominance High ({btc_dominance:.1f}%): Alts Suffering")
        
        regime = "NEUTRAL"
        if score > 20: regime = "RISK_ON"
        elif score < -20: regime = "RISK_OFF"
        
        return {
            "regime": regime,
            "score": score,
            "vix": vix,
            "dxy": dxy,
            "btc_dominance": btc_dominance,
            "reasoning": reasons
        }
