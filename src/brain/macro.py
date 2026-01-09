import aiohttp
import asyncio
import logging
from datetime import datetime
from src.config import Config

logger = logging.getLogger("MACRO_BRAIN")

class MacroBrain:
    """
    The World Observer - PROFESSIONAL EDITION
    Uses Yahoo Finance for accurate real-time data (NO HALLUCINATIONS)
    """
    def __init__(self):
        # NO CACHE - Always fetch fresh data
        pass
        
    async def _fetch_yahoo_finance(self, symbol: str):
        """
        Fetch real-time data from Yahoo Finance API
        """
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': '1d', 'range': '1d'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quote = data['chart']['result'][0]['meta']
                        return quote['regularMarketPrice']
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch error ({symbol}): {e}")
        return None

    async def _get_vix(self):
        """VIX (Fear Index) from Yahoo Finance"""
        return await self._fetch_yahoo_finance("^VIX")
    
    async def _get_dxy(self):
        """DXY (Dollar Index) from Yahoo Finance - ACCURATE!"""
        return await self._fetch_yahoo_finance("DX-Y.NYB")
    
    async def _get_btc_dominance(self):
        """BTC Dominance from CoinGecko"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/global") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('data', {}).get('market_cap_percentage', {}).get('btc', 50.0)
        except Exception as e:
            logger.warning(f"CoinGecko Error: {e}")
        return 50.0

    async def analyze_world(self) -> dict:
        """
        Gathers all macro data and returns a structured analysis.
        NOW WITH ACCURATE DATA!
        """
        logger.info("🌍 Fetching REAL macro data (Yahoo Finance)...")
        
        # Parallel Fetch
        vix_task = self._get_vix()
        dxy_task = self._get_dxy()
        btc_dom_task = self._get_btc_dominance()
        
        vix, dxy, btc_dominance = await asyncio.gather(vix_task, dxy_task, btc_dom_task)
        
        # Scoring Logic
        score = 0
        reasons = []
        
        # VIX Analysis
        if vix:
            if vix > 30: 
                score -= 30
                reasons.append(f"VIX Yüksek ({vix:.2f}): Aşırı Korku")
            elif vix < 15:
                score += 10
                reasons.append(f"VIX Düşük ({vix:.2f}): Stabil Piyasa")
            else:
                reasons.append(f"VIX Nötr ({vix:.2f})")
        
        # DXY Analysis - NOW ACCURATE!
        if dxy:
            if dxy > 106:
                score -= 20
                reasons.append(f"DXY Güçlü ({dxy:.2f}): Dolar Varlıkları Sıkıştırıyor")
            elif dxy < 100:
                score += 20
                reasons.append(f"DXY Zayıf ({dxy:.2f}): Kripto İçin İyi")
            else:
                reasons.append(f"DXY Nötr ({dxy:.2f})")
                
        # BTC Dominance
        if btc_dominance > 60:
            reasons.append(f"BTC Dominansı Yüksek ({btc_dominance:.1f}%): Altcoin'ler Zor Durumda")
        
        regime = "NEUTRAL"
        if score > 20: regime = "RISK_ON"
        elif score < -20: regime = "RISK_OFF"
        
        logger.info(f"📊 Macro: {regime} (VIX: {vix:.2f}, DXY: {dxy:.2f}, BTC Dom: {btc_dominance:.1f}%)")
        
        return {
            "regime": regime,
            "score": score,
            "vix": vix,
            "dxy": dxy,
            "btc_dominance": btc_dominance,
            "reasoning": reasons
        }
