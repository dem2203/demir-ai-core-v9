import aiohttp
import asyncio
import logging
from datetime import datetime
from src.config import Config
from fredapi import Fred

logger = logging.getLogger("MACRO_BRAIN")

class MacroBrain:
    """
    The World Observer - PROFESSIONAL EDITION
    Multi-source data reliability with quality tracking
    """
    def __init__(self):
        self.twelve_data_key = Config.TWELVE_DATA_API_KEY
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_API_KEY
        # FRED API (St. Louis Fed - most reliable)
        self.fred = Fred(api_key=Config.FRED_API_KEY) if Config.FRED_API_KEY else None
        
    async def _fetch_yahoo_finance(self, symbol: str):
        """PRIMARY: Yahoo Finance API"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': '1d', 'range': '1d'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quote = data['chart']['result'][0]['meta']
                        value = quote['regularMarketPrice']
                        logger.info(f"✅ Yahoo Finance: {symbol} = {value}")
                        return value
        except Exception as e:
            logger.warning(f"⚠️ Yahoo Finance failed ({symbol}): {e}")
        return None
    
    async def _fetch_twelve_data(self, symbol: str):
        """FALLBACK 1: Twelve Data API"""
        if not self.twelve_data_key:
            return None
            
        try:
            url = f"https://api.twelvedata.com/price"
            params = {'symbol': symbol, 'apikey': self.twelve_data_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        value = float(data.get('price', 0))
                        if value > 0:
                            logger.info(f"✅ Twelve Data: {symbol} = {value}")
                            return value
        except Exception as e:
            logger.warning(f"⚠️ Twelve Data failed ({symbol}): {e}")
        return None
    
    async def _fetch_alpha_vantage(self, symbol: str):
        """FALLBACK 2: Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return None
            
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        quote = data.get('Global Quote', {})
                        value = float(quote.get('05. price', 0))
                        if value > 0:
                            logger.info(f"✅ Alpha Vantage: {symbol} = {value}")
                            return value
        except Exception as e:
            logger.warning(f"⚠️ Alpha Vantage failed ({symbol}): {e}")
        return None

    def _fetch_fred(self, series_id: str):
        """FRED API (Federal Reserve Economic Data) - Most Reliable"""
        if not self.fred:
            return None
        try:
            data = self.fred.get_series_latest_release(series_id)
            if data is not None and len(data) > 0:
                value = float(data.iloc[-1])
                logger.info(f"✅ FRED: {series_id} = {value}")
                return value
        except Exception as e:
            logger.warning(f"⚠️ FRED failed ({series_id}): {e}")
        return None

    async def _get_vix_with_fallback(self):
        """VIX with multi-source fallback chain"""
        # Try FRED (primary - most reliable)
        try:
            vix = await asyncio.to_thread(self._fetch_fred, "VIXCLS")
            if vix: return vix, "fred"
        except Exception as e:
            logger.warning(f"FRED VIX failed: {e}")
        
        # Try Yahoo Finance (fallback 1)
        vix = await self._fetch_yahoo_finance("^VIX")
        if vix: return vix, "yahoo"
        
        # Try Twelve Data (fallback 2)
        vix = await self._fetch_twelve_data("VIX")
        if vix: return vix, "twelve_data"
        
        # Try Alpha Vantage (fallback 3)
        vix = await self._fetch_alpha_vantage("VIX")
        if vix: return vix, "alpha_vantage"
        
        # ALL SOURCES FAILED
        logger.error("🚨 VIX: ALL DATA SOURCES FAILED!")
        return None, "none"
    
    async def _get_dxy_with_fallback(self):
        """DXY with multi-source fallback chain"""
        # Try FRED (primary - most reliable)
        try:
            dxy = await asyncio.to_thread(self._fetch_fred, "DTWEXBGS")
            if dxy: return dxy, "fred"
        except Exception as e:
            logger.warning(f"FRED DXY failed: {e}")
        
        # Try Yahoo Finance (fallback 1)
        dxy = await self._fetch_yahoo_finance("DX-Y.NYB")
        if dxy: return dxy, "yahoo"
        
        # Try Twelve Data (fallback 2)
        dxy = await self._fetch_twelve_data("DXY")
        if dxy: return dxy, "twelve_data"
        
        # Try Alpha Vantage (fallback 3)  
        dxy = await self._fetch_alpha_vantage("DXY")
        if dxy: return dxy, "alpha_vantage"
        
        # ALL SOURCES FAILED
        logger.error("🚨 DXY: ALL DATA SOURCES FAILED!")
        return None, "none"
    
    async def _get_btc_dominance(self):
        """BTC Dominance from CoinGecko"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/global", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get('data', {}).get('market_cap_percentage', {}).get('btc', 50.0)
        except Exception as e:
            logger.warning(f"CoinGecko Error: {e}")
        return 50.0

    async def analyze_world(self) -> dict:
        """
        Professional macro analysis with data quality tracking
        """
        logger.info("🌍 Fetching macro data (multi-source fallback enabled)...")
        
        # Parallel fetch with source tracking
        vix_task = self._get_vix_with_fallback()
        dxy_task = self._get_dxy_with_fallback()
        btc_dom_task = self._get_btc_dominance()
        
        (vix, vix_source), (dxy, dxy_source), btc_dominance = await asyncio.gather(
            vix_task, dxy_task, btc_dom_task
        )
        
        # DATA QUALITY CHECK
        data_quality = "HIGH"
        failed_sources = []
        
        if vix is None:
            failed_sources.append("VIX")
            vix = 16.0  # Neutral default
            data_quality = "DEGRADED"
            logger.warning("⚠️ VIX unavailable - using neutral default (16.0)")
            
        if dxy is None:
            failed_sources.append("DXY")
            dxy = 104.0  # Typical value
            data_quality = "DEGRADED"
            logger.warning("⚠️ DXY unavailable - using typical default (104.0)")
        
        # Adjust quality based on sources
        if vix_source != "yahoo" or dxy_source != "yahoo":
            if data_quality == "HIGH":
                data_quality = "MEDIUM"
        
        # Continue with analysis even if data unavailable (graceful degradation)
        score = 0
        reasons = []
        
        # Add data source warnings
        if data_quality == "DEGRADED":
            reasons.append(f"⚠️ Macro data degraded (using defaults) - relying on TA/Price Action")
        elif data_quality == "MEDIUM":
            reasons.append(f"⚠️ Data quality: {data_quality} (fallback sources)")
        
        # VIX Analysis
        if vix > 30: 
            score -= 30
            reasons.append(f"VIX Yüksek ({vix:.2f}): Aşırı Korku")
        elif vix < 15:
            score += 10
            reasons.append(f"VIX Düşük ({vix:.2f}): Stabil Piyasa")
        else:
            reasons.append(f"VIX Nötr ({vix:.2f})")
        
        # DXY Analysis
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
        
        logger.info(f"📊 Macro: {regime} | Quality: {data_quality} | VIX: {vix:.2f} ({vix_source}) | DXY: {dxy:.2f} ({dxy_source})")
        
        return {
            "regime": regime,
            "score": score,
            "vix": vix,
            "dxy": dxy,
            "btc_dominance": btc_dominance,
            "data_quality": data_quality,
            "vix_source": vix_source,
            "dxy_source": dxy_source,
            "reasoning": reasons,
            "error": False  # Never error - graceful degradation
        }
