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
        errors = []
        
        for name, series_id in indicators.items():
            value, error = self._get_series_latest(series_id)
            if value is not None:
                result[name] = value
            elif error:
                errors.append(f"{name}: {error}")

        # 2. Market Data Fallback (yfinance)
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
                msg = f"yfinance fallback failed: {e}"
                logger.warning(msg)
                errors.append(msg)
        
        # Store errors in result for debug
        if errors:
            result['debug_errors'] = "; ".join(errors)

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
        
        # v6 MACRO FEATURES: Cross-market correlations + Dashboard Display
        try:
            import ccxt
            exchange = ccxt.binance()
            
            # ETH/BTC Ratio
            eth_btc = exchange.fetch_ticker('ETH/BTC')
            result['eth_btc_ratio'] = float(eth_btc['last']) if eth_btc else 0.0
            
            # Get BTC price for Gold/BTC
            btc_usdt = exchange.fetch_ticker('BTC/USDT')
            btc_price = float(btc_usdt['last']) if btc_usdt else 100000
            
            # ✅ DASHBOARD: Gold Price with 24h change (GC=F futures)
            try:
                gold = yf.Ticker("GC=F")
                gold_info = gold.fast_info
                gold_price = gold_info.get('lastPrice') or gold_info.get('last_price') or gold_info.get('regularMarketPrice', 0)
                gold_prev = gold_info.get('previousClose') or gold_info.get('regularMarketPreviousClose', 0)
                result['gold'] = gold_price if gold_price else 0
                result['gold_change'] = ((gold_price - gold_prev) / gold_prev * 100) if gold_prev and gold_price else 0
                result['gold_btc_ratio'] = gold_price / btc_price if btc_price > 0 and gold_price else 0.0
            except Exception as e:
                logger.warning(f"Gold fetch failed: {e}")
                result['gold'] = 0
                result['gold_change'] = 0
                result['gold_btc_ratio'] = 0.02
            
            # ✅ DASHBOARD: Nasdaq Composite with 24h change (^IXIC)
            try:
                nasdaq = yf.Ticker("^IXIC")
                nasdaq_info = nasdaq.fast_info
                nasdaq_price = nasdaq_info.get('lastPrice') or nasdaq_info.get('last_price') or nasdaq_info.get('regularMarketPrice', 0)
                nasdaq_prev = nasdaq_info.get('previousClose') or nasdaq_info.get('regularMarketPreviousClose', 0)
                result['nasdaq'] = nasdaq_price if nasdaq_price else 0
                result['nasdaq_change'] = ((nasdaq_price - nasdaq_prev) / nasdaq_prev * 100) if nasdaq_prev and nasdaq_price else 0
            except Exception as e:
                logger.warning(f"Nasdaq fetch failed: {e}")
                result['nasdaq'] = 0
                result['nasdaq_change'] = 0
            
            # ✅ DASHBOARD: BTC Dominance with 24h change (CoinGecko free API)
            try:
                cg_response = requests.get("https://api.coingecko.com/api/v3/global", timeout=5)
                if cg_response.status_code == 200:
                    global_data = cg_response.json()
                    btc_d = global_data['data']['market_cap_percentage']['btc']
                    btc_d_24h = global_data['data'].get('market_cap_change_percentage_24h_usd', 0)
                    result['btc_dominance'] = btc_d
                    result['btc_dominance_change'] = btc_d_24h  # Market cap change proxy
                else:
                    result['btc_dominance'] = 0
                    result['btc_dominance_change'] = 0
            except Exception as e:
                logger.warning(f"BTC.D fetch failed: {e}")
                result['btc_dominance'] = 0
                result['btc_dominance_change'] = 0
                
            # S&P500/BTC correlation proxy
            try:
                spy = yf.Ticker("SPY")
                spy_price = spy.fast_info.get('lastPrice') or spy.fast_info.get('last_price') or spy.fast_info.get('regularMarketPrice', 500)
                result['sp500_btc_ratio'] = spy_price / (btc_price / 1000) if btc_price > 0 else 0.0
            except:
                result['sp500_btc_ratio'] = 5.0  # fallback
                
            logger.info(f"📈 Dashboard Macro: Gold=${result.get('gold', 0):,.0f} | Nasdaq={result.get('nasdaq', 0):,.0f} | BTC.D={result.get('btc_dominance', 0):.1f}%")
            
        except Exception as e:
            logger.warning(f"Could not fetch v6 ratios: {e}")
            result['eth_btc_ratio'] = 0.0
            result['gold_btc_ratio'] = 0.0
            result['gold'] = 0
            result['nasdaq'] = 0
            result['btc_dominance'] = 0
            result['sp500_btc_ratio'] = 0.0
        
        # Log summary
        rate_str = f"{rate}%" if result.get("interest_rate") else "N/A"
        dxy_str = f"{result.get('dxy', 0):.2f}" if result.get('dxy') else "N/A"
        logger.info(f"🌍 Macro Data: Rate={rate_str} | DXY(Broad)={dxy_str} | Score={score}")
        
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
