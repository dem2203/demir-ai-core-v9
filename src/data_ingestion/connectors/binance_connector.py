import ccxt.async_support as ccxt
import asyncio
from typing import Dict, List, Optional
import ccxt.async_support as ccxt
import asyncio
from typing import Dict, List, Optional
import logging
import pandas as pd
from src.config.settings import Config
from src.validation.validator import SignalValidator

logger = logging.getLogger("BINANCE_CONNECTOR")

class BinanceConnector:
    """
    PROFESYONEL BINANCE BAĞLANTISI (FUTURES DATA)
    Fiyat, Funding Rate ve Open Interest verilerini çeker.
    """
    
    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.exchange = None
        
        self.exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} # Vadeli İşlemler
        }

    async def connect(self):
        try:
            self.exchange = ccxt.binance(self.exchange_config)
            await self.exchange.load_markets()
            logger.info("CONNECTED: Binance Markets Loaded.")
        except Exception as e:
            logger.critical(f"CONNECTION FAILED: {e}")
            # Zero-Mock: Bağlantı yoksa None kalır, sahte bağlantı objesi oluşturulmaz.
            self.exchange = None

    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List[Dict]]:
        if not self.exchange: await self.connect()
        if not self.exchange: return None # Bağlantı yoksa veri yok
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            formatted_data = []
            for candle in ohlcv:
                data_point = {
                    'symbol': symbol, 'timestamp': candle[0],
                    'open': float(candle[1]), 'high': float(candle[2]),
                    'low': float(candle[3]), 'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'source': 'binance'
                }
                formatted_data.append(data_point)
            
            # Gelen veriyi doğrula
            if not SignalValidator.validate_incoming_data(formatted_data):
                return None
                
            return formatted_data
        except Exception as e:
            logger.error(f"Candle Fetch Error ({symbol}): {e}")
            return None

    async def fetch_futures_data(self, symbol: str) -> Dict:
        """
        Funding Rate ve Open Interest verilerini çeker.
        """
        if not self.exchange: await self.connect()
        if not self.exchange: return {}  # Boş dön, uydurma veri dönme
        
        try:
            # Funding Rate
            funding = await self.exchange.fetch_funding_rate(symbol)
            fr_value = funding.get('fundingRate')
            fr = float(fr_value) if fr_value is not None else 0.0
            
            # Open Interest
            try:
                oi_data = await self.exchange.fetch_open_interest(symbol)
                oi_value = oi_data.get('openInterestValue') if oi_data else None
                oi = float(oi_value) if oi_value is not None else 0.0
            except:
                oi = 0.0
            
            return {'funding_rate': fr, 'open_interest': oi}
        except Exception as e:
            logger.warning(f"Futures Data Error ({symbol}): {e}")
            # Hata durumunda boş dict dönüyoruz
            return {'funding_rate': 0.0, 'open_interest': 0.0}

    async def close(self):
        if self.exchange: await self.exchange.close()
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Synchronous wrapper for Dashboard usage.
        Returns Pandas DataFrame.
        """
        try:
            # Handle async call in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If loop is running (e.g. Streamlit), we can't use run_until_complete directly
                # This is a bit tricky. For now, create a new loop for this specific call if possible,
                # or rely on nest_asyncio if available. 
                # Simpler fallback: just allow the error or use a clean resource.
                # Use a fresh loop in a separate thread if needed, but let's try standard approach first.
                future = asyncio.run_coroutine_threadsafe(self.fetch_candles(symbol, timeframe, limit), loop)
                data = future.result()
            else:
                data = loop.run_until_complete(self.fetch_candles(symbol, timeframe, limit))
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Sync OHLCV Error: {e}")
            return pd.DataFrame()
