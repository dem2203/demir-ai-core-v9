import ccxt.async_support as ccxt
import asyncio
from typing import Dict, List, Optional
import logging
from src.config.settings import Config
from src.validation.validator import SignalValidator

logger = logging.getLogger("BINANCE_CONNECTOR")

class BinanceConnector:
    """
    Profesyonel Binance Bağlantı Yöneticisi.
    Anlık Ticker yerine OHLCV (Mum) verisi çeker.
    """
    
    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.exchange = None
        
        self.exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        }

    async def connect(self):
        try:
            self.exchange = ccxt.binance(self.exchange_config)
            await self.exchange.load_markets()
            logger.info("CONNECTED: Binance Markets Loaded.")
        except Exception as e:
            logger.critical(f"CONNECTION FAILED: {e}")
            raise e

    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List[Dict]]:
        """
        Feature Engineering için gerekli olan GEÇMİŞ MUM (OHLCV) verilerini çeker.
        """
        if not self.exchange:
            await self.connect()
            
        try:
            # OHLCV verisini çek (Open, High, Low, Close, Volume)
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            formatted_data = []
            for candle in ohlcv:
                # CCXT formatı: [timestamp, open, high, low, close, volume]
                data_point = {
                    'symbol': symbol,
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                }
                formatted_data.append(data_point)
            
            # Son mumu (henüz kapanmamış olabilir) validator'dan geçir
            if formatted_data and SignalValidator.validate_incoming_data(formatted_data[-1]):
                return formatted_data
            else:
                return None

        except Exception as e:
            logger.error(f"FETCH ERROR ({symbol}): {str(e)}")
            return None

    async def close(self):
        if self.exchange:
            await self.exchange.close()