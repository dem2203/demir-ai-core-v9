import ccxt.async_support as ccxt
import asyncio
from typing import Dict, List, Optional
import logging
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
            raise e

    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List[Dict]]:
        if not self.exchange: await self.connect()
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            formatted_data = []
            for candle in ohlcv:
                data_point = {
                    'symbol': symbol, 'timestamp': candle[0],
                    'open': float(candle[1]), 'high': float(candle[2]),
                    'low': float(candle[3]), 'close': float(candle[4]),
                    'volume': float(candle[5])
                }
                formatted_data.append(data_point)
            return formatted_data
        except Exception as e:
            logger.error(f"Candle Fetch Error ({symbol}): {e}")
            return None

    async def fetch_futures_data(self, symbol: str) -> Dict:
        """
        Funding Rate ve Open Interest verilerini çeker.
        Piyasadaki 'Kaldıraç Şişkinliğini' ölçmek için kritiktir.
        """
        if not self.exchange: await self.connect()
        try:
            # Funding Rate (Fonlama Oranı)
            funding = await self.exchange.fetch_funding_rate(symbol)
            fr = float(funding['fundingRate'])
            
            # Open Interest (Açık Pozisyon Büyüklüğü - USDT cinsinden)
            # Not: Bazı borsalarda 'openInterestValue' olmayabilir, kontrol edelim
            oi_data = await self.exchange.fetch_open_interest(symbol)
            oi = float(oi_data.get('openInterestValue', 0))
            
            return {'funding_rate': fr, 'open_interest': oi}
        except Exception as e:
            logger.warning(f"Futures Data Error ({symbol}): {e}")
            return {'funding_rate': 0.0, 'open_interest': 0.0}

    async def close(self):
        if self.exchange: await self.exchange.close()
