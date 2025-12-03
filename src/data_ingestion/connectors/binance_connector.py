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
        if not self.exchange: return {} # Boş dön, uydurma veri dönme
        
        try:
            # Funding Rate
            funding = await self.exchange.fetch_funding_rate(symbol)
            fr = float(funding['fundingRate'])
            
            # Open Interest
            oi_data = await self.exchange.fetch_open_interest(symbol)
            oi = float(oi_data.get('openInterestValue', 0))
            
            return {'funding_rate': fr, 'open_interest': oi}
        except Exception as e:
            logger.warning(f"Futures Data Error ({symbol}): {e}")
            # Hata durumunda 0 dönmek 'nötr' veri kabul edilebilir mi? 
            # Zero-Mock kuralına göre, eğer veri yoksa analiz eksik kalmalı.
            # Ancak kodun patlamaması için boş dict veya None dönmek daha iyi.
            # Şimdilik boş dict dönüyoruz, analiz katmanı bunu "veri yok" olarak algılamalı.
            return {}

    async def close(self):
        if self.exchange: await self.exchange.close()
