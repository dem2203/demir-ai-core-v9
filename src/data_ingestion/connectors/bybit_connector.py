import ccxt.async_support as ccxt
import logging
from typing import Dict, List, Optional
from src.config.settings import Config

logger = logging.getLogger("BYBIT_CONNECTOR")

class BybitConnector:
    """
    BYBIT BAĞLANTISI (FUTURES/DERIVATIVES)
    """
    
    def __init__(self):
        self.api_key = Config.BYBIT_API_KEY
        self.api_secret = Config.BYBIT_API_SECRET
        self.exchange = None
        
        self.exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'} # USDT Perpetual
        }

    async def connect(self):
        try:
            self.exchange = ccxt.bybit(self.exchange_config)
            # Bybit bazen load_markets'te yavaş olabilir, timeout ayarı eklenebilir
            await self.exchange.load_markets()
            logger.info("CONNECTED: Bybit Markets Loaded.")
        except Exception as e:
            logger.critical(f"BYBIT CONNECTION FAILED: {e}")
            # Zero-Mock Policy: Hata varsa sessizce geçme, logla ve None dön
            self.exchange = None

    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List[Dict]]:
        if not self.exchange: await self.connect()
        if not self.exchange: return None

        try:
            # Bybit sembol formatı düzeltme (örn: BTC/USDT:USDT)
            # CCXT genellikle standart sembolü kabul eder ama bazen mapping gerekir.
            # Şimdilik standart deniyoruz.
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            formatted_data = []
            for candle in ohlcv:
                data_point = {
                    'symbol': symbol, 'timestamp': candle[0],
                    'open': float(candle[1]), 'high': float(candle[2]),
                    'low': float(candle[3]), 'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'source': 'bybit'
                }
                formatted_data.append(data_point)
            return formatted_data
        except Exception as e:
            logger.error(f"Bybit Candle Fetch Error ({symbol}): {e}")
            return None

    async def fetch_funding_rate(self, symbol: str) -> float:
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return float(funding['fundingRate'])
        except Exception as e:
            logger.warning(f"Bybit Funding Rate Error ({symbol}): {e}")
            return 0.0

    async def close(self):
        if self.exchange: await self.exchange.close()
