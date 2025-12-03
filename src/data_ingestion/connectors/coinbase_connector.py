import ccxt.async_support as ccxt
import logging
from typing import Dict, List, Optional
from src.config.settings import Config

logger = logging.getLogger("COINBASE_CONNECTOR")

class CoinbaseConnector:
    """
    COINBASE BAĞLANTISI (SPOT REFERANS)
    Coinbase genellikle kurumsal akışın öncüsüdür. Fiyat sapmalarını kontrol etmek için kullanılır.
    """
    
    def __init__(self):
        self.api_key = Config.COINBASE_API_KEY
        self.api_secret = Config.COINBASE_API_SECRET
        self.exchange = None
        
        self.exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        }

    async def connect(self):
        try:
            self.exchange = ccxt.coinbase(self.exchange_config)
            await self.exchange.load_markets()
            logger.info("CONNECTED: Coinbase Markets Loaded.")
        except Exception as e:
            logger.critical(f"COINBASE CONNECTION FAILED: {e}")
            self.exchange = None

    async def fetch_price(self, symbol: str) -> Optional[float]:
        """Sadece anlık fiyat çeker (Referans için)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            logger.error(f"Coinbase Price Fetch Error ({symbol}): {e}")
            return None

    async def close(self):
        if self.exchange: await self.exchange.close()
