import ccxt.async_support as ccxt
import logging
import asyncio
import pandas as pd
from typing import Dict, List, Optional
from src.config import Config

logger = logging.getLogger("BINANCE_API")

class BinanceAPI:
    """
    Infrastructure Layer: Direct communication with Binance Futures text
    """
    def __init__(self):
        self.exchange = None
        self.markets_loaded = False
        
    async def connect(self):
        """Initialize CCXT exchange instance"""
        if self.exchange:
            return

        try:
            self.exchange = ccxt.binance({
                'apiKey': Config.BINANCE_API_KEY,
                'secret': Config.BINANCE_API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            await self.exchange.load_markets()
            self.markets_loaded = True
            logger.info("✅ Binance Connected (Futures)")
        except Exception as e:
            logger.critical(f"❌ Binance Connection Failed: {e}")
            self.exchange = None
            
    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            
    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data and return as DataFrame"""
        if not self.exchange: await self.connect()
        if not self.exchange: return pd.DataFrame()
        
        try:
            # Normalize symbol for CCXT (BTCUSDT -> BTC/USDT:USDT or just BTC/USDT depending on ccxt version)
            # But usually for binance futures it's 'BTC/USDT'
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            
            ohlcv = await self.exchange.fetch_ohlcv(ticker, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    async def get_balance(self) -> float:
        """Get USDT Balance"""
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        
        try:
            balance = await self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        try:
             ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
             ticker_data = await self.exchange.fetch_ticker(ticker)
             return float(ticker_data['last'])
        except Exception as e:
            logger.error(f"Price fetch error {symbol}: {e}")
            return 0.0
