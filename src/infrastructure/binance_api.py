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
            # Normalize symbol for CCXT
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            
            # FIX 1.8: Add timeout to prevent hanging
            ohlcv = await asyncio.wait_for(
                self.exchange.fetch_ohlcv(ticker, timeframe, limit=limit),
                timeout=10.0
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Binance API timeout fetching candles for {symbol}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    async def get_balance(self) -> float:
        """Get USDT Balance"""
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        
        try:
            # FIX 1.8: Add timeout
            balance = await asyncio.wait_for(
                self.exchange.fetch_balance(),
                timeout=10.0
            )
            return float(balance['USDT']['free'])
        except asyncio.TimeoutError:
            logger.error("⏱️ Binance API timeout fetching balance")
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        try:
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            # FIX 1.8: Add timeout
            ticker_data = await asyncio.wait_for(
                self.exchange.fetch_ticker(ticker),
                timeout=10.0
            )
            return float(ticker_data['last'])
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Binance API timeout getting price for {symbol}")
            return 0.0
        except Exception as e:
            logger.error(f"Price fetch error {symbol}: {e}")
            return 0.0
