import ccxt.async_support as ccxt
import logging
import asyncio
import pandas as pd
from typing import Dict, List, Optional
from src.config import Config
from src.utils.circuit_breaker import CircuitBreaker  # FIX 2.2

logger = logging.getLogger("BINANCE_API")

class BinanceAPI:
    """
    Infrastructure Layer: Direct communication with Binance Futures text
    """
    def __init__(self):
        self.exchange = None
        self.markets_loaded = False
        # FIX 2.2: Initialize Circuit Breaker
        self.circuit_breaker = CircuitBreaker("BinanceAPI", failure_threshold=5, recovery_timeout=60)
        
    async def connect(self):
        """Initialize CCXT exchange instance"""
        if self.exchange:
            return
            
        # FIX 2.2: Check circuit breaker before connecting
        if not self.circuit_breaker.allow_request():
            logger.warning("⛔ Circuit Breaker OPEN (Binance). Skipping connection.")
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
            self.circuit_breaker.record_success()
        except Exception as e:
            logger.critical(f"❌ Binance Connection Failed: {e}")
            self.circuit_breaker.record_failure()
            self.exchange = None
            
    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
            
    # FIX 2.2: Apply Circuit Breaker via decorator logic (manual here as method decorator needs 'self')
    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data and return as DataFrame"""
        if not self.circuit_breaker.allow_request():
            logger.warning(f"⛔ Circuit Breaker OPEN. Skipping fetch for {symbol}")
            return pd.DataFrame()
            
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
            
            self.circuit_breaker.record_success()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Binance API timeout fetching candles for {symbol}")
            self.circuit_breaker.record_failure()
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            self.circuit_breaker.record_failure()
            return pd.DataFrame()

    async def get_balance(self) -> float:
        """Get USDT Balance"""
        if not self.circuit_breaker.allow_request(): return 0.0
        
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        
        try:
            # FIX 1.8: Add timeout
            balance = await asyncio.wait_for(
                self.exchange.fetch_balance(),
                timeout=10.0
            )
            self.circuit_breaker.record_success()
            return float(balance['USDT']['free'])
        except asyncio.TimeoutError:
            logger.error("⏱️ Binance API timeout fetching balance")
            self.circuit_breaker.record_failure()
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            self.circuit_breaker.record_failure()
            return 0.0

    async def get_current_price(self, symbol: str) -> float:
        if not self.circuit_breaker.allow_request(): return 0.0
        
        if not self.exchange: await self.connect()
        if not self.exchange: return 0.0
        try:
            ticker = symbol.replace("USDT", "/USDT") if "/" not in symbol else symbol
            # FIX 1.8: Add timeout
            ticker_data = await asyncio.wait_for(
                self.exchange.fetch_ticker(ticker),
                timeout=10.0
            )
            self.circuit_breaker.record_success()
            return float(ticker_data['last'])
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Binance API timeout getting price for {symbol}")
            self.circuit_breaker.record_failure()
            return 0.0
        except Exception as e:
            logger.error(f"Price fetch error {symbol}: {e}")
            self.circuit_breaker.record_failure()
            return 0.0
