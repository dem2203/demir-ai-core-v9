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
    
    # === Phase 29.3 Enhancements ===
    
    async def fetch_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Fetch order book depth (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit=limit)
            
            # Calculate bid/ask volumes
            bid_volume = sum([price * amount for price, amount in orderbook['bids'][:limit]])
            ask_volume = sum([price * amount for price, amount in orderbook['asks'][:limit]])
            
            return {
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'bid_volume_usd': bid_volume,
                'ask_volume_usd': ask_volume,
                'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
                'source': 'bybit'
            }
        except Exception as e:
            logger.error(f"Bybit Orderbook Error ({symbol}): {e}")
            return None
    
    async def fetch_recent_trades(self, symbol: str, limit: int = 50) -> Optional[List[Dict]]:
        """Fetch recent trades (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            
            formatted_trades = []
            for trade in trades:
                formatted_trades.append({
                    'timestamp': trade['timestamp'],
                    'price': float(trade['price']),
                    'amount': float(trade['amount']),
                    'side': trade['side'],
                    'value_usd': float(trade['price']) * float(trade['amount']),
                    'source': 'bybit'
                })
            
            return formatted_trades
        except Exception as e:
            logger.error(f"Bybit Trades Error ({symbol}): {e}")
            return None
    
    async def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch open interest (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            # Bybit open interest endpoint
            oi = await self.exchange.fetch_open_interest(symbol)
            
            return {
                'open_interest': float(oi['openInterest']) if oi else 0,
                'open_interest_value': float(oi.get('openInterestValue', 0)),
                'symbol': symbol,
                'source': 'bybit'
            }
        except Exception as e:
            logger.debug(f"Bybit OI Error ({symbol}): {e}")
            return None

    async def close(self):
        if self.exchange: await self.exchange.close()
