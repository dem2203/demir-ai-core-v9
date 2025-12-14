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
    
    # === Phase 29.3 Enhancements ===
    
    async def fetch_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List[Dict]]:
        """Fetch OHLCV candles (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            # Coinbase uses different symbol format (BTC-USD instead of BTC/USDT)
            cb_symbol = symbol.replace('/USDT', '/USD')
            
            ohlcv = await self.exchange.fetch_ohlcv(cb_symbol, timeframe, limit=limit)
            formatted_data = []
            
            for candle in ohlcv:
                formatted_data.append({
                    'symbol': symbol,
                    'timestamp': candle[0],
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'source': 'coinbase'
                })
            
            return formatted_data
        except Exception as e:
            logger.error(f"Coinbase Candles Error ({symbol}): {e}")
            return None
    
    async def fetch_orderbook(self, symbol: str, limit: int = 25) -> Optional[Dict]:
        """Fetch order book depth (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            cb_symbol = symbol.replace('/USDT', '/USD')
            orderbook = await self.exchange.fetch_order_book(cb_symbol, limit=limit)
            
            bid_volume = sum([price * amount for price, amount in orderbook['bids'][:limit]])
            ask_volume = sum([price * amount for price, amount in orderbook['asks'][:limit]])
            
            return {
                'bids': orderbook['bids'][:limit],
                'asks': orderbook['asks'][:limit],
                'bid_volume_usd': bid_volume,
                'ask_volume_usd': ask_volume,
                'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
                'source': 'coinbase'
            }
        except Exception as e:
            logger.error(f"Coinbase Orderbook Error ({symbol}): {e}")
            return None
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch full ticker data (Phase 29.3)"""
        if not self.exchange: await self.connect()
        if not self.exchange: return None
        
        try:
            cb_symbol = symbol.replace('/USDT', '/USD')
            ticker = await self.exchange.fetch_ticker(cb_symbol)
            
            return {
                'symbol': symbol,
                'last': float(ticker['last']),
                'bid': float(ticker.get('bid', 0) or 0),
                'ask': float(ticker.get('ask', 0) or 0),
                'volume_24h': float(ticker.get('quoteVolume', 0) or 0),
                'change_24h': float(ticker.get('percentage', 0) or 0),
                'source': 'coinbase'
            }
        except Exception as e:
            logger.error(f"Coinbase Ticker Error ({symbol}): {e}")
            return None

    async def close(self):
        if self.exchange: await self.exchange.close()
