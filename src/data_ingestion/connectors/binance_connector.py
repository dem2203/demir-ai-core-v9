import ccxt.async_support as ccxt
import asyncio
from typing import Dict, Optional
import logging
from src.config.settings import Config
from src.validation.validator import SignalValidator

logger = logging.getLogger("BINANCE_CONNECTOR")

class BinanceConnector:
    """
    Profesyonel Binance Bağlantı Yöneticisi.
    Railway Env değişkenlerini kullanır. Hardcoded veri İÇERMEZ.
    """
    
    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.api_secret = Config.BINANCE_API_SECRET
        self.exchange = None
        
        # Borsa ayarları
        self.exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,  # Ban yememek için otomatik hız sınırı
            'options': {
                'defaultType': 'future',  # Vadeli işlemler (Futures) modu
            }
        }

    async def connect(self):
        """Borsaya asenkron bağlantı başlatır."""
        try:
            self.exchange = ccxt.binance(self.exchange_config)
            # Bağlantı testi: Server saatini çek
            time = await self.exchange.fetch_time()
            logger.info(f"CONNECTED: Binance System Time: {time}")
            
            # Piyasaları yükle (Symbol map)
            await self.exchange.load_markets()
            logger.info("MARKETS LOADED: Binance symbols ready.")
            
        except Exception as e:
            logger.critical(f"CONNECTION FAILED: Could not connect to Binance. Error: {e}")
            raise e

    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Tek bir parite için CANLI fiyat verisi çeker.
        Örn: symbol='BTC/USDT'
        """
        if not self.exchange:
            await self.connect()
            
        try:
            # 1. Gerçek veriyi çek
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Veriyi standart formata getir
            processed_data = {
                'source': 'binance',
                'symbol': symbol,
                'price': float(ticker['last']),
                'volume': float(ticker['quoteVolume']),
                'timestamp': int(ticker['timestamp']),
                'high': float(ticker['high']),
                'low': float(ticker['low']),
            }
            
            # 2. VALIDATION KATMANI (Az önce yazdığımız Polis Kontrolü)
            if SignalValidator.validate_incoming_data(processed_data):
                return processed_data
            else:
                logger.warning(f"DATA REJECTED: Validation failed for {symbol}")
                return None

        except Exception as e:
            logger.error(f"FETCH ERROR: {symbol} - {str(e)}")
            return None

    async def close(self):
        """Bağlantıyı güvenli kapatır."""
        if self.exchange:
            await self.exchange.close()