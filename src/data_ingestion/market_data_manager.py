import asyncio
from typing import List, Dict
import logging
from src.data_ingestion.connectors.binance_connector import BinanceConnector

logger = logging.getLogger("MARKET_DATA_MANAGER")

class MarketDataManager:
    """
    Tüm borsa bağlantılarını yöneten Merkezi Veri Üssü.
    """
    
    def __init__(self):
        self.binance = BinanceConnector()
        # Takip edilecek pariteler
        self.active_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"]

    async def initialize(self):
        logger.info("Initializing Exchange Connections...")
        await self.binance.connect()
        logger.info("ALL SYSTEMS ONLINE.")

    async def get_live_market_snapshot(self) -> List[List[Dict]]:
        """
        Her parite için son 100 mumluk veri setini çeker.
        Dönüş Formatı: [ [BTC_Mumlari], [ETH_Mumlari], ... ]
        """
        tasks = []
        for pair in self.active_pairs:
            # Artık Ticker değil, Candle (Mum) çekiyoruz
            tasks.append(self.binance.fetch_candles(pair, timeframe='1h', limit=100))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hatalı çekimleri filtrele
        valid_data = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        count = len(valid_data)
        logger.info(f"SNAPSHOT COMPLETE: Collected data for {count} pairs.")
        return valid_data

    async def shutdown(self):
        await self.binance.shutdown()