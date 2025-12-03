import asyncio
from typing import List, Dict
import logging
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.config.settings import Config  # <-- Config dosyasını çağırıyoruz

logger = logging.getLogger("MARKET_DATA_MANAGER")

class MarketDataManager:
    """
    Tüm borsa bağlantılarını yöneten Merkezi Veri Üssü.
    İzlenecek coinleri Config dosyasından alır.
    """
    
    def __init__(self):
        self.binance = BinanceConnector()
        # LİSTEYİ MERKEZDEN ÇEKİYORUZ
        self.active_pairs = Config.TARGET_COINS

    async def initialize(self):
        logger.info("Initializing Exchange Connections...")
        await self.binance.connect()
        logger.info(f"ALL SYSTEMS ONLINE. Tracking {len(self.active_pairs)} Assets: {self.active_pairs}")

    async def get_live_market_snapshot(self) -> List[List[Dict]]:
        """
        Her parite için son 500 mumluk (saatlik) veri setini çeker.
        """
        tasks = []
        for pair in self.active_pairs:
            # Limit 500: Matematik motorunun (Hurst, Ichimoku) veri yemesi için gerekli pay.
            tasks.append(self.binance.fetch_candles(pair, timeframe='1h', limit=500))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hatalı çekimleri filtrele
        valid_data = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        logger.info(f"SNAPSHOT COMPLETE: Collected data for {len(valid_data)} pairs.")
        return valid_data

    async def shutdown(self):
        await self.binance.shutdown()
