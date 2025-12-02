import asyncio
from typing import List, Dict
import logging
from src.data_ingestion.connectors.binance_connector import BinanceConnector
# İleride BybitConnector ve CoinbaseConnector buraya eklenecek

logger = logging.getLogger("MARKET_DATA_MANAGER")

class MarketDataManager:
    """
    Tüm borsa bağlantılarını yöneten Merkezi Veri Üssü.
    Tek komutla tüm piyasadan veri toplar.
    """
    
    def __init__(self):
        self.binance = BinanceConnector()
        # self.bybit = BybitConnector() # Sırada
        self.active_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"]

    async def initialize(self):
        """Tüm borsalara bağlanır."""
        logger.info("Initializing Exchange Connections...")
        await self.binance.connect()
        # await self.bybit.connect()
        logger.info("ALL SYSTEMS ONLINE.")

    async def get_live_market_snapshot(self) -> List[Dict]:
        """
        Takipteki tüm coinlerin anlık fotoğrafını çeker.
        """
        tasks = []
        for pair in self.active_pairs:
            # Her parite için bir görev oluştur (Parallel Processing)
            tasks.append(self.binance.fetch_ticker(pair))
        
        # Hepsini aynı anda çalıştır
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Hataları temizle ve sadece geçerli verileri döndür
        valid_data = [r for r in results if r is not None and not isinstance(r, Exception)]
        
        logger.info(f"SNAPSHOT COMPLETE: Collected {len(valid_data)} valid data points.")
        return valid_data

    async def shutdown(self):
        await self.binance.close()