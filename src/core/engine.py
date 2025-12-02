import asyncio
import logging
from typing import List
from datetime import datetime

# --- MODÜLLERİMİZİ ÇAĞIRIYORUZ ---
from src.config.settings import Config
from src.data_ingestion.market_data_manager import MarketDataManager
from src.brain.market_analyzer import MarketAnalyzer
from src.execution.order_manager import OrderManager
from src.utils.logger import setup_logger

logger = logging.getLogger("DEMIR_AI_CORE_ENGINE")

class BotEngine:
    """
    DEMIR AI v9.0 - MAIN ORCHESTRATOR
    """
    
    def __init__(self):
        self.is_running = False
        
        # --- Alt Sistemlerin Yüklenmesi ---
        logger.info("Initializing Sub-systems...")
        self.data_manager = MarketDataManager()
        self.analyzer = MarketAnalyzer()
        self.order_manager = OrderManager()
        
        # Takip Edilecek Coinler (Dinamik)
        self.target_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        
        self.loop_interval = 60 # Saniye

    async def start(self):
        """Botu Başlatır"""
        logger.info(f"STARTING DEMIR AI v{Config.VERSION} - ENVIRONMENT: {Config.ENVIRONMENT}")
        
        # Borsa Bağlantılarını Aç
        await self.data_manager.initialize()
        
        self.is_running = True
        await self.run_forever()

    async def run_forever(self):
        """Sonsuz Ana Döngü"""
        while self.is_running:
            start_time = datetime.now()
            logger.info("--- NEW ANALYSIS CYCLE STARTED ---")
            
            try:
                # Tüm coinleri asenkron (aynı anda) analiz et
                await self.process_market_cycle()
                
            except Exception as e:
                logger.critical(f"CRITICAL ENGINE FAILURE: {e}")
                await asyncio.sleep(5) 
            
            # Döngü süresini hesapla ve bekle
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, self.loop_interval - elapsed)
            logger.info(f"Cycle completed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)

    async def process_market_cycle(self):
        """
        Tek bir analiz turu.
        """
        # Toplu Veri Çekme (OHLCV Listesi)
        market_data_list = await self.data_manager.get_live_market_snapshot()
        
        current_balance = 10000.0 
        
        tasks = []
        for data in market_data_list:
            # Her coin için analiz görevi
            task = self.analyze_and_execute(data, current_balance)
            tasks.append(task)
            
        # Tüm analizleri paralel çalıştır
        await asyncio.gather(*tasks)

    async def analyze_and_execute(self, ticker_data: List[dict], balance: float):
        """
        TEK BİR COIN İÇİN KARAR MEKANİZMASI
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # --- A. ANALİZ KATMANI ---
        signal = await self.analyzer.analyze_market(symbol, ticker_data)
        
        if not signal:
            return

        logger.info(f"SIGNAL DETECTED: {symbol} -> {signal['side']} (Conf: {signal['confidence']}%)")

        # --- B. EXECUTION KATMANI ---
        # ATR hesabı için son mumun verilerini kullan
        last_candle = ticker_data[-1]
        atr_value = last_candle.get('high') - last_candle.get('low') 
        
        order = await self.order_manager.prepare_order(signal, balance, atr_value)
        
        if order:
            logger.info(f"🚀 EXECUTING ORDER: {order}")

    async def stop(self):
        """Güvenli Kapatma"""
        self.is_running = False
        await self.data_manager.shutdown()
        logger.info("SYSTEM SHUTDOWN COMPLETE.")