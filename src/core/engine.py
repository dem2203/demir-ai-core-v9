import asyncio
import logging
from typing import List
from datetime import datetime

# --- MODÜLLERİMİZİ ÇAĞIRIYORUZ ---
from src.config.settings import Config
from src.data_ingestion.market_data_manager import MarketDataManager
from src.brain.market_analyzer import MarketAnalyzer
from src.utils.logger import setup_logger
from src.utils.notifications import NotificationManager # <-- YENİ EKLENDİ

logger = logging.getLogger("DEMIR_AI_CORE_ENGINE")

class BotEngine:
    """
    DEMIR AI v10.1 - ANALYST ENGINE
    
    Bu motor artık işlem açmaz (Execution Layer devre dışı).
    Bunun yerine:
    1. Piyasayı izler.
    2. AI ile analiz eder.
    3. Dashboard verisini günceller.
    4. Telegram üzerinden Sinyal/Rapor gönderir.
    """
    
    def __init__(self):
        self.is_running = False
        
        # --- Alt Sistemlerin Yüklenmesi ---
        logger.info("Initializing Sub-systems...")
        
        self.data_manager = MarketDataManager()
        self.analyzer = MarketAnalyzer()
        self.notifier = NotificationManager() # <-- OrderManager yerine geldi
        
        # Takip Edilecek Coinler (Otomatik algılamaya geçene kadar manuel liste)
        self.target_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "AVAX/USDT"]
        
        self.loop_interval = 60 # Saniye (1 Dakika)

    async def start(self):
        """Botu Başlatır"""
        logger.info(f"STARTING DEMIR AI v{Config.VERSION} - MODE: AI CO-PILOT")
        
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
                # Hata olsa bile botu durdurma, sadece logla ve devam et
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
        
        # Analist modunda bakiyeye ihtiyacımız yok ama fonksiyon yapısını bozmamak için dummy veri.
        current_balance = 0.0 
        
        tasks = []
        for data in market_data_list:
            # Her coin için analiz görevi oluştur
            task = self.analyze_and_report(data)
            tasks.append(task)
            
        # Tüm analizleri paralel çalıştır
        await asyncio.gather(*tasks)

    async def analyze_and_report(self, ticker_data: List[dict]):
        """
        TEK BİR COIN İÇİN ANALİZ VE RAPORLAMA
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # --- A. ANALİZ KATMANI ---
        # AI burada devreye giriyor, analiz yapıyor ve dashboard için veri kaydediyor.
        signal = await self.analyzer.analyze_market(symbol, ticker_data)
        
        # Eğer kayda değer bir sinyal yoksa çık
        if not signal:
            return

        logger.info(f"SIGNAL DETECTED: {symbol} -> {signal['side']} (Conf: {signal['confidence']}%)")

        # --- B. BİLDİRİM KATMANI (EXECUTION YOK) ---
        # Sinyali Telegram'a gönder
        await self.notifier.send_signal(signal)
        
        logger.info(f"📨 TELEGRAM REPORT SENT: {signal['side']} {symbol}")

    async def stop(self):
        """Güvenli Kapatma"""
        self.is_running = False
        await self.data_manager.shutdown()
        logger.info("SYSTEM SHUTDOWN COMPLETE.")