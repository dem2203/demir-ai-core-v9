import asyncio
import logging
from typing import List
from datetime import datetime

# --- MODÜLLER ---
from src.config.settings import Config
from src.data_ingestion.market_data_manager import MarketDataManager
from src.brain.market_analyzer import MarketAnalyzer
from src.utils.logger import setup_logger
from src.utils.notifications import NotificationManager
from src.execution.paper_trader import PaperTrader # <-- YENİ MODÜL EKLENDİ

logger = logging.getLogger("DEMIR_AI_CORE_ENGINE")

class BotEngine:
    """
    DEMIR AI v11.3 - FULLY INTEGRATED ENGINE (PAPER TRADING EDITION)
    
    Görevi:
    1. Veri Çek (Crypto + Macro)
    2. Analiz Et (LSTM Brain)
    3. İşlem Yap (Paper Trader - Sanal Cüzdan)
    4. Bildir (Telegram + Dashboard)
    """
    
    def __init__(self):
        self.is_running = False
        logger.info("Initializing Sub-systems...")
        
        self.data_manager = MarketDataManager()
        self.analyzer = MarketAnalyzer()
        self.notifier = NotificationManager()
        self.paper_trader = PaperTrader() # <-- Sanal Broker Devrede
        
        self.loop_interval = 60 # 1 Dakika

    async def start(self):
        """Botu Başlatır"""
        logger.info(f"STARTING DEMIR AI v{Config.VERSION} - MODE: PAPER TRADING")
        
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
        
        # --- YENİ: Canlı Fiyatları PaperTrader'a bildir (Equity hesabı için) ---
        # Bu sayede işlem açmasak bile cüzdandaki varlıkların anlık değerini biliriz.
        current_prices = {}
        for data in market_data_list:
            if data and len(data) > 0:
                symbol = data[-1]['symbol'] # Son mumun sembolü
                price = data[-1]['close']   # Son kapanış fiyatı
                current_prices[symbol] = price
        
        self.paper_trader.get_portfolio_status(current_prices)
        # ---------------------------------------------------------------------
        
        tasks = []
        for data in market_data_list:
            # Her coin için analiz görevi oluştur
            tasks.append(self.analyze_and_execute(data))
            
        # Tüm analizleri paralel çalıştır
        await asyncio.gather(*tasks)

    async def analyze_and_execute(self, ticker_data: List[dict]):
        """
        TEK BİR COIN İÇİN ANALİZ VE İCRA
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # 1. ANALİZ KATMANI (BEYİN)
        signal = await self.analyzer.analyze_market(symbol, ticker_data)
        
        # Eğer kayda değer bir sinyal yoksa çık
        if not signal:
            return

        logger.info(f"SIGNAL DETECTED: {symbol} -> {signal['side']} (Conf: {signal['confidence']}%)")

        # 2. İCRA KATMANI (SANAL CÜZDAN)
        # Sinyali Paper Trader'a gönder, o karar versin (Bakiye var mı? Pozisyon var mı?)
        trade_executed = self.paper_trader.execute_trade(signal)
        
        # 3. BİLDİRİM KATMANI
        # Eğer işlem gerçekten yapıldıysa Telegram at
        if trade_executed:
            await self.notifier.send_signal(signal)
            logger.info(f"✅ PAPER TRADE EXECUTED: {signal['side']} {symbol}")
        else:
            # İşlem açılmadıysa (Bakiye yetersiz veya zaten pozisyon var) log düş
            logger.info(f"⏸️ Signal Valid but Trade Skipped (Already Open/No Balance): {symbol}")

    async def stop(self):
        """Güvenli Kapatma"""
        self.is_running = False
        await self.data_manager.shutdown()
        logger.info("SYSTEM SHUTDOWN COMPLETE.")
