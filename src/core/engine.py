import asyncio
import logging
import os
import json
from typing import List, Dict, Optional
from datetime import datetime

# --- MODÜL İMPORTLARI ---
from src.config.settings import Config
from src.data_ingestion.market_data_manager import MarketDataManager
from src.brain.market_analyzer import MarketAnalyzer
from src.utils.logger import setup_logger
from src.utils.notifications import NotificationManager
from src.execution.paper_trader import PaperTrader 

# İleride Gerçek İşlem açmak istersek diye bu modülü pasif olarak tutuyoruz
# from src.execution.order_manager import OrderManager 

logger = logging.getLogger("DEMIR_AI_CORE_ENGINE")

class BotEngine:
    """
    DEMIR AI v13.0 - ENTERPRISE CORE ENGINE
    
    Bu motor, sistemin kalbidir. Tüm organları (Veri, Beyin, Cüzdan, İletişim) yönetir.
    
    YETENEKLER:
    1. Çoklu Varlık Yönetimi (BTC, ETH, LTC...)
    2. Hibrit Zeka Entegrasyonu (LSTM + RL + Makro)
    3. Paper Trading (Sanal Cüzdan) Yönetimi
    4. Kesintisiz Çalışma (Fault Tolerance)
    """
    
    def __init__(self):
        self.is_running = False
        self.loop_interval = 60 # Analiz döngüsü (saniye)
        
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("🚀 INITIALIZING DEMIR AI CORE SYSTEMS...")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 1. Veri Yöneticisi (Gözler)
        self.data_manager = MarketDataManager()
        
        # 2. Piyasa Analisti (Beyin)
        self.analyzer = MarketAnalyzer()
        
        # 3. Bildirim Sistemi (Ses)
        self.notifier = NotificationManager()
        
        # 4. İcra Sistemi (Eller - Sanal)
        self.paper_trader = PaperTrader()
        
        # 5. Gerçek İcra Sistemi (Şu an Devre Dışı - Future Use)
        # self.real_trader = OrderManager()
        
        logger.info("✅ All Sub-systems Initialized Successfully.")

    async def start(self):
        """
        Botu başlatır ve ana döngüye sokar.
        """
        logger.info(f"🌍 ENVIRONMENT: {Config.ENVIRONMENT}")
        logger.info(f"🦅 ACTIVE STRATEGY: Hybrid Intelligence (LSTM + RL + Macro)")
        logger.info(f"💼 TRADING MODE: PAPER TRADING (Simulated Wallet)")
        
        # Borsa Bağlantılarını Başlat
        try:
            await self.data_manager.initialize()
        except Exception as e:
            logger.critical(f"❌ CRITICAL: Failed to connect to Exchange: {e}")
            return # Bağlantı yoksa başlama

        # Telegram'a "Ben Başladım" mesajı at
        await self.notifier.send_message_raw("🦅 **DEMIR AI ONLINE**\nSistem başlatıldı. Piyasa taranıyor...")
        
        self.is_running = True
        await self.run_forever()

    async def run_forever(self):
        """
        Sonsuz Yaşam Döngüsü. Hata olsa bile (internet kopması vb.) kendini toparlar.
        """
        error_count = 0
        
        while self.is_running:
            start_time = datetime.now()
            logger.info(f"--- ⏳ CYCLE START: {start_time.strftime('%H:%M:%S')} ---")
            
            try:
                # Ana İşlem Bloğu
                await self.process_market_cycle()
                error_count = 0 # Başarılı turda hata sayacını sıfırla
                
            except Exception as e:
                error_count += 1
                logger.error(f"⚠️ CYCLE ERROR ({error_count}): {str(e)}")
                
                if error_count > 5:
                    logger.critical("🚨 Too many consecutive errors! Pausing for 5 minutes.")
                    await self.notifier.send_message_raw("⚠️ **SİSTEM UYARISI:** Üst üste hata alındı. 5dk soğuma moduna geçiliyor.")
                    await asyncio.sleep(300)
                else:
                    await asyncio.sleep(5) # Kısa bekleme
            
            # Döngü Süresi Kontrolü
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, self.loop_interval - elapsed)
            
            if sleep_time > 0:
                logger.info(f"✅ Cycle finished in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            else:
                logger.warning(f"🐢 System is lagging! Cycle took {elapsed:.2f}s (Target: {self.loop_interval}s)")

    async def process_market_cycle(self):
        """
        Tek bir analiz turunun adımları.
        """
        # ADIM 1: Verileri Topla (Kripto + Makro)
        # MarketDataManager artık 500 mumluk veriyi çekiyor.
        market_data_list = await self.data_manager.get_live_market_snapshot()
        
        if not market_data_list:
            logger.warning("No market data received this cycle.")
            return

        # ADIM 2: Cüzdan Durumunu Güncelle (Live Portfolio Dashboard için)
        # Anlık fiyatları çekip PaperTrader'a bildiriyoruz ki "Total Equity" hesaplasın.
        current_prices = {}
        for data in market_data_list:
            if data and len(data) > 0:
                symbol = data[-1]['symbol']
                price = data[-1]['close']
                current_prices[symbol] = price
        
        # Dashboard'daki 'Live Portfolio' sekmesi bu veriyi kullanır
        self.paper_trader.get_portfolio_status(current_prices)
        
        # ADIM 3: Her Coin İçin Analiz Yap (Paralel İşlem)
        tasks = []
        for data in market_data_list:
            if data:
                tasks.append(self.analyze_and_execute(data))
            
        # Tüm analizleri aynı anda çalıştır (Hız artışı)
        await asyncio.gather(*tasks)

    async def analyze_and_execute(self, ticker_data: List[dict]):
        """
        Tek bir coinin kaderini belirleyen fonksiyon.
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # --- BEYİN KATMANI (Brain Layer) ---
        # MarketAnalyzer burada LSTM, RL ve Makro verileri birleştirip karar verir.
        # Ayrıca Dashboard verisini (dxy, vix, prediction) JSON dosyasına yazar.
        signal = await self.analyzer.analyze_market(symbol, ticker_data)
        
        # Eğer Nötr ise veya Güven yetersizse işlem yapma
        if not signal:
            return

        # Sinyal bulundu! Logla.
        logger.info(f"🎯 SIGNAL FOUND: {symbol} | {signal['side']} | Conf: {signal['confidence']:.2f}% | Reason: {signal.get('reason')}")

        # --- İCRA KATMANI (Execution Layer) ---
        # Paper Trader'a gönder. Bakiye yetiyorsa sanal işlem açar.
        trade_executed = self.paper_trader.execute_trade(signal)
        
        if trade_executed:
            # İşlem başarılıysa Telegram'dan müjdeyi ver
            await self.notifier.send_signal(signal)
            logger.info(f"✅ ACTION TAKEN: {signal['side']} {symbol} added to Portfolio.")
        else:
            # Bakiye yetersiz veya zaten pozisyon var
            logger.info(f"⏸️ ACTION SKIPPED: {symbol} (Already in position or Insufficient Funds)")

    async def stop(self):
        """Sistemi güvenli kapatır."""
        self.is_running = False
        await self.data_manager.shutdown()
        await self.notifier.send_message_raw("🛑 **DEMIR AI KAPATILIYOR**\nBakım veya yeniden başlatma.")
        logger.info("SYSTEM SHUTDOWN COMPLETE.")
