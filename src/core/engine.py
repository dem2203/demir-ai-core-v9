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
from src.core.risk_manager import RiskManager
from src.brain.strategy_selector import StrategySelector
from src.execution.hedge_manager import HedgeManager
from src.brain.anomaly_detector import AnomalyDetector
from src.utils.alert_manager import AlertManager

# PHASE 11: Advanced Risk & Performance
from src.core.risk_shield import RiskShield
from src.core.performance_tracker import PerformanceTracker
from src.brain.exit_strategy import ExitStrategy 

logger = logging.getLogger("DEMIR_AI_CORE_ENGINE")

class BotEngine:
    """
    DEMIR AI v22.0 - OPPORTUNITY HUNTER ENGINE
    
    Features:
    1. Real-Time Anomaly Detection
    2. Order Flow Imbalance Monitoring
    3. Funding Rate Divergence Alerts
    4. All previous institutional features
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
        
        # 4. İcra Sistemi (Eller - Sanal/Advisory)
        self.paper_trader = PaperTrader()
        
        # 5. Risk Yöneticisi (Kasa)
        self.risk_manager = RiskManager()
        
        # 6. PHASE 4B: Adaptive Strategy & Hedging
        self.strategy_selector = StrategySelector()
        self.hedge_manager = HedgeManager()
        
        # 7. PHASE 6: Opportunity Hunter
        self.anomaly_detectors = {}  # One per symbol
        self.alert_manager = AlertManager()
        
        # 8. PHASE 11: Advanced Risk & Performance
        self.risk_shield = RiskShield()
        self.performance_tracker = PerformanceTracker()
        self.exit_strategy = ExitStrategy()
        self.cycle_count = 0  # For periodic performance tracking
        self.last_heartbeat_time = datetime.now() # Phase 21: Heartbeat
        
        logger.info("✅ All Sub-systems Initialized Successfully.")

    async def start(self):
        """
        Botu başlatır ve ana döngüye sokar.
        """
        logger.info(f"🌍 ENVIRONMENT: {Config.ENVIRONMENT}")
        logger.info(f"🦅 ACTIVE STRATEGY: Hybrid Intelligence (LSTM + RL + Macro)")
        logger.info(f"💼 TRADING MODE: ADVISORY (Signals Only)")
        
        # Borsa Bağlantılarını Başlat
        try:
            await self.data_manager.initialize()
        except Exception as e:
            logger.critical(f"❌ CRITICAL: Failed to connect to Exchange: {e}")
            return # Bağlantı yoksa başlama

        # Telegram'a "Ben Başladım" mesajı at
        await self.notifier.send_message_raw("🦅 **DEMIR AI ONLINE**\nSistem başlatıldı. Beyin modelleri kontrol ediliyor...")
        
        # Ensure AI Brain is Trained (Cold Start Fix)
        await self.analyzer.ensure_active_brain()
        
        self.is_running = True
        await self.run_forever()

    async def run_forever(self):
        """
        Sonsuz Yaşam Döngüsü.
        """
        error_count = 0
        
        while self.is_running:
            start_time = datetime.now()
            logger.info(f"--- ⏳ CYCLE START: {start_time.strftime('%H:%M:%S')} ---")
            
            try:
                # Ana İşlem Bloğu
                await self.process_market_cycle()
                
                # Phase 21: Hourly Heartbeat
                if (datetime.now() - self.last_heartbeat_time).total_seconds() > 3600:
                    await self.notifier.send_message_raw(f"🦅 **STATUS REPORT**\nSystem Active. Scanning markets...\nTime: {datetime.now().strftime('%H:%M')}")
                    self.last_heartbeat_time = datetime.now()
                    logger.info("💓 Heartbeat sent to Telegram.")
                    
                error_count = 0 
                
            except Exception as e:
                error_count += 1
                logger.error(f"⚠️ CYCLE ERROR ({error_count}): {str(e)}")
                
                if error_count > 5:
                    logger.critical("🚨 Too many consecutive errors! Pausing for 5 minutes.")
                    await self.notifier.send_message_raw("⚠️ **SİSTEM UYARISI:** Üst üste hata alındı. 5dk soğuma moduna geçiliyor.")
                    await asyncio.sleep(300)
                else:
                    await asyncio.sleep(5) 
            
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
        market_data_list = await self.data_manager.get_live_market_snapshot()
        
        if not market_data_list:
            logger.warning("No market data received this cycle. Waiting for next cycle.")
            return

        # ADIM 2: Cüzdan Durumunu Güncelle (Live Portfolio Dashboard için)
        current_prices = {}
        for data in market_data_list:
            if data and len(data) > 0:
                symbol = data[-1]['symbol']
                price = data[-1]['close']
                current_prices[symbol] = price
        
        self.paper_trader.get_portfolio_status(current_prices)
        
        # ADIM 3: Her Coin İçin Analiz Yap (Paralel İşlem)
        tasks = []
        for data in market_data_list:
            if data:
                tasks.append(self.analyze_and_execute(data))
            
        await asyncio.gather(*tasks)
        
        # ADIM 4: Performance Tracking (every 10 cycles)
        self.cycle_count += 1
        if self.cycle_count % 10 == 0:
            logger.info("📊 Calculating performance metrics...")
            self.performance_tracker.calculate_metrics()

    async def analyze_and_execute(self, ticker_data: List[dict]):
        """
        Tek bir coinin kaderini belirleyen fonksiyon.
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # --- BEYİN KATMANI (Brain Layer) ---
        signal, snapshot = await self.analyzer.analyze_market(symbol, ticker_data)
        
        if not signal:
            return

        # Sinyal bulundu! Logla.
        logger.info(f"🎯 SIGNAL FOUND: {symbol} | {signal['side']} | Conf: {signal['confidence']:.2f}% | Reason: {signal.get('reason')}")

        # --- İCRA KATMANI (Execution Layer - Advisory) ---
        # PHASE 11: Risk Shield - Check cooling mode & adjust Kelly
        adjusted_kelly = self.risk_shield.adjust_kelly(signal['confidence'])
        
        if adjusted_kelly == 0:
            logger.warning(f"❌ TRADE BLOCKED: Risk Shield in Cooling Mode")
            return
        
        # Apply adjusted Kelly
        kelly_size = self.risk_manager.calculate_kelly_size(adjusted_kelly)
        signal['kelly_size'] = kelly_size
        
        logger.info(f"💰 KELLY SUGGESTION: Risk {kelly_size}% of Equity (Shield: {self.risk_shield.get_status()['risk_level']})")

        # Advisory modunda olduğumuz için sadece 'Paper Trade' yapıyoruz ve bildiriyoruz.
        # Gerçek borsaya emir GİTMİYOR.
        trade_executed = self.paper_trader.execute_trade(signal)
        
        if trade_executed:
            await self.notifier.send_signal(signal, snapshot)  # Pass snapshot for Precision Filter
            logger.info(f"✅ SIGNAL BROADCASTED: {signal['side']} {symbol}")
        else:
            logger.info(f"⏸️ SIGNAL SKIPPED: {symbol} (Already in position or Insufficient Funds)")

    async def stop(self):
        """Sistemi güvenli kapatır."""
        self.is_running = False
        await self.data_manager.shutdown()
        await self.notifier.send_message_raw("🛑 **DEMIR AI KAPATILIYOR**\nBakım veya yeniden başlatma.")
        logger.info("SYSTEM SHUTDOWN COMPLETE.")
