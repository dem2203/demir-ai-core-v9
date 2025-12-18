import asyncio
import logging
import os
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta

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

# PHASE 22: Market Correlation & Derivatives
from src.data_ingestion.correlation_connector import CorrelationConnector
from src.data_ingestion.derivatives_connector import DerivativesConnector

# PHASE 30: Money Flow Analysis
from src.data_ingestion.money_flow_analyzer import MoneyFlowAnalyzer

# PHASE 32: Predictive Intelligence
from src.brain.sentiment_analyzer import SentimentAnalyzer
from src.brain.predictive_analyzer import PredictiveAnalyzer

# PHASE 33: Smart Notification System (15min scan)
from src.brain.market_intelligence import MarketIntelligence

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
        # Phase 21: Heartbeat - Set to 1 hour ago so first heartbeat sends immediately
        self.last_heartbeat_time = datetime.now() - timedelta(hours=1)
        self.latest_prices = {} # For Heartbeat
        
        # 9. PHASE 22: Market Correlation & Derivatives
        self.correlation_connector = CorrelationConnector()
        self.derivatives_connector = DerivativesConnector()
        self.market_correlations = {}  # Cross-asset data
        self.derivatives_data = {}     # OI, L/S ratio, etc.
        
        # 10. PHASE 30: Money Flow Analysis (Mikabot-style)
        self.money_flow_analyzer = MoneyFlowAnalyzer()
        
        # 11. PHASE 32: Predictive Intelligence - Fear & Greed
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # 12. PHASE 32.5: Predictive Analyzer - Leading Indicators
        self.predictive_analyzer = PredictiveAnalyzer()
        
        # 13. PHASE 33: Smart Notification - 15min scan, 1h fallback
        self.market_intelligence = MarketIntelligence()
        self.all_snapshots = {}  # Store all coin snapshots for intelligence
        
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
        await self.notifier.send_message_raw("🦅 **DEMIR AI ONLINE**\nSistem başlatıldı. Beyin eğitimi arka planda başlatılıyor...")
        
        # Ensure AI Brain is Trained (Non-Blocking Background Task)
        # Fixes 502 Error: Training takes time, so we don't block startup.
        asyncio.create_task(self.analyzer.ensure_active_brain())
        
        self.is_running = True
        await self.run_forever()

    async def run_forever(self):
        """
        Sonsuz Yaşam Döngüsü.
        """
        error_count = 0
        last_error = ""
        
        while self.is_running:
            start_time = datetime.now()
            logger.info(f"--- ⏳ CYCLE START: {start_time.strftime('%H:%M:%S')} ---")
            
            try:
                # Ana İşlem Bloğu
                await self.process_market_cycle()
                
                # Phase 31: Check Telegram Commands (inout, status, help)
                await self.notifier.check_telegram_commands(self.money_flow_analyzer)
                
                # Phase 93+94: Check signal gate TP/SL and position risks
                await self.notifier.check_and_update_signals()  # TP/SL vuruldu mu?
                await self.notifier.check_active_position_risks()  # Risk var mı?
                
                # Phase 21: Hourly Heartbeat
                if (datetime.now() - self.last_heartbeat_time).total_seconds() > 3600:
                    await self.notifier.send_heartbeat(self.latest_prices)
                    
                    # Phase 30: Money Flow Report (Mikabot-style)
                    try:
                        money_flow_data = await self.money_flow_analyzer.get_market_money_flow()
                        await self.notifier.send_money_flow_report(money_flow_data)
                        logger.info("📊 Money Flow report sent to Telegram.")
                    except Exception as mf_err:
                        logger.error(f"Money Flow report failed: {mf_err}")
                    
                    self.last_heartbeat_time = datetime.now()
                    logger.info("💓 Heartbeat sent to Telegram.")
                    
                error_count = 0  # Reset on success
                
            except Exception as e:
                error_count += 1
                last_error = str(e)[:200]  # Truncate long errors
                logger.error(f"⚠️ CYCLE ERROR ({error_count}): {last_error}")
                
                # Import traceback for detailed error info
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Traceback: {tb[:500]}")
                
                if error_count > 5:
                    logger.critical("🚨 Too many consecutive errors! Pausing for 5 minutes.")
                    # IMPROVED: Send actual error to Telegram for debugging
                    error_msg = (
                        f"⚠️ **SİSTEM UYARISI**\n"
                        f"━━━━━━━━━━━━━━\n"
                        f"Üst üste {error_count} hata alındı.\n"
                        f"5dk soğuma moduna geçiliyor.\n\n"
                        f"**Son Hata:**\n`{last_error}`"
                    )
                    await self.notifier.send_message_raw(error_msg)
                    await asyncio.sleep(300)
                    error_count = 0  # Reset after cooling
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
        
        self.latest_prices = current_prices
        self.paper_trader.get_portfolio_status(current_prices)
        
        # ADIM 2.5: Correlation & Derivatives Data (PHASE 22)
        try:
            self.market_correlations = self.correlation_connector.fetch_all()
            logger.info(f"📊 Correlations: BTC.D={self.market_correlations.get('btc_dominance', 0)}%, Gold=${self.market_correlations.get('gold', 0)}")
        except Exception as e:
            logger.warning(f"Correlation fetch failed: {e}")
        
        try:
            # Fetch derivatives for first symbol
            first_symbol = list(current_prices.keys())[0].replace('/', '') if current_prices else 'BTCUSDT'
            self.derivatives_data = self.derivatives_connector.fetch_all_derivatives(first_symbol)
            logger.info(f"📊 Derivatives: OI={self.derivatives_data.get('open_interest', 0):.0f}, L/S={self.derivatives_data.get('long_short_ratio', 1):.2f}")
        except Exception as e:
            logger.warning(f"Derivatives fetch failed: {e}")
        
        # PHASE 32: Fetch Fear & Greed Index
        try:
            self.sentiment_data = await self.sentiment_analyzer.get_fear_greed()
            fg_index = self.sentiment_data.get('fear_greed_index', 50)
            logger.info(f"😰 Fear & Greed: {fg_index} ({self.sentiment_data.get('fear_greed_label', 'Neutral')})")
        except Exception as e:
            logger.warning(f"Sentiment fetch failed: {e}")
            self.sentiment_data = {'fear_greed_index': 50, 'fear_greed_label': 'Neutral'}
        
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
        
        # ======================================
        # ADIM 5: SMART NOTIFICATION - 15dk Fırsat Taraması
        # ======================================
        try:
            # 15 dakika doldu mu kontrol et
            if self.market_intelligence.should_run_15min_check():
                logger.info("🔍 15dk Fırsat/Risk Taraması Başlıyor...")
                
                # Tüm snapshot'ları topla (dashboard_data.json'dan)
                import json
                if os.path.exists("dashboard_data.json"):
                    with open("dashboard_data.json", 'r') as f:
                        self.all_snapshots = json.load(f)
                
                # Fırsat/Risk tara
                opportunities = self.market_intelligence.scan_for_opportunities(self.all_snapshots)
                
                if opportunities:
                    # Fırsat bulundu - hemen bildir!
                    report = self.market_intelligence.format_opportunity_report(opportunities)
                    await self.notifier.send_message_raw(report)
                    logger.info(f"🎯 {len(opportunities)} fırsat/risk Telegram'a gönderildi!")
                else:
                    logger.info("✓ 15dk tarama tamamlandı - önemli fırsat/risk yok")
            
            # 1 saat fırsat bulunamadı mı kontrol et
            if self.market_intelligence.should_send_hourly_fallback():
                logger.info("📊 Saatlik Durum Özeti Gönderiliyor...")
                
                # Canlı derivatives data
                live_data = {
                    'open_interest': self.derivatives_data.get('open_interest', 0),
                    'long_short_ratio': self.derivatives_data.get('long_short_ratio', 0),
                    'btc_dominance': self.market_correlations.get('btc_dominance', 0)
                }
                
                hourly_report = self.market_intelligence.format_hourly_status(self.all_snapshots, live_data)
                await self.notifier.send_message_raw(hourly_report)
                logger.info("✅ Saatlik özet gönderildi!")
                
        except Exception as e:
            logger.warning(f"Market Intelligence scan failed: {e}")

    async def analyze_and_execute(self, ticker_data: List[dict]):
        """
        Tek bir coinin kaderini belirleyen fonksiyon.
        """
        if not ticker_data or len(ticker_data) == 0:
            return

        symbol = ticker_data[0]['symbol']
        
        # --- BEYİN KATMANI (Brain Layer) ---
        result = await self.analyzer.analyze_market(symbol, ticker_data)
        
        # FIX: Handle case where analyze_market returns None
        if result is None:
            logger.warning(f"analyze_market returned None for {symbol}")
            return
        
        signal, snapshot = result
        
        # PHASE 32: Inject derivatives and sentiment data into snapshot for early_warning
        if snapshot:
            snapshot['derivatives'] = getattr(self, 'derivatives_data', {})
            snapshot['sentiment'] = getattr(self, 'sentiment_data', {})
            
            # CRITICAL: Save snapshot to dashboard_data.json for dashboard display
            try:
                import json
                import os
                dashboard_path = "dashboard_data.json"
                
                # Load existing data
                if os.path.exists(dashboard_path):
                    with open(dashboard_path, 'r') as f:
                        db = json.load(f)
                else:
                    db = {}
                
                # Update with current snapshot
                db[snapshot.get('symbol', 'UNKNOWN')] = snapshot
                
                # Save back
                with open(dashboard_path, 'w') as f:
                    json.dump(db, f, indent=2, default=str)
                    
                logger.debug(f"Dashboard data saved for {snapshot.get('symbol')}")
            except Exception as e:
                logger.warning(f"Failed to save dashboard data: {e}")
        
        # --- PHASE 32.5: PREDICTIVE SIGNALS (Leading Indicators) ---
        # Check for predictive signals FIRST - these are more valuable than lagging indicators
        try:
            current_price = ticker_data[-1].get('close', 0) if ticker_data else 0
            predictive_signal = await self.predictive_analyzer.analyze_predictive_signals(symbol, current_price)
            
            if predictive_signal.get('has_signal'):
                # Format and send predictive signal
                formatted_msg = self.predictive_analyzer.format_predictive_signal(predictive_signal)
                if formatted_msg:
                    await self.notifier.send_message_raw(formatted_msg)
                    logger.info(f"🔮 PREDICTIVE SIGNAL: {symbol} {predictive_signal['direction']} - {len(predictive_signal.get('reasons', []))} confirmations")
        except Exception as e:
            logger.warning(f"Predictive analysis failed for {symbol}: {e}")
        
        # --- PROAKTİF ERKEN UYARI (Early Warning) ---
        # Send early warnings to Telegram BEFORE signal is generated
        if snapshot and snapshot.get('early_warnings'):
            warnings = snapshot['early_warnings']
            # Only send if there are HIGH or CRITICAL priority warnings
            priority_warnings = [w for w in warnings if w.get('priority') in ['HIGH', 'CRITICAL']]
            if priority_warnings:
                visual_data = snapshot.get('visual_analysis', {})
                await self.notifier.send_early_warning(symbol, priority_warnings, visual_data)
        
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
