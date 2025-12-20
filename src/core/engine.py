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

# PHASE 200: UNIFIED BRAIN & EARLY WARNING (NEW ARCHITECTURE)
from src.brain.unified_brain import get_unified_brain
from src.brain.early_warning import get_warning_system

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
        
        # 14. PHASE 129: Continuous Monitor - 4 coin, WebSocket, anti-spam
        self.continuous_monitor = None  # Lazy init after notifier is ready
        
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
        
        # PHASE 129: Start Continuous Monitor (WebSocket + 4-coin scanning)
        try:
            from src.brain.continuous_monitor import get_continuous_monitor
            
            # Callback for notifications
            async def notify_callback(msg):
                await self.notifier.send_message_raw(msg)
            
            self.continuous_monitor = get_continuous_monitor(callback=notify_callback)
            
            # Start WebSocket streams (background)
            asyncio.create_task(self.continuous_monitor.start_websocket())
            
            # Start continuous scanning (background)
            asyncio.create_task(self.continuous_monitor.run_continuous_scan())
            
            logger.info("🔌 Continuous Monitor started (4 coins + WebSocket)")
        except Exception as e:
            logger.warning(f"Continuous Monitor start failed: {e}")
        
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
                
                # ═══════════════════════════════════════════════════════════════
                # PHASE 200: UNIFIED BRAIN - YENİ MİMARİ (PRIMARY SYSTEM)
                # Tüm eski brain sistemlerini tek noktada birleştiren yeni yapı
                # ═══════════════════════════════════════════════════════════════
                try:
                    unified_brain = get_unified_brain()
                    warning_system = get_warning_system()
                    
                    # Her 5 dakikada bir erken uyarı taraması
                    if not hasattr(self, 'last_warning_scan'):
                        self.last_warning_scan = datetime.now() - timedelta(minutes=10)
                    
                    if (datetime.now() - self.last_warning_scan).total_seconds() >= 300:
                        warnings = await warning_system.scan_all()
                        
                        if warnings:
                            # Sadece HIGH ve CRITICAL uyarıları gönder
                            critical_warnings = [w for w in warnings if w.severity in ['HIGH', 'CRITICAL']]
                            if critical_warnings:
                                msg = warning_system.format_warnings(critical_warnings)
                                await self.notifier.send_message_raw(msg)
                                logger.info(f"⚠️ {len(critical_warnings)} erken uyarı gönderildi")
                        
                        self.last_warning_scan = datetime.now()
                    
                    # Her 15 dakikada bir Unified Brain analizi
                    if not hasattr(self, 'last_unified_analysis'):
                        self.last_unified_analysis = datetime.now() - timedelta(minutes=20)
                    
                    if (datetime.now() - self.last_unified_analysis).total_seconds() >= 900:
                        for symbol in ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']:
                            signal = await unified_brain.analyze(symbol)
                            
                            if signal:
                                msg = unified_brain.format_for_telegram(signal)
                                await self.notifier.send_message_raw(msg)
                                logger.info(f"🧠 UNIFIED BRAIN: {symbol} {signal.direction} %{signal.confidence:.0f}")
                        
                        self.last_unified_analysis = datetime.now()
                        
                except Exception as unified_err:
                    logger.debug(f"Unified Brain skipped: {unified_err}")
                
                # ═══════════════════════════════════════════════════════════════
                # LEGACY SYSTEMS (Eski sistemler - Unified Brain'e yedek)
                # ═══════════════════════════════════════════════════════════════
                
                # PHASE 127: AI REASONING ENGINE - Gerçek Akıl Yürütme
                # Tüm modülleri birleştirip DÜŞÜNEN bir AI
                try:
                    from src.brain.ai_reasoning_engine import get_reasoning_engine
                    
                    reasoning = get_reasoning_engine()
                    
                    # Her 15 dakikada bir akıl yürüt (900 saniye)
                    if not hasattr(self, 'last_reasoning_time'):
                        self.last_reasoning_time = datetime.now() - timedelta(hours=1)
                    
                    if (datetime.now() - self.last_reasoning_time).total_seconds() >= 900:
                        for symbol in ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']:  # All 4 coins
                            prediction = await reasoning.think(symbol)
                            
                            if prediction and prediction.confidence >= 55:
                                msg = reasoning.format_for_telegram(prediction, symbol)
                                await self.notifier.send_message_raw(msg)
                                logger.info(f"🧠 AI REASONING: {symbol} → {prediction.direction} %{prediction.confidence:.0f}")
                        
                        # PHASE 131: TECHNICAL SCANNER - 4 coin x 14 modül
                        try:
                            from src.brain.technical_scanner import get_technical_scanner
                            scanner = get_technical_scanner()
                            scans = await scanner.scan_all_coins()
                            
                            for scan in scans:
                                msg = scanner.format_scan_message(scan)
                                await self.notifier.send_message_raw(msg)
                                logger.info(f"📊 TECH SCAN: {scan.symbol} → {scan.direction} %{scan.overall_confidence:.0f}")
                        except Exception as scan_err:
                            logger.debug(f"Technical scan skipped: {scan_err}")
                        
                        self.last_reasoning_time = datetime.now()
                    
                    # PHASE 128: ADVANCED AI INTELLIGENCE FEATURES
                    now = datetime.now()
                    
                    # 1. Daily Briefing - 09:00 ve 21:00
                    if not hasattr(self, 'last_briefing_date'):
                        self.last_briefing_date = None
                        self.morning_briefing_sent = False
                        self.evening_briefing_sent = False
                    
                    if self.last_briefing_date != now.date():
                        self.last_briefing_date = now.date()
                        self.morning_briefing_sent = False
                        self.evening_briefing_sent = False
                    
                    # Morning: Only at exactly 09:XX (not after restart)
                    if now.hour == 9 and now.minute < 30 and not self.morning_briefing_sent:
                        briefing = await reasoning.generate_daily_briefing(morning=True)
                        await self.notifier.send_message_raw(briefing)
                        self.morning_briefing_sent = True
                        logger.info("🌅 Morning briefing sent")
                    
                    # Evening: Only at exactly 21:XX (not after restart) 
                    if now.hour == 21 and now.minute < 30 and not self.evening_briefing_sent:
                        briefing = await reasoning.generate_daily_briefing(morning=False)
                        await self.notifier.send_message_raw(briefing)
                        self.evening_briefing_sent = True
                        logger.info("🌙 Evening briefing sent")
                    
                    # 2. Weekly Outlook - Pazartesi 10:00
                    if not hasattr(self, 'weekly_outlook_sent'):
                        self.weekly_outlook_sent = False
                    
                    if now.weekday() == 0 and now.hour >= 10 and now.hour < 11:
                        if not self.weekly_outlook_sent:
                            outlook = await reasoning.generate_weekly_outlook()
                            await self.notifier.send_message_raw(outlook)
                            self.weekly_outlook_sent = True
                            logger.info("📅 Weekly outlook sent")
                    elif now.weekday() != 0:
                        self.weekly_outlook_sent = False
                    
                    # 3. Risk Alerts - Her saat başı
                    if not hasattr(self, 'last_risk_check'):
                        self.last_risk_check = datetime.now() - timedelta(hours=2)
                    
                    if (now - self.last_risk_check).total_seconds() >= 3600:
                        for symbol in ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']:  # All 4 coins
                            risk_alert = await reasoning.generate_risk_alerts(symbol)
                            if risk_alert:  # Sadece risk varsa gönder
                                await self.notifier.send_message_raw(risk_alert)
                                logger.info(f"🚨 Risk alert sent for {symbol}")
                        self.last_risk_check = now
                    
                    # 4. Whale Commentary - Her 4 saatte bir
                    if not hasattr(self, 'last_whale_comment'):
                        self.last_whale_comment = datetime.now() - timedelta(hours=5)
                    
                    if (now - self.last_whale_comment).total_seconds() >= 14400:  # 4 saat
                        whale_msg = await reasoning.generate_whale_commentary('BTCUSDT')
                        await self.notifier.send_message_raw(whale_msg)
                        self.last_whale_comment = now
                        logger.info("🐋 Whale commentary sent")
                        
                except Exception as reasoning_err:
                    logger.debug(f"AI Reasoning skipped: {reasoning_err}")
                
                # Phase 103-107: LIVING AI BRAIN - Canlı yapay zeka karar sistemi
                # LSTM + RL Agent + Pattern Recognition + Self-Evaluation
                try:
                    from src.brain.living_ai_brain import get_brain
                    from src.brain.signal_gate import get_gate
                    
                    brain = get_brain()
                    gate = get_gate()
                    
                    # Update market regime
                    await brain.update_regime()
                    
                    for symbol in ['BTCUSDT', 'ETHUSDT']:
                        # Gate kontrolü
                        if not gate.can_send_signal(symbol):
                            continue
                        
                        # Market data topla
                        market_data = {
                            'current_price': self.latest_prices.get(symbol, 0),
                            'rsi': 50,  # Will be calculated from actual data
                            'funding_rate': self.derivatives_data.get('funding_rate', 0),
                            'long_short_ratio': self.derivatives_data.get('long_short_ratio', 1),
                        }
                        
                        # AI DÜŞÜNSÜN
                        decision = await brain.think(symbol, market_data)
                        
                        # Sadece yüksek güvenli kararları gönder
                        if decision.confidence >= 65 and decision.action != 'HOLD':
                            current_price = market_data['current_price']
                            msg = brain.format_decision_for_telegram(decision, symbol, current_price)
                            await self.notifier.send_message_raw(msg)
                            
                            # PHASE 117: Sinyali veritabanına kaydet (Self-Learning)
                            try:
                                from src.brain.signal_database import get_signal_database
                                db = get_signal_database()
                                
                                # TP/SL hesapla
                                if decision.action == 'LONG':
                                    tp_price = current_price * 1.035
                                    sl_price = current_price * 0.985
                                else:
                                    tp_price = current_price * 0.965
                                    sl_price = current_price * 1.015
                                
                                signal_id = db.save_signal({
                                    'symbol': symbol,
                                    'direction': decision.action,
                                    'confidence': decision.confidence,
                                    'entry_price': current_price,
                                    'tp_price': tp_price,
                                    'sl_price': sl_price,
                                    'modules': [{'name': m, 'direction': d} for m, d in decision.module_votes.items()] if hasattr(decision, 'module_votes') else []
                                })
                                logger.info(f"💾 Signal #{signal_id} saved to database")
                            except Exception as db_err:
                                logger.debug(f"Signal DB save skipped: {db_err}")
                            
                            # Gate'i kapat
                            gate.open_gate(symbol, {
                                'direction': decision.action,
                                'entry': market_data['current_price'],
                                'confidence': decision.confidence
                            })
                            
                            logger.warning(f"🧠 LIVING AI: {symbol} {decision.action} {decision.confidence:.0f}%")
                    
                    # Periyodik strateji adaptasyonu
                    if brain.performance_stats.get('total_decisions', 0) % 10 == 0:
                        await brain.adapt_strategy()
                        
                except Exception as brain_err:
                    logger.debug(f"Living AI Brain skipped: {brain_err}")
                
                # Phase 108: AI GÖZLEM - Erken piyasa gözlemleri (sinyal değil)
                try:
                    from src.brain.ai_observation import get_observer
                    observer = get_observer()
                    
                    for symbol in ['BTCUSDT', 'ETHUSDT']:
                        obs = await observer.observe(symbol)
                        
                        if obs:
                            msg = observer.format_observation(obs)
                            await self.notifier.send_message_raw(msg)
                            logger.info(f"👁️ AI GÖZLEM: {symbol} {obs['direction']}")
                            
                except Exception as obs_err:
                    logger.debug(f"AI Observation skipped: {obs_err}")
                
                # Phase 21: Daily Heartbeat (was hourly - PHASE 100: reduced spam)
                if (datetime.now() - self.last_heartbeat_time).total_seconds() > 86400:  # 24 hours
                    await self.notifier.send_heartbeat(self.latest_prices)
                    
                    # Phase 30: Money Flow Report - DISABLED (PHASE 100: spam)
                    # try:
                    #     money_flow_data = await self.money_flow_analyzer.get_market_money_flow()
                    #     await self.notifier.send_money_flow_report(money_flow_data)
                    #     logger.info("📊 Money Flow report sent to Telegram.")
                    # except Exception as mf_err:
                    #     logger.error(f"Money Flow report failed: {mf_err}")
                    
                    self.last_heartbeat_time = datetime.now()
                    logger.info("💓 Daily heartbeat sent to Telegram.")
                
                # PHASE 117: Sinyal sonuçlarını kontrol et (Self-Learning)
                try:
                    from src.brain.signal_result_tracker import get_signal_tracker
                    tracker = get_signal_tracker()
                    
                    results = await tracker.check_active_signals()
                    
                    for result in results:
                        await tracker.send_result_notification(result)
                        logger.info(f"📊 Signal #{result['signal_id']} closed: {result['result']}")
                        
                except Exception as tracker_err:
                    logger.debug(f"Signal tracker skipped: {tracker_err}")
                    
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
        # PHASE 100: DISABLED - 15dk Fırsat Taraması (Spam)
        # Only Signal Gate system now - no more 15dk scans
        # ======================================
        # try:
        #     if self.market_intelligence.should_run_15min_check():
        #         logger.info("🔍 15dk Fırsat/Risk Taraması Başlıyor...")
        #         import json
        #         if os.path.exists("dashboard_data.json"):
        #             with open("dashboard_data.json", 'r') as f:
        #                 self.all_snapshots = json.load(f)
        #         opportunities = self.market_intelligence.scan_for_opportunities(self.all_snapshots)
        #         if opportunities:
        #             report = self.market_intelligence.format_opportunity_report(opportunities)
        #             await self.notifier.send_message_raw(report)
        #             logger.info(f"🎯 {len(opportunities)} fırsat/risk Telegram'a gönderildi!")
        #         else:
        #             logger.info("✓ 15dk tarama tamamlandı - önemli fırsat/risk yok")
        #     if self.market_intelligence.should_send_hourly_fallback():
        #         logger.info("📊 Saatlik Durum Özeti Gönderiliyor...")
        #         live_data = {...}
        #         hourly_report = self.market_intelligence.format_hourly_status(...)
        #         await self.notifier.send_message_raw(hourly_report)
        # except Exception as e:
        #     logger.warning(f"Market Intelligence scan failed: {e}")
        pass  # Phase 100: All spam disabled

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
                direction = predictive_signal['direction']
                entry = predictive_signal.get('entry', current_price)
                confidence = predictive_signal.get('confidence', 50)
                
                # PHASE 93: Signal Gate Check - Only 1 active signal per coin
                try:
                    from src.brain.signal_gate import get_gate
                    gate = get_gate()
                    
                    # Gate kontrolü - aktif sinyal varsa gönderme
                    if not gate.can_send_signal(symbol):
                        active = gate.get_active_signals().get(symbol, {})
                        logger.info(f"🚫 BLOCKED: {symbol} {direction} - Active signal: {active.get('direction')} @ ${active.get('entry', 0):,.0f}")
                    
                    # Minimum confidence filter - %65 altı spam
                    elif confidence < 65:
                        logger.debug(f"🔇 LOW CONF: {symbol} {direction} {confidence}% - not sent")
                    
                    else:
                        # Gate açık ve confidence yeterli - gönder
                        formatted_msg = self.predictive_analyzer.format_predictive_signal(predictive_signal)
                        if formatted_msg:
                            await self.notifier.send_message_raw(formatted_msg)
                            logger.info(f"🔮 PREDICTIVE SIGNAL: {symbol} {direction} - {len(predictive_signal.get('reasons', []))} confirmations")
                            
                            # Gate'i kapat - TP/SL gelene kadar yeni sinyal yok
                            gate.open_gate(symbol, {
                                'direction': direction,
                                'entry': entry,
                                'tp1': predictive_signal.get('take_profit_1', entry * 1.02),
                                'tp2': predictive_signal.get('take_profit_2', entry * 1.04),
                                'sl': predictive_signal.get('stop_loss', entry * 0.97),
                                'confidence': confidence
                            })
                            logger.info(f"🔒 Gate CLOSED for {symbol} - waiting for TP/SL")
                
                except Exception as gate_err:
                    logger.warning(f"Signal gate error: {gate_err}")
                    # Fallback: Eski davranış
                    formatted_msg = self.predictive_analyzer.format_predictive_signal(predictive_signal)
                    if formatted_msg:
                        await self.notifier.send_message_raw(formatted_msg)
        except Exception as e:
            logger.warning(f"Predictive analysis failed for {symbol}: {e}")
        
        # --- PHASE 100: DISABLED - ERKEN UYARI (Early Warning) ---
        # These were causing spam. Only Signal Gate system now.
        # if snapshot and snapshot.get('early_warnings'):
        #     warnings = snapshot['early_warnings']
        #     priority_warnings = [w for w in warnings if w.get('priority') in ['HIGH', 'CRITICAL']]
        #     if priority_warnings:
        #         visual_data = snapshot.get('visual_analysis', {})
        #         await self.notifier.send_early_warning(symbol, priority_warnings, visual_data)
        
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
