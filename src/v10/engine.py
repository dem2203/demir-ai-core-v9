# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ENHANCED MAIN ENGINE + EARLY SIGNAL
===================================================
Ana döngü - Early Signal Engine entegrasyonu.

PHASE 400: Early Signal Engine - Leading indicators kullanır.
"""
import logging
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from src.v10.data_hub import get_data_hub, MarketSnapshot
from src.v10.enhanced_predictor import get_enhanced_predictor, SignalType
from src.v10.smart_notifier import get_notifier
from src.v10.performance_tracker import get_performance_tracker
from src.v10.lstm_predictor import get_lstm_predictor
from src.v10.early_signal_engine import get_early_signal_engine
from src.v10.signal_history import record_early_signal
from src.v10.ai_integration import get_ai_bridge  # RL Agent Integration
from src.v10.online_learner import get_online_learner  # Self-Learning
from src.brain.feedback_db import get_feedback_db  # Trade outcomes
from src.execution.paper_trader import get_paper_trader
from src.execution.feedback_loop import FeedbackLoop
from src.v10.early_signal_trainer import get_trainer
from src.brain.rl_agent.auto_retrain import AutoRetrainPipeline  # Smart Pipeline
from src.brain.correlation_analyzer import get_correlation_analyzer  # Multi-Coin Correlation
from src.v10.signal_quality_filter import get_signal_quality_filter  # Quality Gate

logger = logging.getLogger("V10_ENGINE")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class V10Engine:
    """
    DEMIR AI v10 - ENHANCED Ana Motor + Early Signal Engine
    """
    
    SCAN_INTERVAL = 30
    SIGNAL_COOLDOWN = 30 * 60
    PERFORMANCE_REPORT_INTERVAL = 4 * 60 * 60  # 4 hours
    DAILY_REPORT_INTERVAL = 24 * 60 * 60  # 24 hours
    
    def __init__(self):
        self.data_hub = get_data_hub()
        self.predictor = get_enhanced_predictor()
        self.notifier = get_notifier()
        self.performance_tracker = get_performance_tracker()
        self.lstm_predictor = get_lstm_predictor()
        self.early_signal_engine = None  # Lazy init
        self.paper_trader = get_paper_trader()
        self.feedback_loop = FeedbackLoop()
        self.trainer = get_trainer()
        self.last_retrain_check = datetime.now()
        self.auto_retrain_pipeline = AutoRetrainPipeline(enable_auto_deploy=True) # Pipeline init
        
        # SELF-LEARNING COMPONENTS
        self.online_learner = get_online_learner()
        self.feedback_db = get_feedback_db()
        self._current_regime = "UNKNOWN"  # Gets updated per cycle
        self.correlation_analyzer = get_correlation_analyzer()  # Multi-Coin Correlation

        self._last_signal_time: Dict[str, datetime] = {}
        self._last_performance_report = datetime.now()
        self._last_daily_report = datetime.now()
        self._running = False
        self._cycle_count = 0
        self._signal_count = 0
        self._error_count = 0
        
        logger.info("[START] DEMIR AI v10 + SELF-LEARNING ENGINE initialized")
    
    async def start(self):
        """Ana donguyu baslat"""
        self._running = True
        
        # Initialize AI Brain (RL Agent)
        await get_ai_bridge().initialize()
        
        self.notifier.send_startup_message()
        logger.info("[RUN] V10 Engine started with Early Signal + RL Brain...")
        
        # === START REAL-TIME PREDICTIVE ENGINE (Background) ===
        try:
            from src.brain.instant_alerts import get_instant_alert_system
            instant_system = get_instant_alert_system(self.notifier)
            symbols = ["BTCUSDT", "ETHUSDT"]
            
            # Run as background task (non-blocking)
            asyncio.create_task(instant_system.start(symbols))
            logger.info("⚡ Real-Time Predictive Engine started in background")
        except Exception as e:
            logger.warning(f"Real-time engine startup failed: {e}")
        
        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[ERROR] SCAN CYCLE: {e}")
                self._error_count += 1
                if self._error_count >= 3:
                    self.notifier.send_error_alert(f"Tekrarlayan hata: {e}")
            
            try:
                # 4 saatte bir Retrain Check
                if datetime.now().hour % 4 == 0 and datetime.now().minute < 5:
                     await self._check_auto_retrain()

            except Exception as e:
                 pass
            
            
            # --- AUTO RETRAIN CHECK (Every 15 mins) ---
            try:
                if (datetime.now() - self.last_retrain_check).total_seconds() > 900: # 15 mins
                    self.last_retrain_check = datetime.now()
                    await self._check_auto_retrain()
            except Exception as e:
                logger.error(f"[AUTO-RETRAIN] Error: {e}")
            
            await asyncio.sleep(self.SCAN_INTERVAL)
            
            # --- PERIODIC TASKS ---
            now = datetime.now()
            
            # 1. Performance Report (4 saat)
            if (now - self._last_performance_report).total_seconds() > 4 * 3600:
                self._last_performance_report = now
                # ... existing logic likely uses separate timer, but we add Retrain check here
                
                # Check Auto-Retrain
                try:
                    from src.brain.rl_agent.auto_retrain import AutoRetrainPipeline
                    pipeline = AutoRetrainPipeline()
                    needs = pipeline.check_retrain_needed()
                    
                    symbols_needing_retrain = [s for s, n in needs.items() if n]
                    
                    if symbols_needing_retrain:
                        msg = f"🧠 **AI BRAIN ALERT**\nModeller yeniden eğitim istiyor: {', '.join(symbols_needing_retrain)}\nSebep: Performans düşüşü veya zamanı geldi."
                        self.notifier.send_error_alert(msg)
                        logger.warning(f"[AUTO-RETRAIN] Needed for: {symbols_needing_retrain}")
                except Exception as e:
                    logger.error(f"Auto-retrain check failed: {e}")
    
    async def stop(self):
        """Motoru durdur"""
        self._running = False
        await self.data_hub.close()
        if self.early_signal_engine:
            await self.early_signal_engine.close()
        logger.info("[STOP] V10 Engine stopped")
    
    async def _scan_cycle(self):
        """Tek bir tarama dongusu"""
        self._cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"=== SCAN CYCLE #{self._cycle_count} ===")
        
        snapshots = await self.data_hub.get_all_snapshots()
        
        # === REGIME DETECTION FOR SELF-LEARNING ===
        try:
            # Get BTC regime as market proxy (most reliable)
            if 'BTCUSDT' in snapshots and self.early_signal_engine:
                regime_result = await self.early_signal_engine._analyze_regime('BTCUSDT')
                self._current_regime = regime_result.get('regime', 'UNKNOWN')
                logger.info(f"[REGIME] Market: {self._current_regime}")
        except Exception as e:
            logger.debug(f"Regime detection skipped: {e}")
        
        for symbol, snapshot in snapshots.items():
            try:
                await self._process_coin(symbol, snapshot)
            except Exception as e:
                logger.error(f"[ERROR] {symbol}: {e}")
        
        # === AKILLI POZİSYON YÖNETİMİ ===
        try:
            current_prices = {s: snap.price for s, snap in snapshots.items() if snap.is_valid}
            if current_prices:
                self.paper_trader.update_positions(current_prices)
        except Exception as e:
            logger.error(f"Position update error: {e}")
        
        # === MACRO EVENT PROTECTION ===
        try:
            from src.brain.macro_events import check_upcoming_events
            
            # Check every hour (avoid spam)
            if not hasattr(self, '_last_macro_check'):
                self._last_macro_check = datetime.now() - timedelta(hours=2)
            
            if (datetime.now() - self._last_macro_check).total_seconds() >= 3600:  # 1 hour
                self._last_macro_check = datetime.now()
                
                events = check_upcoming_events(hours_ahead=12)
                
                if events:
                    for event in events:
                        msg = f"""
⚠️ *MAKRO OLAY UYARISI*

🚨 {event['hours_until']} saat içinde: **{event['name']}**

TAVSİYE:
• Yeni pozisyon AÇMA
• Mevcut SL'leri SIKILAŞTIR
• Beklenen volatilite: %{event['impact']}
"""
                        self.notifier._send_message(msg)
                        logger.warning(f"MACRO ALERT: {event['name']} in {event['hours_until']}h")
        except Exception as e:
            logger.debug(f"Macro check skipped: {e}")
        
        # Export data for Dashboard - REAL DATA ONLY
        try:
            import json
            export_data = {
                'updated_at': datetime.now().isoformat(),
                'coins': {}
            }
            
            for s, snapshot in snapshots.items():
                if snapshot.is_valid:
                    metrics = self._calculate_real_metrics(snapshot)
                    s_data = {
                        'price': snapshot.price,
                        'volume': snapshot.volume_24h,
                        'change_24h': snapshot.price_change_24h,
                        'timestamp': datetime.now().timestamp(),
                        'rsi': metrics.get('rsi'),
                        'trend': metrics.get('trend'),
                        'ema21': metrics.get('ema21')
                    }
                    export_data['coins'][s] = s_data
            
            with open("dashboard_data.json", "w") as f:
                json.dump(export_data, f)
        except Exception as e:
            logger.error(f"Dashboard export error: {e}")


        
        # SAVE DASHBOARD DATA
        try:
            self._save_dashboard_data(snapshots)
        except Exception as e:
            logger.error(f"Dashboard data save error: {e}")

        # === ESKI SİSTEMLER DEVRE DIŞI (Premium Signals ile değiştirildi) ===
        
        # 1. PİYASA RAPORU (15 dakika) - DEVRE DIŞI
        # if not hasattr(self, '_last_market_summary'):
        #     self._last_market_summary = datetime.now() - timedelta(minutes=20)
        # 
        # if (datetime.now() - self._last_market_summary).total_seconds() >= 15 * 60:
        #     try:
        #         logger.info("[MARKET] Sending 15-Min Market Summary...")
        #         self._last_market_summary = datetime.now()
        #         await self.notifier.send_market_summary(snapshots)
        #     except Exception as e:
        #         logger.error(f"Market summary error: {e}")
        
        # 2. DERİN TEKNİK ANALİZ (10 dakika) - DEVRE DIŞI
        # if not hasattr(self, '_last_deep_technical'):
        #     self._last_deep_technical = datetime.now() - timedelta(minutes=12)
        # 
        # if (datetime.now() - self._last_deep_technical).total_seconds() >= 10 * 60:
        #     try:
        #         logger.info("[DEEP-TECH] Sending 10-Min Deep Technical...")
        #         self._last_deep_technical = datetime.now()
        #         
        #         # Her coin için derin analiz
        #         for symbol in snapshots.keys():
        #             await self.notifier.send_deep_technical_report(symbol)
        #             await asyncio.sleep(2)  # Rate limit
        #     except Exception as e:
        #         logger.error(f"Deep technical error: {e}")
        
        # Performance report (4 saatte bir)
        if (datetime.now() - self._last_performance_report).total_seconds() >= self.PERFORMANCE_REPORT_INTERVAL:
            try:
                await self.performance_tracker.check_outcomes()
                report_msg = self.performance_tracker.format_report_message()
                self.notifier._send_message(report_msg)
                self._last_performance_report = datetime.now()
            except Exception as e:
                logger.error(f"[ERROR] Performance report: {e}")
        
        # GÜNLÜK PERFORMANS RAPORU (24 saatte bir)
        if (datetime.now() - self._last_daily_report).total_seconds() >= self.DAILY_REPORT_INTERVAL:
            try:
                logger.info("[DAILY] Sending daily performance report...")
                
                # Paper Trade istatistikleri
                stats = self.paper_trader.get_stats()
                
                # Detaylı günlük rapor
                daily_msg = f"""
📊 *GÜNLÜK PERFORMANS RAPORU*
━━━━━━━━━━━━━━━━━━━━━━

💰 *PAPER TRADE İSTATİSTİKLERİ*
  Bakiye: ${stats.get('balance', 10000):,.2f}
  Toplam Trade: {stats.get('total_trades', 0)}
  Win Rate: {stats.get('win_rate', 0)*100:.1f}%
  Toplam P/L: ${stats.get('total_pnl', 0):,.2f}

📈 *EN İYİ / EN KÖTÜ*
  En İyi: ${stats.get('best_trade', 0):,.2f}
  En Kötü: ${stats.get('worst_trade', 0):,.2f}

🎯 *SİNYAL İSTATİSTİKLERİ*
  Bugün Üretilen: {self._signal_count}
  Cycle Sayısı: {self._cycle_count}

━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
📡 *DEMIR AI v10 - DAILY REPORT*
"""
                self.notifier._send_message(daily_msg)
                self._last_daily_report = datetime.now()
                
                # Reset daily counters
                self._signal_count = 0
                
            except Exception as e:
                logger.error(f"[ERROR] Daily report: {e}")
        
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        
        logger.info(
            f"[OK] Cycle #{self._cycle_count}: {cycle_time:.1f}s | "
            f"Signals: {self._signal_count} | Errors: {self._error_count}"
        )
        
    def _save_dashboard_data(self, snapshots: Dict[str, MarketSnapshot]):
        """Snapshots verisini dashboard icin JSON'a kaydet."""
        data = {}

    def _calculate_real_metrics(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Calculate real metrics from snapshot - NO MOCK DATA"""
        metrics = {
            'rsi': None,
            'trend': 'UNKNOWN',
            'ema21': None
        }
        
        # 1. Use DataHub metrics if available and valid
        if hasattr(snapshot, 'rsi_1h') and snapshot.rsi_1h > 0:
            metrics['rsi'] = snapshot.rsi_1h
        
        if hasattr(snapshot, 'trend') and snapshot.trend != 'UNKNOWN':
            metrics['trend'] = snapshot.trend
            
        # 2. If missing, calculate from raw_klines
        if metrics['rsi'] is None and hasattr(snapshot, 'raw_klines') and snapshot.raw_klines:
            try:
                closes = [float(k[4]) for k in snapshot.raw_klines]
                if len(closes) > 14:
                    metrics['rsi'] = self._calc_rsi(closes)
                    
                    # Trend via EMA21
                    if len(closes) > 21:
                        ema_values = self._calc_ema(closes, 21)
                        if ema_values:
                            ema21 = ema_values[-1]
                            current = closes[-1]
                            metrics['ema21'] = ema21
                            metrics['trend'] = "UP" if current > ema21 else "DOWN"
            except Exception:
                pass
                
        return metrics

    def _calc_ema(self, data: list, period: int) -> list:
        if len(data) < period: return []
        c = 2.0 / (period + 1)
        # Start with SMA
        current_ema = sum(data[:period]) / period
        ema_values = [current_ema]
        for value in data[period:]:
            current_ema = (c * value) + ((1 - c) * current_ema)
            ema_values.append(current_ema)
        return ema_values
        for symbol, snap in snapshots.items():
             # Convert dataclass to dict
             snap_dict = asdict(snap)
             # Remove raw_klines to keep file size small (dashboard fetches its own history if needed, or we keep it?)
             # Dashboard likely needs basic info.
             if 'raw_klines' in snap_dict:
                 del snap_dict['raw_klines']
             
             data[symbol] = snap_dict
             
        try:
            with open("dashboard_data.json", "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write dashboard_data.json: {e}")
    
    async def _process_coin(self, symbol: str, snapshot: MarketSnapshot):
        """Tek coin icin sinyal kontrolu - EARLY SIGNAL ENGINE"""
        
        if not snapshot.is_valid:
            if snapshot.errors:
                logger.warning(f"[WARN] {symbol}: Data errors - {snapshot.errors[:2]}")
            return
        
        if self._is_on_cooldown(symbol):
            return
        
        # PHASE 400: EARLY SIGNAL ENGINE
        try:
            if self.early_signal_engine is None:
                self.early_signal_engine = await get_early_signal_engine()
            
            early_signal = await self.early_signal_engine.analyze(symbol)
            
            if early_signal:
                direction = early_signal.leading_signal.direction.value
                logger.info(
                    f"[EARLY] {symbol} | {early_signal.action} | "
                    f"Conf: {early_signal.confidence:.0f}% | {direction}"
                )
                
                # === PROFESSIONAL QUALITY FILTER ===
                # Regime-aligned, Kelly-sized, veto-capable
                quality_filter = get_signal_quality_filter()
                
                # Extract REAL metrics from signal
                rsi_val = 50.0
                orderbook_val = 0.0
                whale_val = 0.0
                funding_val = 0.0
                
                # 1. Get RSI from technical_indicators
                if hasattr(early_signal, 'technical_indicators') and early_signal.technical_indicators:
                    ti = early_signal.technical_indicators
                    rsi_val = ti.get('rsi', 50.0)
                    logger.debug(f"[QUALITY] RSI from tech_indicators: {rsi_val}")
                
                # 2. Get orderbook/whale from leading_signal
                if hasattr(early_signal, 'leading_signal') and early_signal.leading_signal:
                    ls = early_signal.leading_signal
                    orderbook_val = getattr(ls, 'orderbook_score', 0) or 0
                    whale_val = getattr(ls, 'whale_score', 0) or 0
                    logger.debug(f"[QUALITY] Orderbook: {orderbook_val}, Whale: {whale_val}")
                
                # 3. Get funding from liq_data in risk_profile
                if hasattr(early_signal, 'risk_profile') and early_signal.risk_profile:
                    rp = early_signal.risk_profile
                    funding_val = rp.get('funding_rate', 0) or 0
                    # Also try to get orderbook if not set
                    if orderbook_val == 0:
                        orderbook_val = rp.get('orderbook_imbalance', 0) or 0
                    if whale_val == 0:
                        whale_val = rp.get('whale_activity', 0) or 0
                
                # Get sentiment from reasoning
                sentiment = "NEUTRAL"
                if "BULLISH" in early_signal.reasoning:
                    sentiment = "BULLISH"
                elif "BEARISH" in early_signal.reasoning:
                    sentiment = "BEARISH"
                
                quality_result = quality_filter.check_signal(
                    symbol=symbol,
                    action=early_signal.action,
                    confidence=int(early_signal.confidence),
                    regime=self._current_regime,
                    sentiment=sentiment,
                    rsi=rsi_val,
                    orderbook_imbalance=orderbook_val,
                    whale_flow=whale_val,
                    funding_rate=funding_val
                )
                
                # DEBUG: Log what regime was checked
                logger.info(f"[QUALITY] {symbol}: action={early_signal.action}, regime={self._current_regime}, sentiment={sentiment}")
                
                if not quality_result.passed:
                    logger.warning(f"[QUALITY] Signal blocked: {quality_result.reason}")
                    if quality_result.veto_active:
                        self.notifier._send_message(f"🛡️ *SİNYAL VETO*\n{symbol}: {quality_result.reason}")
                    return
                
                # Use adjusted confidence and Kelly from filter
                early_signal.confidence = quality_result.adjusted_confidence
                kelly_position_pct = quality_result.kelly_position_pct
                
                # 3. No HOLD signals - but track as shadow trade for learning
                if early_signal.action == 'HOLD':
                    logger.debug(f"Skipped {symbol}: HOLD signal")
                    return

                # === 5. CORRELATION RISK CHECK (NEW) ===
                try:
                    # Get open paper trade positions
                    open_positions = []
                    for pos_sym, pos_data in self.paper_trader.positions.items():
                        if pos_data.get('quantity', 0) > 0:
                            open_positions.append({
                                'symbol': pos_sym,
                                'direction': 'BUY'  # Paper trades are always long for now
                            })
                    
                    if open_positions:
                        should_skip, corr_reason = await self.correlation_analyzer.should_skip_signal(
                            new_signal_symbol=symbol,
                            new_signal_direction=early_signal.action,
                            open_positions=open_positions
                        )
                        
                        if should_skip:
                            logger.warning(f"[CORRELATION] Signal skipped: {corr_reason}")
                            self.notifier._send_message(f"⚠️ *SINYAL ATLANDI*\n{corr_reason}")
                            return
                except Exception as e:
                    logger.debug(f"Correlation check error: {e}")

                # === SELF-LEARNING CONFIDENCE ADJUSTMENT ===
                base_confidence = early_signal.confidence
                adjusted_prediction = self.online_learner.adjust_prediction(
                    {'action': early_signal.action, 'confidence': base_confidence / 100},
                    current_regime=self._current_regime
                )
                adjusted_confidence = adjusted_prediction['confidence'] * 100
                adjustment_info = adjusted_prediction.get('online_adjustment', {})
                
                # Update confidence in early_signal for paper trading
                final_confidence = adjusted_confidence
                
                # === REGIME-AWARE STRATEGY HINTS ===
                regime_hint = ""
                if self._current_regime == "TRENDING_BULL":
                    regime_hint = "🐂 Bull Market: Hold winners longer, add on dips"
                elif self._current_regime == "TRENDING_BEAR":
                    regime_hint = "🐻 Bear Market: Quick scalps, tight stops"
                elif self._current_regime == "RANGING":
                    regime_hint = "📦 Range: Trade extremes, mean-reversion"
                
                action_tr = "AL (LONG)" if early_signal.action == "BUY" else "SAT (SHORT)"
                arrow = "📈" if early_signal.action == "BUY" else "📉"
                tag = f"{arrow} LONG" if early_signal.action == "BUY" else f"{arrow} SHORT"
                
                # Dogrulama verileri topla
                verification = await self._get_verification_data(symbol)
                
                # Calculate P/L potential
                risk_amount = abs(early_signal.entry_zone[0] - early_signal.stop_loss)
                reward_amount = abs(early_signal.take_profit - early_signal.entry_zone[0])
                
                # Build enhanced message with clear direction - PRO VERSION
                
                # Get Kelly position size - USE VALUE FROM QUALITY FILTER
                # kelly_position_pct is already set by SignalQualityFilter above
                kelly_pct = kelly_position_pct  # From quality filter (dynamic)
                risk_approved = not quality_result.veto_active
                
                risk_status = "✅ Risk Engine OK" if risk_approved else "⚠️ Risk Warning"
                
                # Get AI Council votes from early_signal_engine
                ai_council_section = ""
                try:
                    if self.early_signal_engine and hasattr(self.early_signal_engine, '_last_council_decision'):
                        council = self.early_signal_engine._last_council_decision
                        if council and hasattr(council, 'individual_analyses') and council.individual_analyses:
                            votes = []
                            for anal in council.individual_analyses:
                                model = getattr(anal, 'model_name', 'Unknown')
                                direction = getattr(anal, 'direction', 'HOLD')
                                conf = getattr(anal, 'confidence', 0)
                                emoji = "🟢" if direction in ["LONG", "BUY"] else "🔴" if direction in ["SHORT", "SELL"] else "⚪"
                                votes.append(f"  {emoji} {model}: {direction} ({conf}%)")
                            if votes:
                                ai_council_section = "\n🤖 AI COUNCIL:\n" + "\n".join(votes)
                except Exception as e:
                    logger.debug(f"AI Council parse error: {e}")
                
                msg = f"""{tag} | {symbol}
━━━━━━━━━━━━━━━━━━━━

📍 YÖN: {action_tr}
🎯 GÜVEN: {final_confidence:.0f}% (Sinyalin ne kadar güvenilir olduğu)
📊 R/R: {early_signal.risk_reward:.1f}x (Risk/Kazanç oranı - 2x = 1 birim risk ile 2 birim kazanç hedefi)

💰 GİRİŞ: ${early_signal.entry_zone[0]:,.2f} (Bu fiyattan pozisyon aç)
🛡️ STOP LOSS: ${early_signal.stop_loss:,.2f} (Zarar bu fiyata ulaşırsa kapat)
🎯 HEDEF: ${early_signal.take_profit:,.2f} (Kâr bu fiyata ulaşırsa kapat)

📊 RİSK: ${risk_amount:,.0f} (Max kayıp) | KAZANÇ: ${reward_amount:,.0f} (Hedef kâr)

💰 KELLY BOYUTLANDIRMA:
  📐 Pozisyon: %{kelly_pct:.1f} (Sermayenin ne kadarını riske at - düşük = güvenli)
  🛡️ {risk_status}

🧠 AI BRAIN (Yapay zeka analizi):
{early_signal.reasoning}
{ai_council_section}
🔄 SELF-LEARNING (Öğrenen AI):
  Win Rate: {adjustment_info.get('regime_win_rate', 0)*100:.0f}% (Kazanan işlem oranı)
  {regime_hint}

✅ DOĞRULAMA (Teknik onay):
{verification}
━━ DEMIR AI v10 PRO ━━"""
                
                self.notifier._send_message(msg)
                self._last_signal_time[symbol] = datetime.now()
                self._signal_count += 1
                
                # Sinyal geçmişine kaydet
                try:
                    record_early_signal(early_signal, symbol)
                except Exception as e:
                    logger.warning(f"Signal history error: {e}")
                
                logger.info(f"[SIGNAL] Early Signal sent: {symbol}")
                
                # PAPER TRADING EXECUTION
                try:
                    trade_signal = {
                        "symbol": symbol,
                        "side": early_signal.action,
                        "entry_price": early_signal.entry_zone[0],
                        "sl_price": early_signal.stop_loss,
                        "tp_price": early_signal.take_profit,  # Fixed: was tp_levels[0]
                        "confidence": early_signal.confidence,
                        "rsi": getattr(self, '_last_rsi', -1),
                        "regime": self._current_regime
                    }
                    if self.paper_trader.execute_trade(trade_signal):
                        logger.info(f"📝 Paper Trade Executed: {symbol} | Entry: ${early_signal.entry_zone[0]:,.0f} | SL: ${early_signal.stop_loss:,.0f} | TP: ${trade_signal['tp_price']:,.0f}")
                        self.notifier._send_message(f"📝 *PAPER TRADE AÇILDI* - {symbol}\n💰 Giriş: ${early_signal.entry_zone[0]:,.0f}\n🛡️ SL: ${early_signal.stop_loss:,.0f}\n🎯 TP: ${trade_signal['tp_price']:,.0f}")
                    else:
                        logger.warning(f"⚠️ Paper trade NOT executed: {symbol} (maybe already open?)")
                except Exception as pt_err:
                    logger.error(f"Paper trade execution error: {pt_err}")

                return
                
        except Exception as e:
            logger.error(f"[ERROR] Early Signal: {symbol}: {e}")
    
    async def _get_verification_data(self, symbol: str) -> str:
        """Sinyal icin dogrulama verileri topla."""
        try:
            import aiohttp
            
            lines = []
            warnings = []
            
            async with aiohttp.ClientSession() as session:
                # 1. Klines -> RSI, EMA
                url = f"https://fapi.binance.com/fapi/v1/klines"
                params = {"symbol": symbol, "interval": "1h", "limit": 50}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        closes = [float(k[4]) for k in klines]
                        current = closes[-1]
                        
                        # RSI
                        rsi = self._calc_rsi(closes)
                        if rsi:
                            status = "Düşük - alım fırsatı" if rsi < 40 else "Yüksek - satım riski" if rsi > 60 else "Nötr"
                            lines.append(f"RSI: {rsi:.0f} ({status})")
                            if rsi > 70:
                                warnings.append("RSI aşırı alım (düşüş riski)")
                            elif rsi < 30:
                                warnings.append("RSI aşırı satım (yükseliş fırsatı)")
                        
                        # EMA Trend
                        ema21 = sum(closes[-21:]) / 21
                        trend = "Yukarı ↑" if current > ema21 else "Aşağı ↓"
                        trend_hint = "fiyat ortalamanın üstünde" if current > ema21 else "fiyat ortalamanın altında"
                        lines.append(f"Trend: {trend} ({trend_hint})")
                
                # 2. Order Book
                url = f"https://fapi.binance.com/fapi/v1/depth"
                params = {"symbol": symbol, "limit": 20}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        ob = await resp.json()
                        bid = sum(float(b[1]) for b in ob['bids'])
                        ask = sum(float(a[1]) for a in ob['asks'])
                        imb = (bid - ask) / (bid + ask) * 100
                        status = "Alıcılar güçlü ↑" if imb > 10 else "Satıcılar güçlü ↓" if imb < -10 else "Dengeli"
                        lines.append(f"Order Book: {imb:+.0f}% ({status})")
                        if imb < -20:
                            warnings.append("Güçlü satış baskısı!")
                        elif imb > 20:
                            warnings.append("Güçlü alım desteği")
                
                # 3. Funding
                url = "https://fapi.binance.com/fapi/v1/fundingRate"
                params = {"symbol": symbol, "limit": 1}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            fr = float(data[-1]['fundingRate']) * 100
                            fr_hint = "Long'lar ödüyor" if fr > 0 else "Short'lar ödüyor" if fr < 0 else "Nötr"
                            lines.append(f"Funding: {fr:.4f}% ({fr_hint})")
            
            result = "\n".join(f"  {l}" for l in lines)
            if warnings:
                result += "\n  [!] " + ", ".join(warnings)
            
            return result
            
        except Exception as e:
            return f"  Dogrulama hatasi: {e}"
    
    def _calc_rsi(self, closes, period=14):
        """RSI hesapla."""
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            c = closes[i] - closes[i-1]
            gains.append(c if c > 0 else 0)
            losses.append(abs(c) if c < 0 else 0)
        avg_g = sum(gains[-period:]) / period
        avg_l = sum(losses[-period:]) / period
        if avg_l == 0:
            return 100
        return 100 - (100 / (1 + avg_g / avg_l))
    
    def _is_on_cooldown(self, symbol: str) -> bool:
        if symbol not in self._last_signal_time:
            return False
        elapsed = (datetime.now() - self._last_signal_time[symbol]).total_seconds()
        return elapsed < self.SIGNAL_COOLDOWN
    
    async def _check_auto_retrain(self):
        """
        Auto-retrain LSTM model with latest data.
        Called every 15 minutes, actually checks needs via pipeline.
        """
        try:
            # Check pipeline status (Smart Check)
            needs_retrain = self.auto_retrain_pipeline.check_retrain_needed()
            
            # Filter symbols that need retrain
            symbols_to_train = [s for s, needed in needs_retrain.items() if needed]
            
            if symbols_to_train:
                logger.info(f"🧠 Smart Auto-Retrain triggered for: {symbols_to_train}")
                
                # Execute retrain using pipeline (handles backup, train, compare, rollback)
                results = await self.auto_retrain_pipeline.retrain_all()
                
                for symbol, metrics in results.items():
                    if metrics.status == "success":
                         # Deployed automatically if enable_auto_deploy=True
                         logger.info(f"✅ {symbol} Retrain Success: Sharpe={metrics.sharpe_ratio:.2f}")
                         self.notifier.send_telegram_message(f"🧠 **AI Self-Learning**\n\n{symbol} Modeli güncellendi.\nYeni Sharpe: {metrics.sharpe_ratio:.2f}")
                    elif metrics.status == "failed":
                         logger.warning(f"❌ {symbol} Retrain Failed (Rolled back)")
                         
            else:
                 # No retrain needed
                 pass

        except Exception as e:
            logger.error(f"Auto-retrain error: {e}")


_engine: Optional[V10Engine] = None

def get_v10_engine() -> V10Engine:
    global _engine
    if _engine is None:
        _engine = V10Engine()
    return _engine


async def run_v10():
    """V10 Engine'i baslat"""
    engine = get_v10_engine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(run_v10())
