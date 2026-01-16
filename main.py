import asyncio
import logging
import signal
import sys
from datetime import datetime
from src.config import Config


from src.utils.logger import setup_logging
from src.infrastructure.binance_api import BinanceAPI
from src.infrastructure.telegram import TelegramBot
from src.infrastructure.live_monitor import LiveMarketMonitor
from src.brain.ai_cortex import AICortex
from src.execution.trader import Trader
from src.utils.position_manager import PositionManager
from src.utils.signal_tracker import SignalPerformanceTracker
from src.utils.daily_summary import DailySummaryReporter

# Setup Logging
logger = setup_logging("AI_PHOENIX")

class AIPhoenixBot:
    """
    AI-Powered Trading Bot with 7/24 Real-Time Monitoring
    """
    def __init__(self):
        self.running = True
        self.binance = BinanceAPI()
        self.telegram = TelegramBot()
        
        # The AI Brain
        self.cortex = AICortex(self.binance)
        self.trader = Trader(self.binance, self.telegram)
        
        # Live Monitor (WebSocket)
        self.monitor = LiveMarketMonitor(self.on_market_event)
        
        # Position & Signal Tracking
        self.position_manager = PositionManager()
        self.signal_tracker = SignalPerformanceTracker()
        
        # Daily Reporter
        self.daily_reporter = DailySummaryReporter(self.signal_tracker, self.telegram)
        
        # Track last decisions to avoid spam (with thread safety)
        self.last_decisions = {}
        self._decision_lock = asyncio.Lock()  # FIX 1.4: Thread safety
        
    async def on_market_event(self, symbol: str, reason: str):
        """
        Callback when market moves significantly.
        This is where AI gets triggered INSTANTLY.
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"⚡ LIVE EVENT: {symbol}")
            logger.info(f"Trigger: {reason}")
            logger.info(f"{'='*60}\n")
            
            # Get current price for position monitoring
            current_price = await self.binance.get_current_price(symbol)
            
            # CHECK ACTIVE POSITION FIRST
            if self.position_manager.has_active_position(symbol):
                status = self.position_manager.check_position_status(symbol, current_price)
                
                if status['should_close']:
                    # TP or SL hit!
                    position = self.position_manager.get_position(symbol)
                    
                    # Calculate PnL for notification
                    pnl_emoji = "🎯" if status['outcome'] == "TP" else "🛑"
                    pnl_text = f"{status['pnl']:+.2f}%"
                    
                    await self.telegram.send_message(
                        f"{pnl_emoji} **{status['outcome']} HIT!**\n"
                        f"━━━━━━━━━━━━━━━━━━\n"
                        f"Symbol: {symbol}\n"
                        f"Position: {position.signal_type}\n"
                        f"Entry: ${position.entry_price:,.2f}\n"
                        f"Exit: ${current_price:,.2f}\n"
                        f"PnL: {pnl_text}\n"
                        f"Confidence: {position.confidence}/10"
                    )
                    
                    # Update signal tracker
                    pnl_absolute = abs(current_price - position.entry_price) * 100  # Example, adjust based on position size
                    self.signal_tracker.update_outcome(
                        position.signal_id,
                        current_price,
                        status['outcome'],
                        pnl_absolute
                    )
                    
                    # Close position
                    self.position_manager.close_position(symbol, current_price)
                    
                elif status['status'] == 'REVERSAL_WARNING':
                    # Send reversal warning (only sent once)
                    await self.telegram.send_message(status['message'])
                    
                else:
                    # Still monitoring
                    logger.info(status['message'])
                
                # Don't send new signal while position active
                return
            
            # Run AI analysis (ALWAYS analyze)
            decision = await self.cortex.think(symbol)
        
            logger.info(f"📋 AI DECISION for {symbol}:")
            logger.info(f"Position: {decision.position}")
            logger.info(f"Confidence: {decision.confidence}/10")
            logger.info(f"\n{decision.reasoning}\n")
            
            # Use centralized configuration
            if decision.confidence < Config.MIN_CONFIDENCE_FOR_NOTIFICATION:
                logger.info(f"🔇 Low confidence ({decision.confidence}/10) - no notification sent")
                logger.info(f"💤 Waiting for stronger signal (minimum: {Config.MIN_CONFIDENCE_FOR_NOTIFICATION}/10)")
                return
            
            # Check if this is a new/changed decision (THREAD-SAFE)
            current_decision_key = f"{symbol}_{decision.position}_{decision.confidence}"
            
            async with self._decision_lock:  # FIX 1.4: Race condition protection
                should_notify = (current_decision_key != self.last_decisions.get(symbol))
            
            if not should_notify:
                logger.info("🔇 No notification (same as previous analysis)")
            else:
                # Translate risk to Turkish
                risk_tr = {"HIGH": "YÜKSEK", "MEDIUM": "ORTA", "LOW": "DÜŞÜK"}.get(decision.risk_level, decision.risk_level)
                
                # Extract Claude's professional trade setup if available
                trade_setup = ""
                if isinstance(decision.entry_conditions, dict):
                    # Claude's professional trade setup
                    trade_setup += f"\n\n💹 *TRADE SETUP (Claude):*\n"
                    if decision.entry_conditions.get('entry_price'):
                        trade_setup += f"Entry: {decision.entry_conditions['entry_price']}\n"
                    if decision.entry_conditions.get('stop_loss'):
                        trade_setup += f"Stop Loss: {decision.entry_conditions['stop_loss']}\n"
                    if decision.entry_conditions.get('target_1'):
                        trade_setup += f"Target 1: {decision.entry_conditions['target_1']}\n"
                    if decision.entry_conditions.get('target_2'):
                        trade_setup += f"Target 2: {decision.entry_conditions['target_2']}\n"
                    if decision.entry_conditions.get('risk_reward'):
                        trade_setup += f"R:R: {decision.entry_conditions['risk_reward']}\n"
                    if decision.entry_conditions.get('conviction'):
                        trade_setup += f"Conviction: {decision.entry_conditions['conviction']}/10\n"
                    if decision.entry_conditions.get('market_view'):
                        trade_setup += f"\n📝 *View:* {decision.entry_conditions['market_view']}\n"
                    if decision.entry_conditions.get('reasoning'):
                        reasoning = decision.entry_conditions['reasoning']
                        if len(reasoning) > 200:
                            reasoning = reasoning[:197].rsplit(' ', 1)[0] + "..."
                        trade_setup += f"💡 *Reason:* {reasoning}"
                else:
                    # Fallback to old format
                    entry_text = str(decision.entry_conditions)
                    if isinstance(decision.entry_conditions, list):
                        entry_text = "\n• " + "\n• ".join(decision.entry_conditions)
                    trade_setup = f"\n\n📋 *Entry Conditions:*{entry_text}"
                
                # Build SINGLE unified notification message
                emoji = "🟢" if decision.position == "LONG" else "🔴"
                message = (
                    f"{emoji} *{symbol} Trading Signal*\n"
                    f"⚡ *LIVE TRIGGER*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Event: {reason}\n\n"
                    f"{decision.get_consensus_report()}\n\n"
                    f"✅ *Final Decision: {decision.position}*\n"
                    f"Confidence: {decision.confidence}/10\n"
                    f"Risk: {risk_tr}"
                    f"{trade_setup}"
                )
                
                # Send SINGLE Telegram notification (no duplicate!)
                await self.telegram.send_message(message)
                
                logger.info("📱 Telegram notification sent (unified message)")
            
            # Track ALL signals sent to Telegram (not just high-confidence)
            if decision.position != "CASH":
                # Extract TP/SL from Claude's trade setup
                entry_price = current_price
                stop_loss = None
                take_profit = None
                
                if isinstance(decision.entry_conditions, dict):
                    stop_loss = decision.entry_conditions.get('stop_loss')
                    take_profit = decision.entry_conditions.get('target_1') or decision.entry_conditions.get('target_2')
                
                # Fallback: Calculate TP/SL if not provided
                if not stop_loss:
                    if decision.position == "LONG":
                        stop_loss = entry_price * 0.98  # -2%
                        take_profit = entry_price * 1.04 if not take_profit else take_profit  # +4%
                    else:  # SHORT
                        stop_loss = entry_price * 1.02  # +2%
                        take_profit = entry_price * 0.96 if not take_profit else take_profit  # -4%
                
                # Extract AI votes for tracking
                ai_votes_dict = {vote.name: vote.vote for vote in decision.votes}
                
                # Log signal to tracker
                signal_id = self.signal_tracker.log_signal(
                    symbol,
                    decision.position,
                    entry_price,
                    ai_votes_dict,
                    decision.confidence
                )
                
                # Open position in manager (ANTI-SPAM FOR ALL SIGNALS!)
                self.position_manager.open_position(
                    symbol,
                    decision.position,
                    entry_price,
                    stop_loss,
                    take_profit,
                    signal_id,
                    decision.confidence
                )
                logger.info(f"📝 Position tracked: {decision.position} @ ${entry_price} (TP: ${take_profit}, SL: ${stop_loss})")
            
            # Execute trade ONLY if confidence high enough (≥7)
            if decision.confidence >= 7 and decision.position != "CASH":
                signal = {
                    "action": "BUY" if decision.position == "LONG" else "SELL",
                    "entry_price": current_price,
                    "stop_loss": decision.entry_conditions.get('stop_loss') if isinstance(decision.entry_conditions, dict) else None,
                    "reason": f"Live AI ({reason})",
                    "cortex_note": decision.reasoning,
                    "ai_votes": {vote.name: vote.vote for vote in decision.votes},
                    "confidence": decision.confidence
                }
                await self.trader.execute(symbol, signal)
                logger.info(f"✅ HIGH CONFIDENCE - Trade executed")
            else:
                logger.info(f"💤 Low confidence ({decision.confidence}/10) - position tracked but no trade execution")
                
        except Exception as e:
            logger.error(f"Error in market event handler: {e}")
            import traceback
            traceback.print_exc()
        
    async def _daily_scheduler_task(self):
        """Background task to check for daily summary time"""
        logger.info("📅 Daily scheduler started")
        while self.running:
            now = datetime.now()
            # Send report at 23:55 (just before day end)
            if now.hour == 23 and now.minute == 55:
                logger.info("⏰ Triggering daily summary...")
                await self.daily_reporter.generate_and_send()
                # Wait 61 seconds to avoid double trigger
                await asyncio.sleep(61)
            else:
                # Check every minute
                await asyncio.sleep(60)

    async def cleanup(self):
        logger.info("🛑 Shutting down AI Phoenix...")
        await self.monitor.stop()
        await self.binance.close()
        await self.telegram.send_alert("SYSTEM", "AI Phoenix Stopping...", "🛑")
        
    async def run(self):
        logger.info(f"🔥 AI Phoenix v{Config.VERSION} Starting...")
        logger.info("🤖 Powered by: Gemini Vision + Claude + GPT-4 + DeepSeek")
        logger.info("📡 Mode: LIVE 7/24 WebSocket Monitoring")
        logger.info("🔇 Smart Notifications: Only on real changes")
        
        # Verify Config
        if not Config.validate():
            logger.error("❌ Critical config missing. Exiting.")
            return

        # Connect
        await self.binance.connect()
        await self.telegram.send_alert(
            f"Versiyon: {Config.VERSION}\n"
            f"AI Takımı: Claude 4 Sonnet, GPT-4, DeepSeek (Gemini Kapalı)\n"
            f"Mod: 🔴 CANLI 7/24 WebSocket\n"
            f"Hedefler: BTCUSDT, ETHUSDT\n\n"
            f"⚡ Tetikleyiciler:\n"
            f"• Fiyat ±0.5% hareket\n"
            f"• Hacim artışı 2x+\n"
            f"• Saatlik kontrol\n\n"
            f"🔇 Akıllı Bildirimler:\n"
            f"• Sadece pozisyon/güven değişikliklerinde\n"
            f"• Aynı analiz tekrarı yok", 
            "🟢"
        )
        
        # Start Daily Scheduler
        asyncio.create_task(self._daily_scheduler_task())
        
        # Start WebSocket monitoring (blocks here)
        try:
            await self.monitor.start()
        except asyncio.CancelledError:
            logger.info("Bot cancelled, shutting down...")

if __name__ == "__main__":
    bot = AIPhoenixBot()
    
    # Graceful Shutdown
    def handle_exit(sig, frame):
        bot.running = False
        
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(bot.cleanup())
