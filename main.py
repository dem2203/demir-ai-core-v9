import asyncio
import logging
import signal
import sys
from src.config import Config
from src.utils.logger import setup_logging
from src.infrastructure.binance_api import BinanceAPI
from src.infrastructure.telegram import TelegramBot
from src.infrastructure.live_monitor import LiveMarketMonitor
from src.brain.ai_cortex import AICortex
from src.execution.trader import Trader

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
                logger.info("🔇 Bildirim yok (önceki analizle aynı)")
            else:
                # Translate risk to Turkish
                risk_tr = {"HIGH": "YÜKSEK", "MEDIUM": "ORTA", "LOW": "DÜŞÜK"}.get(decision.risk_level, decision.risk_level)
                
                # Extract Claude's professional trade setup if available
                # entry_conditions now contains Claude's JSON response with entry_price, stop_loss, targets, etc.
                trade_setup = ""
                if isinstance(decision.entry_conditions, dict):
                    # Claude's professional trade setup
                    trade_setup += f"\n💹 *TRADE SETUP (Claude):*\n"
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
                        trade_setup += f"\n📝 *Görüş:* {decision.entry_conditions['market_view']}\n"
                    if decision.entry_conditions.get('reasoning'):
                    # FIX 2.4: Safe truncation preserving words and markdown
                        reasoning = decision.entry_conditions['reasoning']
                        if len(reasoning) > 200:
                            reasoning = reasoning[:197].rsplit(' ', 1)[0] + "..."
                        trade_setup += f"💡 *Reason:* {reasoning}"
                    
                    await self.telegram.send_alert(
                        f"🤖 {symbol} Trading Signal",
                        trade_setup,
                        color="🟢" if decision.position == "LONG" else "🔴"
                    )
                else:
                    # Fallback to old format
                    entry_text = str(decision.entry_conditions)
                    if isinstance(decision.entry_conditions, list):
                        entry_text = "\n• " + "\n• ".join(decision.entry_conditions)
                    trade_setup = f"\n📋 *Giriş Koşulları:*{entry_text}"
                
                # Build notification message
                message = (
                    f"⚡ *CANLI TETİKLEME: {symbol}*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Olay: {reason}\n\n"
                    f"{decision.get_consensus_report()}\n\n"
                    f"✅ *Nihai Karar: {decision.position}*\n"
                    f"Güven: {decision.confidence}/10\n"
                    f"Risk: {risk_tr}"
                    f"{trade_setup}"
                )
                
                # Send Telegram notification
                await self.telegram.send_message(message)
                
                logger.info("📱 Telegram bildirimi gönderildi (değişiklik tespit edildi)")
                async with self._decision_lock:  # FIX 1.4: Thread-safe update
                    self.last_decisions[symbol] = current_decision_key
            
            # Execute if confidence high enough
            if decision.confidence >= 7 and decision.position != "CASH":
                # Extract AI votes for tracking
                ai_votes_dict = {vote.name: vote.vote for vote in decision.votes}
                
                signal = {
                    "action": "BUY" if decision.position == "LONG" else "SELL",
                    "entry_price": await self.binance.get_current_price(symbol),
                    "stop_loss": 0,
                    "reason": f"Live AI ({reason})",
                    "cortex_note": decision.reasoning,
                    "ai_votes": ai_votes_dict,
                    "confidence": decision.confidence
                }
                await self.trader.execute(symbol, signal)
            else:
                logger.info(f"💤 Low confidence or CASH - holding position")
                
        except Exception as e:
            logger.error(f"Error in market event handler: {e}")
            import traceback
            traceback.print_exc()
        
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
            "🤖 AI PHOENIX AKTİF", 
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
