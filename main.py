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
            
            # Run AI analysis
            decision = await self.cortex.think(symbol)
            
            # Log reasoning
            logger.info(f"📋 AI DECISION for {symbol}:")
            logger.info(f"Position: {decision.position}")
            logger.info(f"Confidence: {decision.confidence}/10")
            logger.info(f"\n{decision.reasoning}\n")
            
            # Send Telegram notification with AI voting results
            await self.telegram.send_message(
                f"⚡ *LIVE TRIGGER: {symbol}*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"Event: {reason}\n\n"
                f"{decision.get_consensus_report()}\n\n"
                f"✅ *Final Decision: {decision.position}*\n"
                f"Confidence: {decision.confidence}/10\n"
                f"Risk: {decision.risk_level}\n\n"
                f"_{decision.entry_conditions}_"
            )
            
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
                    "ai_votes": ai_votes_dict,  # For performance tracking
                    "confidence": decision.confidence  # For performance tracking
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
        logger.info("🤖 Powered by: Gemini Vision + Claude + GPT-4")
        logger.info("📡 Mode: LIVE 7/24 WebSocket Monitoring")
        
        # Verify Config
        if not Config.validate():
            logger.error("❌ Critical config missing. Exiting.")
            return

        # Connect
        await self.binance.connect()
        await self.telegram.send_alert(
            "🤖 AI PHOENIX ONLINE", 
            f"Version: {Config.VERSION}\n"
            f"AI Stack: Gemini Vision, Claude 3.5, GPT-4\n"
            f"Mode: 🔴 LIVE 7/24 WebSocket\n"
            f"Targets: BTCUSDT, ETHUSDT\n\n"
            f"⚡ Triggers:\n"
            f"• Price ±0.5% move\n"
            f"• Volume spike 2x+\n"
            f"• Hourly fallback check", 
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
