import asyncio
import logging
import signal
import sys
from src.config import Config
from src.utils.logger import setup_logging
from src.infrastructure.binance_api import BinanceAPI
from src.infrastructure.telegram import TelegramBot
from src.brain.ai_cortex import AICortex
from src.execution.trader import Trader

# Setup Logging
logger = setup_logging("AI_PHOENIX")

class AIPhoenixBot:
    """
    AI-Powered Trading Bot with True Intelligence.
    """
    def __init__(self):
        self.running = True
        self.binance = BinanceAPI()
        self.telegram = TelegramBot()
        
        # The AI Brain
        self.cortex = AICortex(self.binance)
        self.trader = Trader(self.binance, self.telegram)
        
    async def cleanup(self):
        logger.info("🛑 Shutting down AI Phoenix...")
        await self.binance.close()
        await self.telegram.send_alert("SYSTEM", "AI Phoenix Stopping...", "🛑")
        
    async def run(self):
        logger.info(f"🔥 AI Phoenix v{Config.VERSION} Starting...")
        logger.info("🤖 Powered by: Gemini Vision + Claude + GPT-4")
        
        # Verify Config
        if not Config.validate():
            logger.error("❌ Critical config missing. Exiting.")
            return

        # Connect
        await self.binance.connect()
        await self.telegram.send_alert(
            "🤖 AI PHOENIX ONLINE", 
            f"Version: {Config.VERSION}\nAI Stack: Gemini Vision, Claude 3.5, GPT-4\nTargets: BTC, ETH", 
            "🟢"
        )
        
        # Main AI Loop
        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"🧠 AI CYCLE #{iteration}")
                logger.info(f"{'='*60}")
                
                # Analyze each symbol with full AI stack
                for symbol in ["BTCUSDT", "ETHUSDT"]:
                    logger.info(f"\n🎯 Analyzing {symbol}...")
                    
                    # AI Cortex Decision
                    decision = await self.cortex.think(symbol)
                    
                    # Log the reasoning
                    logger.info(f"\n📋 AI REPORT for {symbol}:")
                    logger.info(f"Position: {decision.position}")
                    logger.info(f"Confidence: {decision.confidence}/10")
                    logger.info(f"Risk Level: {decision.risk_level}")
                    logger.info(f"\nReasoning:\n{decision.reasoning}")
                    logger.info(f"\nEntry Conditions: {decision.entry_conditions}")
                    
                    # Execute if confidence is high enough
                    if decision.confidence >= 7 and decision.position != "CASH":
                        signal = {
                            "action": "BUY" if decision.position == "LONG" else "SELL",
                            "entry_price": await self.binance.get_current_price(symbol),
                            "stop_loss": 0,  # Trader will calculate
                            "reason": f"AI Decision (Confidence: {decision.confidence}/10)",
                            "cortex_note": decision.reasoning
                        }
                        await self.trader.execute(symbol, signal)
                    else:
                        logger.info(f"💤 {symbol}: Holding (Low confidence or CASH directive)")
                
                # Report every 4 hours
                if iteration % 4 == 0:
                    await self.telegram.send_message(
                        f"🤖 *AI Phoenix Status*\n"
                        f"Cycle: {iteration}\n"
                        f"Active Positions: {len(self.trader.active_positions)}\n"
                        f"All AI models operational ✅"
                    )
                
                # Wait 1 hour
                logger.info("\n⏳ Sleeping for 1 hour... (Next AI cycle in 60 min)")
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ CRITICAL LOOP ERROR: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3600)

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
