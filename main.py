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
        
        # Track last decisions to avoid spam
        self.last_decisions = {}  # {symbol: {position, confidence, votes_hash}}
        
    def _should_notify(self, symbol: str, decision) -> bool:
        """
        Check if this decision is different enough to warrant a notification.
        Only notify if there's a REAL change.
        """
        if symbol not in self.last_decisions:
            # First decision for this symbol - notify
            return True
        
        last = self.last_decisions[symbol]
        
        # Check for significant changes
        position_changed = last['position'] != decision.position
        confidence_changed = abs(last['confidence'] - decision.confidence) >= 2
        
        # Hash of AI votes to detect vote changes
        current_votes = ",".join([f"{v.name}:{v.vote}" for v in decision.votes])
        votes_changed = last.get('votes_hash') != current_votes
        
        # Notify if ANY significant change
        if position_changed or confidence_changed or votes_changed:
            return True
        
        return False
    
    def _update_last_decision(self, symbol: str, decision):
        """Save this decision for comparison"""
        votes_hash = ",".join([f"{v.name}:{v.vote}" for v in decision.votes])
        self.last_decisions[symbol] = {
            'position': decision.position,
            'confidence': decision.confidence,
            'votes_hash': votes_hash
        }
        
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
            
            # HIGH CONFIDENCE FILTER
            MIN_CONFIDENCE_FOR_NOTIFICATION = 5  # Lowered from 6 to get more signals
            
            if decision.confidence < MIN_CONFIDENCE_FOR_NOTIFICATION:
                logger.info(f"🔇 Low confidence ({decision.confidence}/10) - no notification sent")
                logger.info(f"💤 Waiting for stronger signal (minimum: {MIN_CONFIDENCE_FOR_NOTIFICATION}/10)")
                return
            
            # Check if this is a new/changed decision
            current_decision_key = f"{symbol}_{decision.position}_{decision.confidence}"
            
            if current_decision_key == self.last_decisions.get(symbol):
                logger.info("🔇 Bildirim yok (önceki analizle aynı)")
            else:
                # Translate risk to Turkish
                risk_tr = {"HIGH": "YÜKSEK", "MEDIUM": "ORTA", "LOW": "DÜŞÜK"}.get(decision.risk_level, decision.risk_level)
                
                # Format entry conditions as readable text
                entry_text = decision.entry_conditions
                if isinstance(entry_text, list):
                    entry_text = "\n• " + "\n• ".join(entry_text)
                
                # Build notification message
                message = (
                    f"⚡ *CANLI TETİKLEME: {symbol}*\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Olay: {reason}\n\n"
                    f"{decision.get_consensus_report()}\n\n"
                    f"✅ *Nihai Karar: {decision.position}*\n"
                    f"Güven: {decision.confidence}/10\n"
                    f"Risk: {risk_tr}\n\n"
                    f"📋 *Giriş Koşulları:*{entry_text}"
                )
                
                # Add stop loss/take profit if available
                if decision.stop_loss and decision.take_profit:
                    message += f"\n\n🎯 *Risk Yönetimi:*\n"
                    message += f"Stop Loss: ${decision.stop_loss:,.2f}\n"
                    message += f"Take Profit: ${decision.take_profit:,.2f}"
                
                # Add position size if available
                if decision.position_size:
                    message += f"\n💰 Pozisyon: {decision.position_size:.4f} {symbol[:3]}"
                
                # Send Telegram notification
                await self.telegram.send_message(message)
                
                logger.info("📱 Telegram bildirimi gönderildi (değişiklik tespit edildi)")
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
