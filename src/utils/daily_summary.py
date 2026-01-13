import logging
import asyncio
from datetime import datetime, date
from src.utils.signal_tracker import SignalPerformanceTracker
from src.infrastructure.telegram import TelegramBot

logger = logging.getLogger("DAILY_SUMMARY")

class DailySummaryReporter:
    """
    Generates and sends daily performance reports to Telegram.
    """
    def __init__(self, tracker: SignalPerformanceTracker, telegram: TelegramBot):
        self.tracker = tracker
        self.telegram = telegram
        
    async def generate_and_send(self):
        """Generate daily report and send it to Telegram"""
        try:
            today_str = date.today().isoformat()
            
            # Filter signals for today
            today_signals = []
            today_completed = []
            
            for sig in self.tracker.signals.values():
                # Parse timestamp (ISO format)
                try:
                    sig_date = datetime.fromisoformat(sig.timestamp).date()
                    if sig_date.isoformat() == today_str:
                        today_signals.append(sig)
                        if sig.outcome:
                            today_completed.append(sig)
                except:
                    continue
            
            if not today_signals:
                logger.info("No signals today, skipping summary.")
                return

            # Calculate stats
            total_signals = len(today_signals)
            completed_count = len(today_completed)
            
            wins = len([s for s in today_completed if s.outcome == "TP"])
            losses = len([s for s in today_completed if s.outcome == "SL"])
            
            win_rate = (wins / completed_count * 100) if completed_count > 0 else 0
            total_pnl = sum(s.pnl for s in today_completed if s.pnl)
            
            # Identify Best/Worst
            best_trade = max(today_completed, key=lambda x: x.pnl_pct) if today_completed else None
            worst_trade = min(today_completed, key=lambda x: x.pnl_pct) if today_completed else None
            
            # Format Message
            message = (
                f"📅 **GÜNLÜK PERFORMANS RAPORU**\n"
                f"🗓️ {today_str}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"📊 **Genel Durum:**\n"
                f"• Toplam Sinyal: {total_signals}\n"
                f"• Tamamlanan: {completed_count}\n"
                f"• Başarı Oranı: %{win_rate:.1f}\n"
                f"💰 **Net PnL:** ${total_pnl:+.2f}\n"
                f"━━━━━━━━━━━━━━━━━━\n"
            )
            
            if best_trade:
                message += (
                    f"🏆 **Günün Yıldızı:**\n"
                    f"{best_trade.symbol} {best_trade.signal_type} (+%{best_trade.pnl_pct:.1f})\n\n"
                )
                
            if worst_trade and worst_trade.pnl_pct < 0:
                message += (
                    f"💀 **Günün Kaybı:**\n"
                    f"{worst_trade.symbol} {worst_trade.signal_type} (%{worst_trade.pnl_pct:.1f})\n\n"
                )
            
            # Add AI Performance Note
            message += f"🤖 AI bugün {total_signals} işlemden {wins} tanesini doğru bildi."
            
            # Send
            await self.telegram.send_message(message)
            logger.info("✅ Daily summary sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            import traceback
            traceback.print_exc()
