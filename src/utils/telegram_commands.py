# -*- coding: utf-8 -*-
"""
DEMIR AI - TELEGRAM BOT COMMANDS
================================
Interactive Telegram bot commands: /status, /portfolio, /predict, /help

Kullanım:
- /status - Sistem durumu
- /portfolio - Paper trading özeti
- /predict BTC - Manuel tahmin isteği
- /backtest BTC 30 - Son 30 günlük backtest
- /help - Komut listesi
"""
import logging
import asyncio
from typing import Optional, Callable
from datetime import datetime

logger = logging.getLogger("TELEGRAM_BOT")


class TelegramBotCommands:
    """Interactive Telegram bot command handler"""
    
    COMMANDS = {
        '/status': 'Sistem durumunu göster',
        '/portfolio': 'Paper trading özeti',
        '/predict': 'Manuel tahmin: /predict BTC',
        '/backtest': 'Backtest: /backtest BTC 30',
        '/stats': 'AI performans istatistikleri',
        '/help': 'Komut listesi'
    }
    
    def __init__(self, notifier=None):
        """
        Args:
            notifier: SmartNotifier instance for sending messages
        """
        self.notifier = notifier
        self._engine = None
    
    def set_engine(self, engine):
        """Set engine reference for data access"""
        self._engine = engine
    
    async def handle_command(self, command: str, args: list = None) -> str:
        """
        Handle incoming command
        
        Args:
            command: Command string (e.g., '/status')
            args: Command arguments (e.g., ['BTC', '30'])
        
        Returns:
            Response message
        """
        args = args or []
        
        handlers = {
            '/status': self._cmd_status,
            '/portfolio': self._cmd_portfolio,
            '/predict': self._cmd_predict,
            '/backtest': self._cmd_backtest,
            '/stats': self._cmd_stats,
            '/help': self._cmd_help
        }
        
        handler = handlers.get(command.lower())
        if handler:
            return await handler(args)
        else:
            return f"❓ Bilinmeyen komut: {command}\n/help ile komutları görün"
    
    async def _cmd_status(self, args: list) -> str:
        """System status"""
        try:
            from src.execution.paper_trader import get_paper_trader
            trader = get_paper_trader()
            
            positions = len(trader.portfolio.get('positions', {}))
            balance = trader.portfolio.get('balance', 0)
            
            return f"""
📊 *SİSTEM DURUMU*

🟢 Engine: Çalışıyor
💰 Balance: ${balance:,.2f}
📈 Açık Pozisyon: {positions}
⏰ Zaman: {datetime.now().strftime('%H:%M:%S')}

🤖 AI Modülleri:
• LSTM Predictor: ✅ Aktif
• RL Agent: ✅ Aktif
• Pattern Engine: ✅ Aktif
"""
        except Exception as e:
            return f"❌ Status hatası: {e}"
    
    async def _cmd_portfolio(self, args: list) -> str:
        """Portfolio summary"""
        try:
            from src.execution.paper_trader import get_paper_trader
            trader = get_paper_trader()
            
            portfolio = trader.portfolio
            balance = portfolio.get('balance', 0)
            positions = portfolio.get('positions', {})
            history = portfolio.get('history', [])
            
            # Calculate P&L
            total_pnl = sum(h.get('pnl', 0) for h in history)
            wins = sum(1 for h in history if h.get('pnl', 0) > 0)
            total = len(history)
            win_rate = (wins / total * 100) if total > 0 else 0
            
            msg = f"""
💼 *PAPER TRADING PORTFÖLYÖsü*

💰 Bakiye: ${balance:,.2f}
📊 Toplam P/L: ${total_pnl:,.2f}
🎯 Win Rate: {win_rate:.1f}%
📈 Toplam İşlem: {total}

*Açık Pozisyonlar:*
"""
            if positions:
                for symbol, pos in positions.items():
                    side = pos.get('side', 'LONG')
                    entry = pos.get('entry_price', 0)
                    msg += f"• {symbol}: {side} @ ${entry:,.0f}\n"
            else:
                msg += "• Açık pozisyon yok\n"
            
            return msg
        except Exception as e:
            return f"❌ Portfolio hatası: {e}"
    
    async def _cmd_predict(self, args: list) -> str:
        """Manual prediction request"""
        if not args:
            return "❌ Kullanım: /predict BTC veya /predict ETH"
        
        symbol = args[0].upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        try:
            from src.v10.early_signal_engine import get_early_signal_engine
            engine = await get_early_signal_engine()
            
            signal = await engine.generate_signal(symbol, None)
            
            if signal:
                return f"""
🔮 *MANUEL TAHMİN: {symbol}*

📊 Aksiyon: {signal.action}
💪 Güven: {signal.confidence}%
📈 Entry: ${signal.entry_zone[0]:,.0f} - ${signal.entry_zone[1]:,.0f}
🛡️ SL: ${signal.stop_loss:,.0f}
🎯 TP: ${signal.take_profit:,.0f}
📏 R/R: {signal.risk_reward:.1f}x

📝 Neden: {signal.reasoning[:200]}...
"""
            else:
                return f"⚠️ {symbol} için şu an sinyal yok"
        except Exception as e:
            return f"❌ Tahmin hatası: {e}"
    
    async def _cmd_backtest(self, args: list) -> str:
        """Run backtest"""
        symbol = args[0].upper() if args else "BTCUSDT"
        days = int(args[1]) if len(args) > 1 else 30
        
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
        
        try:
            from src.backtest.backtester import get_backtester
            backtester = get_backtester()
            
            result = await backtester.run_backtest(symbol, days=days)
            
            return backtester.format_results_for_telegram()
        except Exception as e:
            return f"❌ Backtest hatası: {e}"
    
    async def _cmd_stats(self, args: list) -> str:
        """AI performance stats"""
        try:
            from src.brain.feedback_db import get_feedback_db
            db = get_feedback_db()
            stats = db.get_stats()
            
            return f"""
📈 *AI PERFORMANS İSTATİSTİKLERİ*

📊 Toplam İşlem: {stats.get('total_trades', 0)}
🎯 Win Rate: {stats.get('win_rate', 0)*100:.1f}%
💰 Ortalama P/L: ${stats.get('avg_pnl', 0):.2f}
🏆 En İyi: ${stats.get('best_trade', 0):.2f}
📉 En Kötü: ${stats.get('worst_trade', 0):.2f}
"""
        except Exception as e:
            return f"❌ Stats hatası: {e}"
    
    async def _cmd_help(self, args: list) -> str:
        """Command help"""
        msg = "📖 *TELEGRAM BOT KOMUTLARI*\n\n"
        for cmd, desc in self.COMMANDS.items():
            msg += f"`{cmd}` - {desc}\n"
        return msg


# Singleton
_bot_commands: Optional[TelegramBotCommands] = None


def get_telegram_bot() -> TelegramBotCommands:
    """Get or create TelegramBotCommands singleton"""
    global _bot_commands
    if _bot_commands is None:
        _bot_commands = TelegramBotCommands()
    return _bot_commands
