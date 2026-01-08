import logging
import asyncio
from typing import Dict
from src.infrastructure.binance_api import BinanceAPI
from src.infrastructure.telegram import TelegramBot
from src.utils.signal_tracker import SignalPerformanceTracker
from src.config import Config

logger = logging.getLogger("TRADER")

class Trader:
    """
    The Executioner with Self-Learning.
    """
    def __init__(self, binance: BinanceAPI, telegram: TelegramBot):
        self.binance = binance
        self.telegram = telegram
        self.active_positions = {}  # symbol -> {entry, size, stop, signal_id}
        self.tracker = SignalPerformanceTracker()
        
    async def execute(self, symbol: str, signal: dict):
        action = signal["action"]
        
        if action == "HOLD":
            await self._manage_position(symbol, signal)
            return
            
        # Check if already in position
        if symbol in self.active_positions:
            current_side = "BUY" if self.active_positions[symbol]['size'] > 0 else "SELL"
            if (current_side == "BUY" and action == "SELL") or (current_side == "SELL" and action == "BUY"):
                await self._close_position(symbol, "Signal Flip")
            else:
                return
                
        if action in ["BUY", "SELL"]:
            await self._open_position(symbol, action, signal)
            
    async def _open_position(self, symbol: str, side: str, signal: dict):
        balance = await self.binance.get_balance()
        if balance < 10:
            logger.warning(f"Insufficient funds ({balance} USDT)")
            return
            
        # Risk Management
        entry_price = signal["entry_price"]
        stop_loss = signal.get("stop_loss", entry_price * 0.97 if side == "BUY" else entry_price * 1.03)
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0: return
        
        risk_amount = balance * 0.02
        qty = risk_amount / risk_per_share
        max_qty = (balance * 0.5) / entry_price
        qty = min(qty, max_qty)
        
        if qty * entry_price < 6:
            return
            
        # Log signal to tracker
        ai_votes = signal.get("ai_votes", {})
        confidence = signal.get("confidence", 5)
        signal_id = self.tracker.log_signal(
            symbol, 
            side, 
            entry_price, 
            ai_votes, 
            confidence
        )
        
        logger.info(f"🚀 OPENING {side} {symbol} | Qty: {qty:.4f} | Entry: {entry_price} | Signal: {signal_id}")
        
        # Real trade would go here
        # await self.binance.create_order(symbol, side, qty)
        
        self.active_positions[symbol] = {
            "entry": entry_price,
            "size": qty if side == "BUY" else -qty,
            "stop": stop_loss,
            "side": side,
            "signal_id": signal_id  # Track signal ID
        }
        
        await self.telegram.send_alert(
            f"🚀 {side} SIGNAL: {symbol}",
            f"Entry: {entry_price}\nStop: {stop_loss}\nReason: {signal['reason']}\nSignal ID: {signal_id}",
            color="🟢" if side == "BUY" else "🔴"
        )
        
    async def _close_position(self, symbol: str, reason: str):
        if symbol not in self.active_positions: return
        
        pos = self.active_positions[symbol]
        price = await self.binance.get_current_price(symbol)
        
        # Calculate PnL
        pnl = (price - pos['entry']) * pos['size']
        
        # Determine outcome
        if "Stop Loss" in reason or pnl < 0:
            outcome = "SL"
        else:
            outcome = "TP"
        
        # Update tracker
        if 'signal_id' in pos:
            self.tracker.update_outcome(pos['signal_id'], price, outcome, pnl)
        
        logger.info(f"💰 CLOSING {symbol} | PnL: {pnl:.2f} | {outcome} | Reason: {reason}")
        
        # Real trade close would go here
        # await self.binance.create_order(...)
        
        del self.active_positions[symbol]
        
        # Get updated stats
        stats = self.tracker.get_performance_stats()
        
        color = "🟢" if pnl > 0 else "🔴"
        await self.telegram.send_alert(
            f"{color} POSITION CLOSED: {symbol}",
            f"Outcome: {outcome}\nPnL: {pnl:.2f} USDT\nReason: {reason}\n\n"
            f"📊 Bot Stats:\n"
            f"Win Rate: {stats.get('win_rate', 0)}%\n"
            f"Total Trades: {stats.get('total_trades', 0)}",
            color=color
        )
        
    async def _manage_position(self, symbol: str, signal: dict):
        """Trailing Stop Handling"""
        if symbol not in self.active_positions: return
        
        pos = self.active_positions[symbol]
        current_price = await self.binance.get_current_price(symbol)
        
        new_stop = signal.get("stop_loss", pos['stop'])
        
        if pos['side'] == "BUY":
            if new_stop > pos['stop']:
                pos['stop'] = new_stop
                logger.info(f"🛡️ Trailing Stop Updated {symbol}: {new_stop}")
            
            if current_price < pos['stop']:
                await self._close_position(symbol, "Stop Loss Hit")
                
        elif pos['side'] == "SELL":
            if new_stop < pos['stop']:
                pos['stop'] = new_stop
                logger.info(f"🛡️ Trailing Stop Updated {symbol}: {new_stop}")
                
            if current_price > pos['stop']:
                await self._close_position(symbol, "Stop Loss Hit")
    
    def get_performance_feedback(self) -> str:
        """Get AI feedback for self-learning"""
        return self.tracker.get_ai_feedback_prompt()
