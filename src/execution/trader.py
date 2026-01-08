import logging
import asyncio
from typing import Dict
from src.infrastructure.binance_api import BinanceAPI
from src.infrastructure.telegram import TelegramBot
from src.config import Config

logger = logging.getLogger("TRADER")

class Trader:
    """
    The Executioner.
    Manages positions and talks to the Exchange.
    """
    def __init__(self, binance: BinanceAPI, telegram: TelegramBot):
        self.binance = binance
        self.telegram = telegram
        self.active_positions = {} # symbol -> {entry: float, size: float, stop: float}
        
    async def execute(self, symbol: str, signal: dict):
        action = signal["action"]
        
        if action == "HOLD":
            await self._manage_position(symbol, signal)
            return
            
        # Check if already in position
        if symbol in self.active_positions:
            # If opposite signal, close current
            current_side = "BUY" if self.active_positions[symbol]['size'] > 0 else "SELL"
            if (current_side == "BUY" and action == "SELL") or (current_side == "SELL" and action == "BUY"):
                await self._close_position(symbol, "Signal Flip")
            else:
                return # Already in same direction
                
        if action in ["BUY", "SELL"]:
            await self._open_position(symbol, action, signal)
            
    async def _open_position(self, symbol: str, side: str, signal: dict):
        balance = await self.binance.get_balance()
        if balance < 10:
            logger.warning(f"Insufficient funds ({balance} USDT)")
            return
            
        # Risk Management: Risk 2% of account
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0: return
        
        risk_amount = balance * 0.02 # 2% risk
        qty = risk_amount / risk_per_share
        
        # Simple size cap (max 50% of balance leverage 1x equivalent)
        max_qty = (balance * 0.5) / entry_price
        qty = min(qty, max_qty)
        
        if qty * entry_price < 6: # Min notional
            return
            
        # Execute Real Trade (Placeholder for now until User enables full Real Trading flag)
        # For now, we simulate formatting the order
        logger.info(f"🚀 OPENING {side} {symbol} | Qty: {qty:.4f} | Entry: {entry_price}")
        
        # Real API call would go here:
        # await self.binance.create_order(symbol, side, qty)
        
        self.active_positions[symbol] = {
            "entry": entry_price,
            "size": qty if side == "BUY" else -qty,
            "stop": stop_loss,
            "side": side
        }
        
        await self.telegram.send_alert(
            f"🚀 {side} SIGNAL: {symbol}",
            f"Entry: {entry_price}\nStop: {stop_loss}\nReason: {signal['reason']}\nBrain: {signal['cortex_note']}",
            color="🟢" if side == "BUY" else "🔴"
        )
        
    async def _close_position(self, symbol: str, reason: str):
        if symbol not in self.active_positions: return
        
        pos = self.active_positions[symbol]
        price = await self.binance.get_current_price(symbol)
        
        # Calculate PnL
        pnl = (price - pos['entry']) * pos['size']
        
        logger.info(f"💰 CLOSING {symbol} | PnL: {pnl:.2f} | Reason: {reason}")
        
        # await self.binance.create_order(symbol, "SELL" if pos['side']=="BUY" else "BUY", abs(pos['size']))
        
        del self.active_positions[symbol]
        
        color = "🟢" if pnl > 0 else "🔴"
        await self.telegram.send_alert(
            f"{color} POSITION CLOSED: {symbol}",
            f"PnL: {pnl:.2f} USDT\nReason: {reason}",
            color=color
        )
        
    async def _manage_position(self, symbol: str, signal: dict):
        """Trailing Stop Handling"""
        if symbol not in self.active_positions: return
        
        pos = self.active_positions[symbol]
        current_price = await self.binance.get_current_price(symbol)
        
        # Update trailing stop if price moved favorably
        new_stop = signal.get("stop_loss", pos['stop'])
        
        if pos['side'] == "BUY":
            if new_stop > pos['stop']: # Move stop UP only
                pos['stop'] = new_stop
                logger.info(f"🛡️ Trailing Stop Updated {symbol}: {new_stop}")
            
            if current_price < pos['stop']:
                await self._close_position(symbol, "Stop Loss Hit")
                
        elif pos['side'] == "SELL":
            if new_stop < pos['stop']: # Move stop DOWN only
                pos['stop'] = new_stop
                logger.info(f"🛡️ Trailing Stop Updated {symbol}: {new_stop}")
                
            if current_price > pos['stop']:
                await self._close_position(symbol, "Stop Loss Hit")
