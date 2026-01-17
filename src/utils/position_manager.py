import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional
from src.config import Config

logger = logging.getLogger("POSITION_MANAGER")

class ActivePosition:
    """Represents an active trading position being monitored"""
    def __init__(self, symbol: str, signal_type: str, entry_price: float, 
                 stop_loss: float, take_profit: float, signal_id: str, confidence: int):
        self.symbol = symbol
        self.signal_type = signal_type  # LONG or SHORT
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.signal_id = signal_id
        self.confidence = confidence
        self.entry_timestamp = datetime.now().isoformat()
        self.last_check_timestamp = datetime.now().isoformat()
        self.last_price = entry_price
        self.reversal_warned = False  # Track if we've already warned about reversal
        
    def to_dict(self):
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "signal_id": self.signal_id,
            "confidence": self.confidence,
            "entry_timestamp": self.entry_timestamp,
            "last_check_timestamp": self.last_check_timestamp,
            "last_price": self.last_price,
            "reversal_warned": self.reversal_warned
        }
    
    @staticmethod
    def from_dict(data):
        pos = ActivePosition(
            data["symbol"],
            data["signal_type"],
            data["entry_price"],
            data["stop_loss"],
            data["take_profit"],
            data["signal_id"],
            data["confidence"]
        )
        pos.entry_timestamp = data.get("entry_timestamp", datetime.now().isoformat())
        pos.last_check_timestamp = data.get("last_check_timestamp", datetime.now().isoformat())
        pos.last_price = data.get("last_price", pos.entry_price)
        pos.reversal_warned = data.get("reversal_warned", False)
        return pos


class PositionManager:
    """
    Manages active trading positions to prevent spam and track TP/SL.
    
    Features:
    - Blocks duplicate signals for same symbol while position active
    - Real-time TP/SL monitoring
    - Reversal alerts (entry LONG but price dropping)
    - Position persistence (survives bot restarts)
    """
    def __init__(self):
        self.db_path = os.path.join(Config.DATA_DIR, "active_positions.json")
        self.positions: Dict[str, ActivePosition] = {}
        self._load_positions()
        
    def _load_positions(self):
        """Load active positions from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.positions = {
                        k: ActivePosition.from_dict(v)
                        for k, v in data.items()
                    }
                logger.info(f"📦 Loaded {len(self.positions)} active positions")
            except Exception as e:
                logger.error(f"Failed to load active positions: {e}")
                self.positions = {}
        else:
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            
    def _save_positions(self):
        """Save active positions to disk"""
        try:
            data = {k: v.to_dict() for k, v in self.positions.items()}
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save active positions: {e}")
    
    def has_active_position(self, symbol: str) -> bool:
        """Check if there's an active position for this symbol"""
        return symbol in self.positions
    
    def get_position(self, symbol: str) -> Optional[ActivePosition]:
        """Get active position for symbol"""
        return self.positions.get(symbol)
    
    def open_position(self, symbol: str, signal_type: str, entry_price: float,
                     stop_loss: float, take_profit: float, signal_id: str, confidence: int):
        """
        Open a new position.
        This should only be called after checking has_active_position() returns False.
        """
        position = ActivePosition(
            symbol, signal_type, entry_price, stop_loss, take_profit, signal_id, confidence
        )
        self.positions[symbol] = position
        self._save_positions()
        
        duration_estimate = self._estimate_duration(entry_price, take_profit, stop_loss)
        
        logger.info(
            f"🎯 Position OPENED: {symbol} {signal_type} @ ${entry_price:,.2f} | "
            f"TP: ${take_profit:,.2f} | SL: ${stop_loss:,.2f} | "
            f"Est. Duration: {duration_estimate}"
        )
        
    def check_position_status(self, symbol: str, current_price: float) -> dict:
        """
        Check position status against current price.
        
        Returns:
            dict with:
                - status: "MONITORING", "TP_HIT", "SL_HIT", "REVERSAL_WARNING"
                - message: Human-readable status message
                - pnl: Current unrealized PnL percentage
                - should_close: Boolean, True if TP or SL hit
        """
        if symbol not in self.positions:
            return {
                "status": "NO_POSITION",
                "message": "No active position",
                "pnl": 0,
                "should_close": False
            }
        
        pos = self.positions[symbol]
        pos.last_check_timestamp = datetime.now().isoformat()
        pos.last_price = current_price
        
        # Calculate current PnL
        if pos.signal_type == "LONG":
            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
            tp_hit = current_price >= pos.take_profit
            sl_hit = current_price <= pos.stop_loss
            reversal = pnl_pct < -1.5  # Entry was LONG but down 1.5%
        else:  # SHORT
            pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
            tp_hit = current_price <= pos.take_profit
            sl_hit = current_price >= pos.stop_loss
            reversal = pnl_pct < -1.5  # Entry was SHORT but price rising
        
        # Check TP
        if tp_hit:
            return {
                "status": "TP_HIT",
                "message": f"🎯 TP HIT! {symbol} {pos.signal_type} closed @ ${current_price:,.2f}",
                "pnl": pnl_pct,
                "should_close": True,
                "outcome": "TP"
            }
        
        # Check SL
        if sl_hit:
            return {
                "status": "SL_HIT",
                "message": f"🛑 SL HIT! {symbol} {pos.signal_type} closed @ ${current_price:,.2f}",
                "pnl": pnl_pct,
                "should_close": True,
                "outcome": "SL"
            }
        
        # Check Reversal (only warn once)
        if reversal and not pos.reversal_warned:
            pos.reversal_warned = True
            self._save_positions()
            return {
                "status": "REVERSAL_WARNING",
                "message": (
                    f"⚠️ REVERSAL WARNING!\n"
                    f"{symbol} {pos.signal_type} @ ${pos.entry_price:,.2f}\n"
                    f"Current: ${current_price:,.2f} ({pnl_pct:+.2f}%)\n"
                    f"Position going against entry direction!"
                ),
                "pnl": pnl_pct,
                "should_close": False
            }
        
        # Still monitoring
        return {
            "status": "MONITORING",
            "message": f"{symbol} {pos.signal_type} monitoring | PnL: {pnl_pct:+.2f}%",
            "pnl": pnl_pct,
            "should_close": False
        }
    
    def close_position(self, symbol: str, exit_price: float = None):
        """
        Close an active position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price (if None, use last known price)
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot close {symbol}: No active position")
            return
        
        pos = self.positions[symbol]
        
        if exit_price is None:
            exit_price = pos.last_price
        
        # Calculate final PnL
        if pos.signal_type == "LONG":
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        else:
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
        
        # Calculate duration
        try:
            entry_time = datetime.fromisoformat(pos.entry_timestamp)
            exit_time = datetime.now()
            duration = (exit_time - entry_time).total_seconds() / 3600
            duration_str = f"{duration:.1f}h"
        except:
            duration_str = "N/A"
        
        logger.info(
            f"🔒 Position CLOSED: {symbol} {pos.signal_type} | "
            f"Entry: ${pos.entry_price:,.2f} → Exit: ${exit_price:,.2f} | "
            f"PnL: {pnl_pct:+.2f}% | Duration: {duration_str}"
        )
        
        # Prepare trade result for RL learning
        trade_result = {
            "symbol": symbol,
            "signal_type": pos.signal_type,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "duration_hours": duration if isinstance(duration, float) else 0,
            "outcome": "WIN" if pnl_pct > 0 else "LOSS",
            "confidence": pos.confidence,
            "timestamp": pos.entry_timestamp
        }
        
        # Remove from active positions
        del self.positions[symbol]
        self._save_positions()
        
        return trade_result  # Return for RL learning
    
    def get_all_active(self) -> Dict[str, ActivePosition]:
        """Get all active positions"""
        return self.positions.copy()
    
    def _estimate_duration(self, entry: float, tp: float, sl: float) -> str:
        """Estimate how long this position might take"""
        tp_dist = abs((tp - entry) / entry) * 100
        sl_dist = abs((entry - sl) / entry) * 100
        
        # Rough heuristics based on distance
        if tp_dist > 5:
            return "12-24h (large move)"
        elif tp_dist > 3:
            return "6-12h (moderate)"
        else:
            return "2-6h (tight)"
    
    def check_correlation_risk(self, new_symbol: str, new_position: str) -> tuple[bool, str]:
        """
        Check if opening new position creates correlation risk.
        SOFT CHECK: Returns warning but does NOT block trade.
        
        Args:
            new_symbol: Symbol for new trade (e.g., 'ETHUSDT')
            new_position: Direction of new trade ('LONG' or 'SHORT')
        
        Returns:
            (allowed: bool, warning_message: str)
            - allowed is ALWAYS True (soft check)
            - warning_message explains correlation if exists
        """
        # Define correlated pairs
        correlated_pairs = [
            ('BTCUSDT', 'ETHUSDT'),  # BTC and ETH move together
        ]
        
        for pair in correlated_pairs:
            # Check if new symbol is in a correlated pair
            if new_symbol in pair:
                # Find the other symbol in the pair
                other_symbol = pair[0] if new_symbol == pair[1] else pair[1]
                
                # Check if we have active position on the correlated symbol
                if other_symbol in self.positions:
                    other_pos = self.positions[other_symbol]
                    
                    # Same direction = correlation risk (but allow it!)
                    if other_pos.signal_type == new_position:
                        warning = f"⚠️ CORRELATION NOTICE: {other_symbol} {other_pos.signal_type} already active. Opening {new_symbol} {new_position} = double exposure. Allowed if signals strong!"
                        return True, warning  # ALLOW but warn
        
        return True, "OK"  # No correlation, proceed
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """
        Update trailing stop loss if price moved favorably.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        
        Returns:
  (updated: bool, message: str)
        """
        if symbol not in self.positions:
            return False, "No active position"
        
        pos = self.positions[symbol]
        old_sl = pos.stop_loss
        
        if pos.signal_type == "LONG":
            # LONG: Price went up → move SL up (lock in profit)
            # New SL = current price - 2%
            new_sl = current_price * 0.98
            
            # Only update if new SL is higher (tighter)
            if new_sl > pos.stop_loss:
                pos.stop_loss = new_sl
                self._save_positions()
                
                pnl_locked = ((new_sl - pos.entry_price) / pos.entry_price) * 100
                return True, f"📈 Trailing SL updated: ${old_sl:,.2f} → ${new_sl:,.2f} (Profit locked: {pnl_locked:+.1f}%)"
        
        elif pos.signal_type == "SHORT":
            # SHORT: Price went down → move SL down (lock in profit)
            # New SL = current price + 2%
            new_sl = current_price * 1.02
            
            # Only update if new SL is lower (tighter)
            if new_sl < pos.stop_loss:
                pos.stop_loss = new_sl
                self._save_positions()
                
                pnl_locked = ((pos.entry_price - new_sl) / pos.entry_price) * 100
                return True, f"📉 Trailing SL updated: ${old_sl:,.2f} → ${new_sl:,.2f} (Profit locked: {pnl_locked:+.1f}%)"
        
        return False, "No update needed"
    
    def get_summary(self) -> str:
        """Get summary of all active positions"""
        if not self.positions:
            return "📊 No active positions"
        
        summary = f"📊 Active Positions ({len(self.positions)}):\n"
        for symbol, pos in self.positions.items():
            try:
                entry_time = datetime.fromisoformat(pos.entry_timestamp)
                duration = (datetime.now() - entry_time).total_seconds() / 3600
                summary += f"- {symbol} {pos.signal_type} @ ${pos.entry_price:,.2f} | {duration:.1f}h ago\n"
            except:
                summary += f"- {symbol} {pos.signal_type} @ ${pos.entry_price:,.2f}\n"
        
        return summary
