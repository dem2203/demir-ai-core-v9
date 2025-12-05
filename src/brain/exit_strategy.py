import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("EXIT_STRATEGY")

class ExitStrategy:
    """
    EXIT STRATEGY AI - Intelligent Position Management
    
    Optimizes exits through:
    1. Trailing Stop Logic (move SL as price moves toward TP)
    2. Pattern-Based Early Exit (reversal pattern detection)
    3. Time-Based Exit (close stale positions >48h)
    """
    
    def __init__(self):
        logger.info("🎯 Exit Strategy AI initialized")
    
    def check_exit_signal(self, position: Dict, current_price: float, 
                          detected_pattern: str = None) -> Optional[Dict]:
        """
        Check if position should be exited.
        
        Args:
            position: Current position data (entry, SL, TP, side, entry_time)
            current_price: Current market price
            detected_pattern: Latest detected pattern (if any)
        
        Returns:
            Exit signal dict if exit recommended, None otherwise
        """
        if not position:
            return None
        
        entry_price = float(position['entry_price'])
        sl_price = float(position['sl_price'])
        tp_price = float(position['tp_price'])
        side = position['side']
        entry_time_str = position.get('entry_time')
        
        # Parse entry time
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
        except:
            entry_time = datetime.now() - timedelta(hours=1)  # Fallback
        
        # 1. CHECK TRAILING STOP CONDITIONS
        trailing_exit = self._check_trailing_stop(
            entry_price, sl_price, tp_price, current_price, side
        )
        if trailing_exit:
            return trailing_exit
        
        # 2. CHECK PATTERN-BASED EXIT
        pattern_exit = self._check_pattern_exit(side, detected_pattern)
        if pattern_exit:
            return pattern_exit
        
        # 3. CHECK TIME-BASED EXIT
        time_exit = self._check_time_exit(entry_time, entry_price, current_price, side)
        if time_exit:
            return time_exit
        
        return None
    
    def _check_trailing_stop(self, entry_price: float, sl_price: float, 
                            tp_price: float, current_price: float, side: str) -> Optional[Dict]:
        """
        Implement trailing stop logic.
        
        Rules:
        - If price moves 50% toward TP → move SL to breakeven
        - If price reaches 75% of TP → move SL to lock 50% profit
        """
        if side == "BUY":
            # Calculate progress toward TP
            price_move = current_price - entry_price
            tp_distance = tp_price - entry_price
            
            if tp_distance <= 0:
                return None
            
            progress = (price_move / tp_distance) * 100
            
            # 50% progress → Breakeven SL
            if progress >= 50 and sl_price < entry_price:
                new_sl = entry_price
                logger.info(f"📈 TRAILING STOP TRIGGERED (50% progress) | Moving SL to breakeven: ${new_sl:.2f}")
                return {
                    "action": "UPDATE_SL",
                    "new_sl": new_sl,
                    "reason": "Trailing Stop: 50% to TP (Breakeven)"
                }
            
            # 75% progress → Lock 50% profit
            if progress >= 75:
                profit_distance = tp_price - entry_price
                new_sl = entry_price + (profit_distance * 0.5)
                
                if new_sl > sl_price:  # Only move SL up
                    logger.info(f"📈 TRAILING STOP TRIGGERED (75% progress) | Locking 50% profit: ${new_sl:.2f}")
                    return {
                        "action": "UPDATE_SL",
                        "new_sl": new_sl,
                        "reason": "Trailing Stop: 75% to TP (50% Profit Lock)"
                    }
        
        elif side == "SELL":
            # Calculate progress toward TP (for SHORT)
            price_move = entry_price - current_price
            tp_distance = entry_price - tp_price
            
            if tp_distance <= 0:
                return None
            
            progress = (price_move / tp_distance) * 100
            
            # 50% progress → Breakeven SL
            if progress >= 50 and sl_price > entry_price:
                new_sl = entry_price
                logger.info(f"📉 TRAILING STOP TRIGGERED (50% progress) | Moving SL to breakeven: ${new_sl:.2f}")
                return {
                    "action": "UPDATE_SL",
                    "new_sl": new_sl,
                    "reason": "Trailing Stop: 50% to TP (Breakeven)"
                }
            
            # 75% progress → Lock 50% profit
            if progress >= 75:
                profit_distance = entry_price - tp_price
                new_sl = entry_price - (profit_distance * 0.5)
                
                if new_sl < sl_price:  # Only move SL down (for SHORT)
                    logger.info(f"📉 TRAILING STOP TRIGGERED (75% progress) | Locking 50% profit: ${new_sl:.2f}")
                    return {
                        "action": "UPDATE_SL",
                        "new_sl": new_sl,
                        "reason": "Trailing Stop: 75% to TP (50% Profit Lock)"
                    }
        
        return None
    
    def _check_pattern_exit(self, side: str, detected_pattern: str) -> Optional[Dict]:
        """
        Check for reversal patterns that suggest early exit.
        
        Reversal patterns for LONG: Bearish Engulfing, Evening Star, Shooting Star
        Reversal patterns for SHORT: Bullish Engulfing, Morning Star, Hammer
        """
        if not detected_pattern or detected_pattern == "None":
            return None
        
        bearish_patterns = ["Bearish Engulfing", "Evening Star", "Shooting Star", "Dark Cloud Cover"]
        bullish_patterns = ["Bullish Engulfing", "Morning Star", "Hammer", "Piercing Pattern"]
        
        if side == "BUY" and detected_pattern in bearish_patterns:
            logger.warning(f"⚠️ REVERSAL PATTERN DETECTED: {detected_pattern} | Exiting LONG early")
            return {
                "action": "CLOSE_POSITION",
                "reason": f"Pattern Exit: {detected_pattern} (Bearish Reversal)"
            }
        
        elif side == "SELL" and detected_pattern in bullish_patterns:
            logger.warning(f"⚠️ REVERSAL PATTERN DETECTED: {detected_pattern} | Exiting SHORT early")
            return {
                "action": "CLOSE_POSITION",
                "reason": f"Pattern Exit: {detected_pattern} (Bullish Reversal)"
            }
        
        return None
    
    def _check_time_exit(self, entry_time: datetime, entry_price: float, 
                        current_price: float, side: str) -> Optional[Dict]:
        """
        Exit stale positions that have been open for > 48 hours.
        
        Only exit if:
        - Position is profitable OR
        - Position has been open > 72 hours (force close)
        """
        time_in_position = (datetime.now() - entry_time).total_seconds() / 3600  # hours
        
        if time_in_position < 48:
            return None
        
        # Calculate current PnL
        if side == "BUY":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Exit if profitable after 48h
        if time_in_position >= 48 and pnl_pct > 0:
            logger.info(f"⏰ TIME EXIT: Position open {time_in_position:.1f}h | Profitable ({pnl_pct:.2f}%) → Closing")
            return {
                "action": "CLOSE_POSITION",
                "reason": f"Time Exit: {time_in_position:.1f}h (Profitable)"
            }
        
        # Force exit after 72h regardless of PnL
        if time_in_position >= 72:
            logger.warning(f"⏰ FORCE TIME EXIT: Position open {time_in_position:.1f}h | PnL: {pnl_pct:.2f}% → Force closing")
            return {
                "action": "CLOSE_POSITION",
                "reason": f"Force Time Exit: {time_in_position:.1f}h (Stale Position)"
            }
        
        return None
