import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger("HEDGE_MANAGER")

class HedgeManager:
    """
    DEMIR AI V20.0 - POSITION HEDGING MODULE
    
    Protects long positions by opening small short hedges during high volatility.
    Hedges are automatically closed when volatility normalizes.
    """
    
    VIX_THRESHOLD = 30  # If VIX > 30, hedging activates
    HEDGE_RATIO = 0.10  # Hedge 10% of position value
    
    def __init__(self):
        self.active_hedges = {}  # {symbol: hedge_details}
    
    def should_hedge(self, vix: float, current_positions: Dict) -> bool:
        """
        Determines if we should open a hedge.
        """
        if vix > self.VIX_THRESHOLD and len(current_positions) > 0:
            logger.warning(f"⚠️ HIGH VOLATILITY DETECTED: VIX={vix:.2f}")
            return True
        return False
    
    def calculate_hedge_size(self, position_value: float) -> float:
        """
        Calculates hedge position size (10% of main position).
        """
        return position_value * self.HEDGE_RATIO
    
    def open_hedge(self, symbol: str, position_value: float, current_price: float) -> Dict:
        """
        Opens a short hedge against a long position.
        
        Returns hedge details for logging/tracking.
        """
        hedge_value = self.calculate_hedge_size(position_value)
        hedge_amount = hedge_value / current_price
        
        hedge = {
            "symbol": symbol,
            "side": "SHORT",
            "entry_price": current_price,
            "amount": hedge_amount,
            "value": hedge_value,
            "status": "ACTIVE"
        }
        
        self.active_hedges[symbol] = hedge
        
        logger.info(f"🛡️ HEDGE OPENED: {symbol} Short x{hedge_amount:.4f} at ${current_price:.2f}")
        return hedge
    
    def should_close_hedge(self, symbol: str, vix: float) -> bool:
        """
        Determines if a hedge should be closed (VIX normalized).
        """
        if symbol not in self.active_hedges:
            return False
        
        if vix < self.VIX_THRESHOLD * 0.8:  # 20% buffer to avoid flip-flopping
            logger.info(f"✅ VIX NORMALIZED ({vix:.2f}). Closing hedge for {symbol}")
            return True
        
        return False
    
    def close_hedge(self, symbol: str, current_price: float) -> Dict:
        """
        Closes an active hedge and calculates P/L.
        """
        if symbol not in self.active_hedges:
            return {}
        
        hedge = self.active_hedges[symbol]
        
        # Short P/L: Profit if price went down
        pnl = (hedge['entry_price'] - current_price) * hedge['amount']
        
        hedge['exit_price'] = current_price
        hedge['pnl'] = pnl
        hedge['status'] = "CLOSED"
        
        logger.info(f"🛡️ HEDGE CLOSED: {symbol} | PnL: ${pnl:.2f}")
        
        del self.active_hedges[symbol]
        
        return hedge
    
    def get_active_hedges(self) -> Dict:
        """
        Returns all currently active hedges.
        """
        return self.active_hedges
