import logging
import pandas as pd
from src.brain.indicators import Indicators
from src.brain.cortex import StrategicDirective

logger = logging.getLogger("TACTICAL_ANALYST")

class TacticalAnalyst:
    """
    The Analyst.
    Executes the 'How' based on the Cortex's 'What'.
    """
    def __init__(self):
        pass
        
    def analyze_chart(self, symbol: str, df: pd.DataFrame, directive: StrategicDirective) -> dict:
        """
        Returns a concrete signal: BUY, SELL, or HOLD.
        """
        if df.empty:
            return {"action": "HOLD", "reason": "No Data"}
            
        # 0. Check Directive
        if directive.allowed_direction == "NONE":
            return {"action": "HOLD", "reason": f"Cortex Forbidden: {directive.reasoning}"}
            
        # 1. Calculate Indicators
        st = Indicators.supertrend(df)
        upper, lower, bandwidth = Indicators.bollinger_bands(df)
        
        last_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        st_val = st['supertrend'].iloc[-1]
        st_trend = st['trend'].iloc[-1]
        
        # 2. Strategy Logic: Trend Follow + Volatility
        signal = "HOLD"
        reason = "Waiting for setup"
        
        # --- LONG Logic ---
        if directive.allowed_direction in ["LONG", "BOTH"]:
            # SuperTrend Uptrend AND Price just broke out or verified support
            if st_trend == 1:
                # Condition A: Trend Flip (Fresh Start)
                if st['trend'].iloc[-2] == -1:
                   signal = "BUY" 
                   reason = "SuperTrend Flip BULLISH"
                # Condition B: Bollinger Breakout
                elif last_close > upper.iloc[-1] and bandwidth.iloc[-1] < 0.10: # Squeeze
                    signal = "BUY"
                    reason = "Bollinger Breakout (Squeeze)"
                    
        # --- SHORT Logic ---
        if directive.allowed_direction in ["SHORT", "BOTH"]:
            if st_trend == -1:
                if st['trend'].iloc[-2] == 1:
                    signal = "SELL"
                    reason = "SuperTrend Flip BEARISH"
                elif last_close < lower.iloc[-1] and bandwidth.iloc[-1] < 0.10:
                    signal = "SELL"
                    reason = "Bollinger Breakdown (Squeeze)"
        
        return {
            "action": signal,
            "entry_price": last_close,
            "stop_loss": st_val, # Use SuperTrend as initial stop
            "reason": reason,
            "cortex_note": directive.reasoning
        }
