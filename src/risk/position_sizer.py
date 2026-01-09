import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger("KELLY_SIZER")

class KellyPositionSizer:
    """
    Professional position sizing using Kelly Criterion
    Maximizes long-term growth while managing risk
    """
    
    def __init__(self, max_risk_per_trade=0.02, use_half_kelly=True):
        self.max_risk = max_risk_per_trade  # 2% max per trade
        self.half_kelly = use_half_kelly  # Conservative Kelly
        
    def calculate_position_size(self, 
                                 account_balance: float,
                                 win_rate: float,
                                 avg_win_pct: float,
                                 avg_loss_pct: float,
                                 current_confidence: int) -> Dict:
        """
        Kelly Formula: f* = (bp - q) / b
        
        Args:
            account_balance: Total capital
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average winning trade % (e.g., 0.03 for 3%)
            avg_loss_pct: Average losing trade % (e.g., 0.015 for 1.5%)
            current_confidence: AI confidence 1-10
        
        Returns:
            Dict with position size and details
        """
        
        # Adjust win rate by AI confidence
        adjusted_win_rate = win_rate * (current_confidence / 10)
        adjusted_win_rate = max(0.01, min(0.99, adjusted_win_rate))  # Clamp
        
        # Calculate odds (b)
        b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1.5
        
        # Probabilities
        p = adjusted_win_rate
        q = 1 - p
        
        # Kelly percentage
        kelly_pct = (b * p - q) / b
        
        # Apply safety multiplier
        if self.half_kelly:
            kelly_pct *= 0.5
        
        # Never go negative or exceed max risk
        kelly_pct = max(0, min(kelly_pct, self.max_risk))
        
        # Calculate position value
        position_value = account_balance * kelly_pct
        
        # Calculate quantity (will be divided by price later)
        logger.info(f"💰 Kelly: {kelly_pct*100:.1f}% of capital (${position_value:.2f})")
        logger.info(f"   Win Rate: {win_rate*100:.0f}% → Adjusted: {adjusted_win_rate*100:.0f}% (AI conf: {current_confidence}/10)")
        logger.info(f"   Odds: {b:.2f}:1 | Kelly fraction: {kelly_pct*100:.1f}%")
        
        return {
            'kelly_pct': kelly_pct,
            'position_value': position_value,
            'adjusted_win_rate': adjusted_win_rate,
            'odds': b,
            'half_kelly': self.half_kelly
        }
    
    def get_default_sizing(self, account_balance: float, confidence: int) -> float:
        """
        Fallback sizing when no historical data
        Conservative fixed percentage based on confidence
        """
        base_pct = 0.01  # 1% base
        confidence_multiplier = confidence / 10
        
        position_pct = base_pct * confidence_multiplier
        position_pct = min(position_pct, 0.02)  # Cap at 2%
        
        return account_balance * position_pct


class ATRStopCalculator:
    """
    Dynamic stop-loss based on volatility (ATR)
    Industry standard for crypto
    """
    
    def calculate_atr(self, df: pd.DataFrame, period=14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def calculate_stops(self, entry_price: float, atr: float, 
                       direction: str, multiplier=2.0) -> Dict:
        """
        Calculate stop-loss and take-profit levels
        
        Standard: 2 ATR for SL, 3 ATR for TP (1.5 R:R)
        """
        if direction == "LONG":
            stop_loss = entry_price - (multiplier * atr)
            take_profit = entry_price + (multiplier * 1.5 * atr)
        else:  # SHORT
            stop_loss = entry_price + (multiplier * atr)
            take_profit = entry_price - (multiplier * 1.5 * atr)
        
        # Calculate risk amount
        risk_per_unit = abs(entry_price - stop_loss)
        reward_per_unit = abs(take_profit - entry_price)
        risk_reward = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0
        
        logger.info(f"🎯 ATR Stops: SL=${stop_loss:.2f} | TP=${take_profit:.2f}")
        logger.info(f"   R:R Ratio: {risk_reward:.2f}:1 | ATR: ${atr:.2f}")
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'risk_reward_ratio': round(risk_reward, 2),
            'atr_value': round(atr, 2),
            'risk_per_unit': round(risk_per_unit, 2)
        }
