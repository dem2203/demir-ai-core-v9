"""
Risk-Adjusted Position Sizing Module
Professional implementation with Kelly Criterion + Volatility-based adjustments

Features:
- Full Kelly and Fractional Kelly options
- ATR-based volatility adjustment
- Maximum drawdown protection
- Risk per trade limits
- Dynamic position scaling
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("POSITION_SIZER")


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    size_percent: float  # Percentage of portfolio
    size_usd: float      # Dollar amount
    kelly_raw: float     # Raw Kelly percentage
    kelly_adjusted: float  # Volatility-adjusted Kelly
    risk_amount: float   # Dollar amount at risk
    max_loss_percent: float  # Maximum loss percentage
    confidence_factor: float  # Confidence adjustment factor
    volatility_factor: float  # Volatility adjustment factor
    reason: str


class PositionSizer:
    """
    Professional Position Sizing with Kelly Criterion and Volatility Adjustments.
    
    Kelly Formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to bet
        b = odds received on bet (win/loss ratio)
        p = probability of winning
        q = probability of losing (1 - p)
    """
    
    def __init__(
        self,
        max_position_percent: float = 25.0,  # Maximum 25% of portfolio
        min_position_percent: float = 1.0,   # Minimum 1% of portfolio
        kelly_fraction: float = 0.25,        # Use 1/4 Kelly (conservative)
        max_risk_per_trade: float = 2.0,     # Max 2% portfolio risk per trade
        atr_multiplier: float = 2.0,         # ATR multiplier for stop loss
        volatility_lookback: int = 20,       # Periods for volatility calc
        enable_volatility_scaling: bool = True,
        enable_drawdown_protection: bool = True,
        max_drawdown_limit: float = 10.0     # Reduce size after 10% drawdown
    ):
        self.max_position_percent = max_position_percent
        self.min_position_percent = min_position_percent
        self.kelly_fraction = kelly_fraction
        self.max_risk_per_trade = max_risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.volatility_lookback = volatility_lookback
        self.enable_volatility_scaling = enable_volatility_scaling
        self.enable_drawdown_protection = enable_drawdown_protection
        self.max_drawdown_limit = max_drawdown_limit
        
        # Performance tracking
        self.historical_win_rate = 0.55  # Default 55% win rate
        self.historical_rr_ratio = 1.5   # Default 1.5:1 R:R
        self.current_drawdown = 0.0
        
        logger.info(f"PositionSizer initialized: Kelly={kelly_fraction}, MaxRisk={max_risk_per_trade}%")
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        current_drawdown: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using Kelly Criterion with adjustments.
        
        Args:
            portfolio_value: Total portfolio value in USD
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            take_profit: Take profit price (TP1)
            confidence: AI confidence (0-100)
            atr: Average True Range (optional)
            volatility: Historical volatility (optional)
            win_rate: Override win rate (optional)
            current_drawdown: Current portfolio drawdown (optional)
            
        Returns:
            PositionSizeResult with all sizing details
        """
        if portfolio_value <= 0 or entry_price <= 0:
            return self._empty_result("Invalid portfolio or entry price")
        
        # Calculate risk/reward metrics
        risk_per_unit = abs(entry_price - stop_loss) / entry_price
        reward_per_unit = abs(take_profit - entry_price) / entry_price
        
        if risk_per_unit <= 0:
            return self._empty_result("Invalid stop loss (no risk)")
        
        # Risk to reward ratio
        rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 1.0
        
        # Use provided or historical win rate
        p = (win_rate or self.historical_win_rate) / 100  # Convert to decimal
        q = 1 - p
        b = rr_ratio  # Odds
        
        # Calculate raw Kelly percentage
        kelly_raw = (b * p - q) / b if b > 0 else 0
        kelly_raw = max(0, kelly_raw)  # No negative bets
        
        # Apply Kelly fraction (conservative)
        kelly_fraction = kelly_raw * self.kelly_fraction * 100  # Convert to percentage
        
        # Confidence adjustment (scale based on AI confidence)
        confidence_factor = min(confidence / 80, 1.0)  # 80% confidence = 100% factor
        kelly_adjusted = kelly_fraction * confidence_factor
        
        # Volatility adjustment
        volatility_factor = 1.0
        if self.enable_volatility_scaling and volatility:
            # Lower position size when volatility is high
            # Base volatility: 20% annualized = normal, >30% = high, <10% = low
            base_vol = 0.20
            volatility_factor = base_vol / max(volatility, 0.05)
            volatility_factor = np.clip(volatility_factor, 0.5, 1.5)
            kelly_adjusted *= volatility_factor
        
        # ATR-based stop adjustment
        if atr and atr > 0:
            atr_stop_distance = (atr * self.atr_multiplier) / entry_price
            if atr_stop_distance > risk_per_unit:
                # If ATR suggests wider stop, reduce position size
                atr_factor = risk_per_unit / atr_stop_distance
                kelly_adjusted *= atr_factor
        
        # Drawdown protection
        dd = current_drawdown or self.current_drawdown
        if self.enable_drawdown_protection and dd > 0:
            # Reduce position size during drawdown
            # At max_drawdown_limit, reduce by 50%
            dd_factor = max(0.5, 1 - (dd / self.max_drawdown_limit) * 0.5)
            kelly_adjusted *= dd_factor
        
        # Apply limits
        position_percent = np.clip(
            kelly_adjusted,
            self.min_position_percent,
            self.max_position_percent
        )
        
        # Calculate actual sizes
        position_usd = portfolio_value * (position_percent / 100)
        
        # Verify risk per trade limit
        risk_amount = position_usd * risk_per_unit
        max_risk_amount = portfolio_value * (self.max_risk_per_trade / 100)
        
        if risk_amount > max_risk_amount:
            # Scale down to meet risk limit
            scale_factor = max_risk_amount / risk_amount
            position_percent *= scale_factor
            position_usd = portfolio_value * (position_percent / 100)
            risk_amount = position_usd * risk_per_unit
        
        # Build result
        result = PositionSizeResult(
            size_percent=round(position_percent, 2),
            size_usd=round(position_usd, 2),
            kelly_raw=round(kelly_raw * 100, 2),
            kelly_adjusted=round(kelly_adjusted, 2),
            risk_amount=round(risk_amount, 2),
            max_loss_percent=round(risk_per_unit * 100, 2),
            confidence_factor=round(confidence_factor, 2),
            volatility_factor=round(volatility_factor, 2),
            reason=self._build_reason(kelly_raw, confidence_factor, volatility_factor, dd)
        )
        
        logger.info(f"Position Size: {result.size_percent}% (${result.size_usd:,.0f}) - {result.reason}")
        
        return result
    
    def calculate_kelly_simple(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Simple Kelly calculation from historical performance.
        
        Args:
            win_rate: Historical win rate (0-100)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            
        Returns:
            Kelly percentage (0-100)
        """
        if avg_loss <= 0:
            return 0.0
        
        p = win_rate / 100
        q = 1 - p
        b = avg_win / avg_loss  # Win/loss ratio
        
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0, kelly) * 100
        
        # Apply fraction
        return kelly * self.kelly_fraction
    
    def update_performance(self, win_rate: float, rr_ratio: float):
        """Update historical performance metrics."""
        self.historical_win_rate = win_rate
        self.historical_rr_ratio = rr_ratio
        logger.info(f"Performance updated: WR={win_rate}%, R:R={rr_ratio}")
    
    def update_drawdown(self, drawdown: float):
        """Update current drawdown level."""
        self.current_drawdown = abs(drawdown)
    
    def _empty_result(self, reason: str) -> PositionSizeResult:
        """Return empty result with reason."""
        return PositionSizeResult(
            size_percent=0,
            size_usd=0,
            kelly_raw=0,
            kelly_adjusted=0,
            risk_amount=0,
            max_loss_percent=0,
            confidence_factor=0,
            volatility_factor=0,
            reason=reason
        )
    
    def _build_reason(
        self,
        kelly_raw: float,
        conf_factor: float,
        vol_factor: float,
        drawdown: float
    ) -> str:
        """Build human-readable reason string."""
        parts = [f"Kelly={kelly_raw*100:.1f}%"]
        
        if conf_factor < 1.0:
            parts.append(f"conf_adj={conf_factor:.2f}")
        
        if vol_factor != 1.0:
            parts.append(f"vol_adj={vol_factor:.2f}")
        
        if drawdown > 0:
            parts.append(f"DD_protection={drawdown:.1f}%")
        
        return ", ".join(parts)
    
    def get_status(self) -> Dict:
        """Get current sizer status."""
        return {
            "max_position_percent": self.max_position_percent,
            "kelly_fraction": self.kelly_fraction,
            "max_risk_per_trade": self.max_risk_per_trade,
            "historical_win_rate": self.historical_win_rate,
            "historical_rr_ratio": self.historical_rr_ratio,
            "current_drawdown": self.current_drawdown,
            "volatility_scaling": self.enable_volatility_scaling,
            "drawdown_protection": self.enable_drawdown_protection
        }
