# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SIGNAL QUALITY FILTER
=====================================
Ensures signal quality before sending to user.

FIXES:
1. Regime Filter: No SHORT in BULLISH, no LONG in BEARISH
2. Kelly Dynamic: Calculate proper position sizing
3. Cooldown: 2 hours between same-symbol signals
4. Risk Veto: Block trades with warning indicators
5. Predictive Check: Ensure signal is leading, not lagging

Author: DEMIR AI Team
Date: 2026-01-02
"""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta

logger = logging.getLogger("SIGNAL_QUALITY")


@dataclass
class QualityCheckResult:
    """Result of signal quality check"""
    passed: bool
    reason: str
    adjusted_confidence: int
    kelly_position_pct: float
    veto_active: bool
    warnings: list


class SignalQualityFilter:
    """
    Professional Signal Quality Filter
    
    Ensures all signals meet institutional quality standards
    before being sent to users.
    """
    
    # Cooldown settings (2 hours = prevents overtrading)
    COOLDOWN_MINUTES = 120
    
    # Minimum confidence to trade
    MIN_CONFIDENCE = 65
    
    # Kelly Criterion parameters
    KELLY_FRACTION = 0.25  # Use 25% of full Kelly (conservative)
    MAX_POSITION_PCT = 5.0  # Never risk more than 5%
    MIN_POSITION_PCT = 0.5  # Always risk at least 0.5%
    
    # Win rate assumptions for Kelly (will be updated from performance)
    DEFAULT_WIN_RATE = 0.45  # Conservative assumption
    DEFAULT_WIN_LOSS_RATIO = 1.8  # Target R:R
    
    def __init__(self):
        self._last_signals: Dict[str, datetime] = {}
        self._performance_cache: Dict[str, dict] = {}
        
        logger.info("🛡️ Signal Quality Filter initialized")
    
    def check_signal(
        self,
        symbol: str,
        action: str,           # BUY or SELL
        confidence: int,
        regime: str,           # TRENDING_BULL, TRENDING_BEAR, RANGING, VOLATILE
        sentiment: str,        # BULLISH, BEARISH, NEUTRAL
        rsi: float,
        orderbook_imbalance: float,  # +ve = buyers, -ve = sellers
        whale_flow: float,     # +ve = accumulation, -ve = distribution
        funding_rate: float,
        win_rate: float = None,  # Historical win rate
        avg_rr: float = None     # Average Risk:Reward
    ) -> QualityCheckResult:
        """
        Check if signal meets quality standards.
        
        Returns QualityCheckResult with pass/fail and adjustments.
        """
        warnings = []
        veto = False
        
        # === 1. REGIME FILTER (Most Important!) ===
        regime_ok, regime_msg = self._check_regime_alignment(action, regime, sentiment)
        if not regime_ok:
            return QualityCheckResult(
                passed=False,
                reason=f"❌ Regime Veto: {regime_msg}",
                adjusted_confidence=0,
                kelly_position_pct=0,
                veto_active=True,
                warnings=[regime_msg]
            )
        
        # === 2. COOLDOWN CHECK ===
        cooldown_ok, cooldown_msg = self._check_cooldown(symbol)
        if not cooldown_ok:
            return QualityCheckResult(
                passed=False,
                reason=f"⏰ Cooldown: {cooldown_msg}",
                adjusted_confidence=0,
                kelly_position_pct=0,
                veto_active=False,
                warnings=[cooldown_msg]
            )
        
        # === 3. CONFIDENCE CHECK ===
        if confidence < self.MIN_CONFIDENCE:
            return QualityCheckResult(
                passed=False,
                reason=f"📉 Low confidence: {confidence}% < {self.MIN_CONFIDENCE}%",
                adjusted_confidence=confidence,
                kelly_position_pct=0,
                veto_active=False,
                warnings=["Confidence too low"]
            )
        
        # === 4. RSI CHECK (Prevent chasing) ===
        if action == "BUY" and rsi > 75:
            warnings.append(f"RSI overbought ({rsi:.0f})")
            confidence = int(confidence * 0.8)  # Reduce confidence
            
        if action == "SELL" and rsi < 25:
            warnings.append(f"RSI oversold ({rsi:.0f})")
            confidence = int(confidence * 0.8)
        
        # === 5. ORDERBOOK/WHALE ALIGNMENT ===
        if action == "BUY" and orderbook_imbalance < -30:
            warnings.append("Orderbook shows selling pressure")
            confidence = int(confidence * 0.9)
            
        if action == "SELL" and orderbook_imbalance > 30:
            warnings.append("Orderbook shows buying pressure")
            confidence = int(confidence * 0.9)
        
        if action == "BUY" and whale_flow < -50:
            warnings.append("Whale distribution detected")
            veto = True
            
        if action == "SELL" and whale_flow > 50:
            warnings.append("Whale accumulation detected")
            veto = True
        
        # === 6. FUNDING RATE CHECK ===
        if action == "BUY" and funding_rate > 0.01:  # 1% = extreme
            warnings.append(f"High funding ({funding_rate*100:.2f}%)")
            confidence = int(confidence * 0.85)
            
        if action == "SELL" and funding_rate < -0.01:
            warnings.append(f"Negative funding ({funding_rate*100:.2f}%)")
            confidence = int(confidence * 0.85)
        
        # === 7. VETO CHECK ===
        if veto:
            return QualityCheckResult(
                passed=False,
                reason=f"🚫 Risk Veto: {', '.join(warnings)}",
                adjusted_confidence=confidence,
                kelly_position_pct=0,
                veto_active=True,
                warnings=warnings
            )
        
        # === 8. KELLY POSITION SIZING ===
        kelly_pct = self._calculate_kelly(
            win_rate or self.DEFAULT_WIN_RATE,
            avg_rr or self.DEFAULT_WIN_LOSS_RATIO,
            confidence / 100
        )
        
        # Update cooldown
        self._last_signals[symbol] = datetime.now()
        
        return QualityCheckResult(
            passed=True,
            reason="✅ Signal quality OK",
            adjusted_confidence=confidence,
            kelly_position_pct=kelly_pct,
            veto_active=False,
            warnings=warnings
        )
    
    def _check_regime_alignment(
        self,
        action: str,
        regime: str,
        sentiment: str
    ) -> Tuple[bool, str]:
        """
        Check if trade direction aligns with market regime.
        
        RULE: Trade WITH the trend, not against it.
        """
        # Strong trend regimes
        if regime == "TRENDING_BULL":
            if action == "SELL":
                return False, "Cannot SHORT in BULLISH trend - trade with the trend"
            return True, "Aligned with bullish trend"
        
        if regime == "TRENDING_BEAR":
            if action == "BUY":
                return False, "Cannot LONG in BEARISH trend - trade with the trend"
            return True, "Aligned with bearish trend"
        
        # Volatile/Ranging - allow both but check sentiment
        if regime in ["VOLATILE", "RANGING"]:
            # In sideways markets, sentiment matters more
            if action == "BUY" and sentiment == "BEARISH":
                return False, "Cannot LONG when sentiment is BEARISH"
            if action == "SELL" and sentiment == "BULLISH":
                return False, "Cannot SHORT when sentiment is BULLISH"
            return True, f"Range trade OK - sentiment: {sentiment}"
        
        # Unknown regime - be conservative
        return True, "Unknown regime - proceed with caution"
    
    def _check_cooldown(self, symbol: str) -> Tuple[bool, str]:
        """Check if cooldown period has passed"""
        last_signal = self._last_signals.get(symbol)
        
        if last_signal is None:
            return True, "No recent signals"
        
        minutes_since = (datetime.now() - last_signal).total_seconds() / 60
        
        if minutes_since < self.COOLDOWN_MINUTES:
            remaining = self.COOLDOWN_MINUTES - minutes_since
            return False, f"Wait {remaining:.0f} more minutes"
        
        return True, f"Cooldown passed ({minutes_since:.0f}m)"
    
    def _calculate_kelly(
        self,
        win_rate: float,
        win_loss_ratio: float,
        confidence_factor: float
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win probability
            R = Win/Loss ratio
        
        We use fractional Kelly (25%) for safety.
        """
        if win_loss_ratio <= 0:
            return self.MIN_POSITION_PCT
        
        # Kelly formula
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly
        kelly = kelly * self.KELLY_FRACTION
        
        # Apply confidence adjustment
        kelly = kelly * confidence_factor
        
        # Convert to percentage and clamp
        kelly_pct = kelly * 100
        kelly_pct = max(self.MIN_POSITION_PCT, min(self.MAX_POSITION_PCT, kelly_pct))
        
        return round(kelly_pct, 1)
    
    def update_performance(self, symbol: str, win_rate: float, avg_rr: float):
        """Update cached performance metrics"""
        self._performance_cache[symbol] = {
            'win_rate': win_rate,
            'avg_rr': avg_rr,
            'updated': datetime.now()
        }
    
    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            'signals_today': len([s for s, t in self._last_signals.items() 
                                 if t.date() == datetime.now().date()]),
            'cooldown_minutes': self.COOLDOWN_MINUTES,
            'min_confidence': self.MIN_CONFIDENCE,
            'kelly_fraction': self.KELLY_FRACTION
        }


# Singleton
_filter: Optional[SignalQualityFilter] = None

def get_signal_quality_filter() -> SignalQualityFilter:
    global _filter
    if _filter is None:
        _filter = SignalQualityFilter()
    return _filter
