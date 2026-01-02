# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - PROFESSIONAL RISK ENGINE
========================================
Institutional-grade risk management layer.

FEATURES:
1. Kelly Criterion Position Sizing
2. Drawdown Protection (Daily/Weekly limits)
3. Correlation Guard (Prevent stacked bets)
4. Dynamic Leverage (Volatility-based)
5. Portfolio Heat Tracking

This module sits between signal generation and execution.
"""
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger("RISK_ENGINE")


@dataclass
class RiskProfile:
    """Output from risk engine - how to size and manage a trade"""
    approved: bool                    # Is trade allowed?
    rejection_reason: str = ""        # Why rejected if not approved
    position_size_pct: float = 0.0    # % of portfolio to risk
    position_size_usd: float = 0.0    # Absolute USD amount
    leverage: int = 1                 # Recommended leverage
    max_loss_usd: float = 0.0         # Maximum acceptable loss
    risk_score: int = 0               # 0-100 (higher = riskier)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    """Current portfolio status"""
    total_equity_usd: float
    available_margin_usd: float
    open_positions: Dict[str, dict]   # symbol -> position info
    daily_pnl_usd: float
    daily_pnl_pct: float
    weekly_pnl_usd: float
    weekly_pnl_pct: float
    current_drawdown_pct: float
    peak_equity_usd: float


class RiskEngine:
    """
    Professional Risk Management Engine
    
    Implements institutional-grade risk controls:
    - Kelly Criterion for optimal position sizing
    - Drawdown limits to protect capital
    - Correlation checks to avoid concentration
    - Volatility-adjusted leverage
    """
    
    # === RISK LIMITS (Configurable) ===
    MAX_DAILY_LOSS_PCT = 5.0          # Stop trading if down 5% today
    MAX_WEEKLY_LOSS_PCT = 15.0        # Stop trading if down 15% this week
    MAX_DRAWDOWN_PCT = 20.0           # Emergency stop at 20% drawdown
    MAX_SINGLE_POSITION_PCT = 10.0    # No single position > 10% of portfolio
    MAX_CORRELATED_EXPOSURE_PCT = 20.0 # Max 20% in correlated assets
    MIN_CONFIDENCE_TO_TRADE = 60      # Don't trade below 60% confidence
    
    # === KELLY CRITERION SETTINGS ===
    KELLY_FRACTION = 0.25             # Use 25% of Kelly (conservative)
    MIN_WIN_RATE_REQUIRED = 0.45      # Need at least 45% win rate
    
    # === CORRELATION MATRIX (Simplified) ===
    CORRELATION_GROUPS = {
        'btc_correlated': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
        'altcoins': ['ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT'],
        'defi': ['UNIUSDT', 'AAVEUSDT', 'LINKUSDT', 'MKRUSDT']
    }
    
    def __init__(self, portfolio_equity_usd: float = 10000):
        self.portfolio_equity = portfolio_equity_usd
        self.peak_equity = portfolio_equity_usd
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.open_positions: Dict[str, dict] = {}
        self.trade_history: List[dict] = []
        self.is_trading_enabled = True
        self.disable_reason = ""
        
        # Performance tracking for Kelly
        self.win_count = 0
        self.loss_count = 0
        self.total_wins_usd = 0.0
        self.total_losses_usd = 0.0
        
        # Data persistence
        self.data_dir = Path("data/risk_engine")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_state()
        logger.info(f"🛡️ Risk Engine initialized | Portfolio: ${portfolio_equity_usd:,.0f}")
    
    # =========================================================
    # MAIN API
    # =========================================================
    
    def evaluate_trade(
        self,
        symbol: str,
        direction: str,  # BUY, SELL
        confidence: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_volatility: float = 0.02  # ATR as % of price
    ) -> RiskProfile:
        """
        Main entry point - evaluate if a trade should be taken and how to size it.
        
        Returns RiskProfile with:
        - approved: True/False
        - position_size_pct: Kelly-optimized size
        - leverage: Volatility-adjusted
        - warnings: Any concerns
        """
        profile = RiskProfile(approved=True)
        
        # === CHECK 1: Is trading enabled? ===
        if not self.is_trading_enabled:
            profile.approved = False
            profile.rejection_reason = f"Trading disabled: {self.disable_reason}"
            return profile
        
        # === CHECK 2: Confidence threshold ===
        if confidence < self.MIN_CONFIDENCE_TO_TRADE:
            profile.approved = False
            profile.rejection_reason = f"Confidence {confidence}% < minimum {self.MIN_CONFIDENCE_TO_TRADE}%"
            return profile
        
        # === CHECK 3: Drawdown limits ===
        drawdown_check = self._check_drawdown_limits()
        if not drawdown_check['ok']:
            profile.approved = False
            profile.rejection_reason = drawdown_check['reason']
            self._disable_trading(drawdown_check['reason'])
            return profile
        
        # === CHECK 4: Correlation / Concentration ===
        correlation_check = self._check_correlation(symbol, direction)
        if not correlation_check['ok']:
            profile.approved = False
            profile.rejection_reason = correlation_check['reason']
            return profile
        
        # === CALCULATE: Position Size (Kelly Criterion) ===
        kelly_size = self._calculate_kelly_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence
        )
        
        # === CALCULATE: Leverage (Volatility-based) ===
        recommended_leverage = self._calculate_dynamic_leverage(current_volatility)
        
        # === CALCULATE: Risk metrics ===
        risk_per_trade = abs(entry_price - stop_loss) / entry_price
        position_usd = self.portfolio_equity * kelly_size
        max_loss = position_usd * risk_per_trade
        
        # === Apply limits ===
        if kelly_size > self.MAX_SINGLE_POSITION_PCT / 100:
            kelly_size = self.MAX_SINGLE_POSITION_PCT / 100
            profile.warnings.append(f"Position capped at {self.MAX_SINGLE_POSITION_PCT}% limit")
        
        # === Calculate risk score ===
        risk_score = self._calculate_risk_score(
            position_size_pct=kelly_size * 100,
            leverage=recommended_leverage,
            volatility=current_volatility,
            confidence=confidence
        )
        
        # === Build final profile ===
        profile.position_size_pct = kelly_size * 100
        profile.position_size_usd = self.portfolio_equity * kelly_size
        profile.leverage = recommended_leverage
        profile.max_loss_usd = max_loss
        profile.risk_score = risk_score
        
        # Add warnings for elevated risk
        if risk_score > 70:
            profile.warnings.append("⚠️ HIGH RISK: Consider reducing size")
        if self.daily_pnl < 0:
            profile.warnings.append(f"📉 Already down ${abs(self.daily_pnl):.0f} today")
        
        logger.info(
            f"🛡️ Risk Evaluation: {symbol} {direction} | "
            f"Size: {profile.position_size_pct:.1f}% (${profile.position_size_usd:,.0f}) | "
            f"Leverage: {profile.leverage}x | Risk Score: {risk_score}"
        )
        
        return profile
    
    def record_trade_result(
        self,
        symbol: str,
        direction: str,
        pnl_usd: float,
        pnl_pct: float
    ):
        """Record a completed trade for performance tracking"""
        is_win = pnl_usd > 0
        
        if is_win:
            self.win_count += 1
            self.total_wins_usd += pnl_usd
        else:
            self.loss_count += 1
            self.total_losses_usd += abs(pnl_usd)
        
        self.daily_pnl += pnl_usd
        self.weekly_pnl += pnl_usd
        self.portfolio_equity += pnl_usd
        
        # Update peak equity for drawdown tracking
        if self.portfolio_equity > self.peak_equity:
            self.peak_equity = self.portfolio_equity
        
        # Log and persist
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'direction': direction,
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'is_win': is_win
        })
        
        self._save_state()
        
        # Check if we need to disable trading
        self._check_drawdown_limits()
        
        logger.info(
            f"📊 Trade recorded: {symbol} | P&L: ${pnl_usd:+,.0f} ({pnl_pct:+.2f}%) | "
            f"Win Rate: {self.win_rate:.1f}%"
        )
    
    # =========================================================
    # KELLY CRITERION
    # =========================================================
    
    def _calculate_kelly_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning (estimated from confidence + historical)
            q = probability of losing (1 - p)
            b = win/loss ratio (reward/risk)
        
        We use fractional Kelly (25%) for safety.
        """
        # Calculate win/loss ratio from the trade setup
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.01  # Minimum position
        
        b = reward / risk  # Win/loss ratio
        
        # Estimate win probability from confidence + historical performance
        historical_win_rate = self.win_rate / 100 if self.total_trades > 10 else 0.5
        estimated_win_rate = (confidence / 100 * 0.6) + (historical_win_rate * 0.4)
        
        p = min(0.75, max(0.35, estimated_win_rate))  # Clamp between 35-75%
        q = 1 - p
        
        # Kelly formula
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply fractional Kelly and minimum
        kelly = max(0.01, kelly * self.KELLY_FRACTION)
        
        # Cap at maximum single position
        kelly = min(kelly, self.MAX_SINGLE_POSITION_PCT / 100)
        
        logger.debug(
            f"Kelly Calc: p={p:.2f}, b={b:.2f}, raw_kelly={kelly/self.KELLY_FRACTION:.3f}, "
            f"fractional={kelly:.3f}"
        )
        
        return kelly
    
    # =========================================================
    # DRAWDOWN PROTECTION
    # =========================================================
    
    def _check_drawdown_limits(self) -> dict:
        """Check if any drawdown limits are breached"""
        
        # Calculate current drawdown from peak
        current_drawdown_pct = 0
        if self.peak_equity > 0:
            current_drawdown_pct = (self.peak_equity - self.portfolio_equity) / self.peak_equity * 100
        
        daily_loss_pct = abs(self.daily_pnl / self.portfolio_equity * 100) if self.daily_pnl < 0 else 0
        weekly_loss_pct = abs(self.weekly_pnl / self.portfolio_equity * 100) if self.weekly_pnl < 0 else 0
        
        # Check limits
        if current_drawdown_pct >= self.MAX_DRAWDOWN_PCT:
            return {
                'ok': False,
                'reason': f"🚨 MAX DRAWDOWN: {current_drawdown_pct:.1f}% (limit: {self.MAX_DRAWDOWN_PCT}%)"
            }
        
        if daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            return {
                'ok': False,
                'reason': f"🛑 DAILY LIMIT: Lost {daily_loss_pct:.1f}% today (limit: {self.MAX_DAILY_LOSS_PCT}%)"
            }
        
        if weekly_loss_pct >= self.MAX_WEEKLY_LOSS_PCT:
            return {
                'ok': False,
                'reason': f"🛑 WEEKLY LIMIT: Lost {weekly_loss_pct:.1f}% this week (limit: {self.MAX_WEEKLY_LOSS_PCT}%)"
            }
        
        return {'ok': True, 'reason': ''}
    
    # =========================================================
    # CORRELATION GUARD
    # =========================================================
    
    def _check_correlation(self, symbol: str, direction: str) -> dict:
        """Check if adding this position creates too much correlated exposure"""
        
        # Find which correlation group this symbol belongs to
        symbol_group = None
        for group_name, symbols in self.CORRELATION_GROUPS.items():
            if symbol in symbols:
                symbol_group = group_name
                break
        
        if not symbol_group:
            return {'ok': True, 'reason': ''}  # Unknown symbol, allow
        
        # Calculate existing exposure in this group
        group_exposure_usd = 0
        for pos_symbol, pos_info in self.open_positions.items():
            if pos_symbol in self.CORRELATION_GROUPS.get(symbol_group, []):
                group_exposure_usd += abs(pos_info.get('size_usd', 0))
        
        group_exposure_pct = group_exposure_usd / self.portfolio_equity * 100
        
        if group_exposure_pct >= self.MAX_CORRELATED_EXPOSURE_PCT:
            return {
                'ok': False,
                'reason': f"⚠️ CORRELATION: Already {group_exposure_pct:.1f}% in {symbol_group} (limit: {self.MAX_CORRELATED_EXPOSURE_PCT}%)"
            }
        
        return {'ok': True, 'reason': ''}
    
    # =========================================================
    # DYNAMIC LEVERAGE
    # =========================================================
    
    def _calculate_dynamic_leverage(self, volatility: float) -> int:
        """
        Calculate recommended leverage based on current volatility.
        
        Higher volatility = lower leverage (risk reduction)
        """
        # Volatility thresholds (as fraction of price)
        if volatility > 0.05:      # >5% daily volatility - VERY HIGH
            return 1
        elif volatility > 0.03:    # 3-5% - HIGH
            return 2
        elif volatility > 0.02:    # 2-3% - MEDIUM
            return 3
        elif volatility > 0.01:    # 1-2% - LOW
            return 5
        else:                      # <1% - VERY LOW
            return 10
    
    # =========================================================
    # RISK SCORE
    # =========================================================
    
    def _calculate_risk_score(
        self,
        position_size_pct: float,
        leverage: int,
        volatility: float,
        confidence: float
    ) -> int:
        """Calculate overall risk score 0-100"""
        
        score = 0
        
        # Position size contribution (0-30 points)
        score += min(30, position_size_pct * 3)
        
        # Leverage contribution (0-30 points)
        score += min(30, leverage * 3)
        
        # Volatility contribution (0-20 points)
        score += min(20, volatility * 400)
        
        # Confidence inverse (0-20 points)
        # Lower confidence = higher risk
        score += max(0, (100 - confidence) * 0.2)
        
        return min(100, int(score))
    
    # =========================================================
    # STATE MANAGEMENT
    # =========================================================
    
    def _disable_trading(self, reason: str):
        """Disable trading due to risk limit breach"""
        self.is_trading_enabled = False
        self.disable_reason = reason
        logger.warning(f"🚨 TRADING DISABLED: {reason}")
        self._save_state()
    
    def enable_trading(self):
        """Re-enable trading (manual override)"""
        self.is_trading_enabled = True
        self.disable_reason = ""
        logger.info("✅ Trading re-enabled")
        self._save_state()
    
    def reset_daily_pnl(self):
        """Call at start of new trading day"""
        self.daily_pnl = 0.0
        self._save_state()
    
    def reset_weekly_pnl(self):
        """Call at start of new trading week"""
        self.weekly_pnl = 0.0
        self._save_state()
    
    @property
    def win_rate(self) -> float:
        """Calculate current win rate"""
        total = self.win_count + self.loss_count
        if total == 0:
            return 50.0  # Default assumption
        return self.win_count / total * 100
    
    @property
    def total_trades(self) -> int:
        return self.win_count + self.loss_count
    
    @property
    def avg_win(self) -> float:
        if self.win_count == 0:
            return 0
        return self.total_wins_usd / self.win_count
    
    @property
    def avg_loss(self) -> float:
        if self.loss_count == 0:
            return 0
        return self.total_losses_usd / self.loss_count
    
    def _save_state(self):
        """Persist state to disk"""
        state = {
            'portfolio_equity': self.portfolio_equity,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'total_wins_usd': self.total_wins_usd,
            'total_losses_usd': self.total_losses_usd,
            'is_trading_enabled': self.is_trading_enabled,
            'disable_reason': self.disable_reason,
            'open_positions': self.open_positions,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.data_dir / 'state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from disk"""
        state_file = self.data_dir / 'state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.portfolio_equity = state.get('portfolio_equity', self.portfolio_equity)
                self.peak_equity = state.get('peak_equity', self.peak_equity)
                self.daily_pnl = state.get('daily_pnl', 0)
                self.weekly_pnl = state.get('weekly_pnl', 0)
                self.win_count = state.get('win_count', 0)
                self.loss_count = state.get('loss_count', 0)
                self.total_wins_usd = state.get('total_wins_usd', 0)
                self.total_losses_usd = state.get('total_losses_usd', 0)
                self.is_trading_enabled = state.get('is_trading_enabled', True)
                self.disable_reason = state.get('disable_reason', '')
                self.open_positions = state.get('open_positions', {})
                logger.info(f"📂 Loaded risk state: Equity=${self.portfolio_equity:,.0f}, Win Rate={self.win_rate:.1f}%")
            except Exception as e:
                logger.warning(f"Could not load risk state: {e}")
    
    def get_status_report(self) -> str:
        """Get human-readable status report"""
        drawdown_pct = (self.peak_equity - self.portfolio_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        
        return f"""
🛡️ RISK ENGINE STATUS
━━━━━━━━━━━━━━━━━━━━━━
💰 Portfolio: ${self.portfolio_equity:,.0f}
📈 Peak: ${self.peak_equity:,.0f}
📉 Drawdown: {drawdown_pct:.1f}%

📊 Performance:
  • Win Rate: {self.win_rate:.1f}% ({self.win_count}W / {self.loss_count}L)
  • Avg Win: ${self.avg_win:,.0f}
  • Avg Loss: ${self.avg_loss:,.0f}

📅 P&L:
  • Daily: ${self.daily_pnl:+,.0f}
  • Weekly: ${self.weekly_pnl:+,.0f}

🚦 Status: {'✅ TRADING ENABLED' if self.is_trading_enabled else '🛑 TRADING DISABLED'}
{f'   Reason: {self.disable_reason}' if not self.is_trading_enabled else ''}
━━━━━━━━━━━━━━━━━━━━━━
"""


# Singleton instance
_risk_engine: Optional[RiskEngine] = None

def get_risk_engine(portfolio_usd: float = 10000) -> RiskEngine:
    """Get or create the risk engine instance"""
    global _risk_engine
    if _risk_engine is None:
        _risk_engine = RiskEngine(portfolio_usd)
    return _risk_engine
