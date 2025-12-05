import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger("RISK_SHIELD")

class RiskShield:
    """
    RISK SHIELD - Adaptive Drawdown Protection
    
    Protects capital during losing streaks by:
    - Tracking consecutive losses
    - Reducing Kelly % after 2 losses (50% reduction)
    - Activating cooling mode after 3 losses (24h pause)
    - Gradually recovering after wins
    """
    
    def __init__(self):
        self.state_file = "risk_shield_state.json"
        self.consecutive_losses = 0
        self.recovery_mode = False
        self.cooling_until = None
        self.total_trades = 0
        self.total_wins = 0
        
        self._load_state()
        logger.info(f"🛡️ Risk Shield initialized | Losses: {self.consecutive_losses} | Cooling: {self.is_cooling()}")
    
    def _load_state(self):
        """Load shield state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.consecutive_losses = state.get('consecutive_losses', 0)
                    self.recovery_mode = state.get('recovery_mode', False)
                    self.total_trades = state.get('total_trades', 0)
                    self.total_wins = state.get('total_wins', 0)
                    
                    cooling_str = state.get('cooling_until')
                    if cooling_str:
                        self.cooling_until = datetime.fromisoformat(cooling_str)
            except Exception as e:
                logger.error(f"Error loading shield state: {e}")
    
    def _save_state(self):
        """Save shield state to file"""
        try:
            state = {
                'consecutive_losses': self.consecutive_losses,
                'recovery_mode': self.recovery_mode,
                'cooling_until': self.cooling_until.isoformat() if self.cooling_until else None,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving shield state: {e}")
    
    def is_cooling(self) -> bool:
        """Check if in cooling mode"""
        if self.cooling_until is None:
            return False
        
        if datetime.now() < self.cooling_until:
            return True
        else:
            # Cooling period ended
            self.cooling_until = None
            self.consecutive_losses = 0
            self._save_state()
            logger.info("❄️ Cooling period ended. Shield reset.")
            return False
    
    def adjust_kelly(self, base_kelly: float) -> float:
        """
        Adjust Kelly % based on current shield state.
        
        Returns:
            Adjusted Kelly percentage (0 = no trade, 0.5 = half risk, 1.0 = full risk)
        """
        # Check cooling mode first
        if self.is_cooling():
            remaining = (self.cooling_until - datetime.now()).total_seconds() / 3600
            logger.warning(f"❄️ COOLING MODE ACTIVE: {remaining:.1f}h remaining | NO TRADES")
            return 0.0
        
        # Normal risk adjustment
        if self.consecutive_losses >= 2:
            logger.warning(f"⚠️ RISK REDUCED: {self.consecutive_losses} consecutive losses | Kelly → 50%")
            return base_kelly * 0.5
        elif self.recovery_mode and self.consecutive_losses == 0:
            logger.info(f"🔄 RECOVERY MODE: Gradual return | Kelly → 75%")
            return base_kelly * 0.75
        else:
            return base_kelly
    
    def record_trade_result(self, result: str, pnl_pct: float = 0):
        """
        Record trade outcome and update shield state.
        
        Args:
            result: "WIN" or "LOSS"
            pnl_pct: Profit/Loss percentage (for tracking)
        """
        self.total_trades += 1
        
        if result == "WIN":
            self.total_wins += 1
            self.consecutive_losses = 0
            self.recovery_mode = True
            logger.info(f"✅ WIN RECORDED | PnL: {pnl_pct:.2f}% | Shield reset to RECOVERY mode")
        
        elif result == "LOSS":
            self.consecutive_losses += 1
            self.recovery_mode = False
            
            logger.warning(f"❌ LOSS RECORDED | PnL: {pnl_pct:.2f}% | Consecutive: {self.consecutive_losses}")
            
            # Activate cooling mode after 3 losses
            if self.consecutive_losses >= 3:
                self.cooling_until = datetime.now() + timedelta(hours=24)
                logger.critical(f"🚨 COOLING MODE ACTIVATED: 3 losses detected | Paused until {self.cooling_until.strftime('%Y-%m-%d %H:%M')}")
        
        self._save_state()
    
    def get_status(self) -> Dict:
        """Get current shield status for dashboard"""
        win_rate = (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "consecutive_losses": self.consecutive_losses,
            "is_cooling": self.is_cooling(),
            "cooling_until": self.cooling_until.isoformat() if self.cooling_until else None,
            "recovery_mode": self.recovery_mode,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "win_rate": win_rate,
            "risk_level": "PAUSED" if self.is_cooling() else ("REDUCED" if self.consecutive_losses >= 2 else ("RECOVERY" if self.recovery_mode else "NORMAL"))
        }
    
    def force_reset(self):
        """Manually reset shield (admin use only)"""
        self.consecutive_losses = 0
        self.recovery_mode = False
        self.cooling_until = None
        self._save_state()
        logger.warning("⚡ SHIELD MANUALLY RESET")
        
    def check_news_risk(self) -> bool:
        """
        [Phase 17] Volatile News Shield
        Returns True if high-impact news is imminent (within 30 mins).
        Currently a placeholder structure for future API integration.
        """
        # Logic to check economic calendar would go here
        # Return True to block trades
        return False

