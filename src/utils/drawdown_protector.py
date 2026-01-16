import logging
import os
import json
from datetime import datetime, timedelta
from src.config import Config

logger = logging.getLogger("DRAWDOWN_PROTECTOR")

class DailyDrawdownProtector:
    """
    Protects capital by tracking daily P&L and enforcing loss limits.
    Prevents catastrophic losses while allowing position tracking for learning.
    """
    
    def __init__(self, max_daily_loss_pct: float = 5.0):
        """
        Args:
            max_daily_loss_pct: Maximum daily loss percentage (default: 5%)
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.db_path = os.path.join(Config.DATA_DIR, "daily_pnl.json")
        
        # Load or initialize daily stats
        self.daily_stats = self._load_daily_stats()
        self.today = datetime.now().date().isoformat()
        
        # Initialize today's stats if new day
        if self.today not in self.daily_stats:
            self.daily_stats[self.today] = {
                "start_balance": None,
                "current_balance": None,
                "trades_taken": 0,
                "trades_blocked": 0,
                "total_pnl": 0.0,
                "limit_hit": False
            }
            self._save_daily_stats()
    
    def _load_daily_stats(self) -> dict:
        """Load daily P&L history"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load daily stats: {e}")
        return {}
    
    def _save_daily_stats(self):
        """Save daily P&L history"""
        try:
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.daily_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save daily stats: {e}")
    
    def set_start_balance(self, balance: float):
        """Set starting balance for the day"""
        if self.daily_stats[self.today]["start_balance"] is None:
            self.daily_stats[self.today]["start_balance"] = balance
            self.daily_stats[self.today]["current_balance"] = balance
            self._save_daily_stats()
            logger.info(f"📊 Daily balance initialized: ${balance:.2f}")
    
    def update_balance(self, new_balance: float, pnl: float = 0):
        """Update current balance and P&L"""
        if self.daily_stats[self.today]["start_balance"] is None:
            self.set_start_balance(new_balance)
        
        self.daily_stats[self.today]["current_balance"] = new_balance
        self.daily_stats[self.today]["total_pnl"] += pnl
        self._save_daily_stats()
    
    def check_trade_allowed(self, current_balance: float) -> tuple[bool, str, dict]:
        """
        Check if new trade is allowed based on daily drawdown.
        
        Returns:
            (allowed: bool, reason: str, stats: dict)
        """
        # Initialize balance if first check of the day
        if self.daily_stats[self.today]["start_balance"] is None:
            self.set_start_balance(current_balance)
        
        start_balance = self.daily_stats[self.today]["start_balance"]
        
        # Calculate current loss percentage
        loss_pct = ((current_balance - start_balance) / start_balance) * 100
        
        # Get today's stats
        stats = {
            "daily_pnl": current_balance - start_balance,
            "daily_pnl_pct": loss_pct,
            "trades_taken": self.daily_stats[self.today]["trades_taken"],
            "trades_blocked": self.daily_stats[self.today]["trades_blocked"],
            "remaining_room": self.max_daily_loss_pct + loss_pct  # How much room left
        }
        
        # Check if limit hit
        if loss_pct <= -self.max_daily_loss_pct:
            self.daily_stats[self.today]["limit_hit"] = True
            self.daily_stats[self.today]["trades_blocked"] += 1
            self._save_daily_stats()
            
            return False, f"🚨 DAILY LOSS LIMIT HIT! ({loss_pct:.1f}% loss, max: {self.max_daily_loss_pct}%)", stats
        
        # Trade allowed
        allowed_reason = f"✅ Trade allowed (Daily P&L: {loss_pct:+.1f}%, Room: {stats['remaining_room']:.1f}%)"
        return True, allowed_reason, stats
    
    def record_trade_decision(self, approved: bool):
        """Record whether trade was approved by user"""
        if approved:
            self.daily_stats[self.today]["trades_taken"] += 1
        else:
            self.daily_stats[self.today]["trades_blocked"] += 1
        self._save_daily_stats()
    
    def get_daily_summary(self) -> str:
        """Get formatted daily summary for user"""
        today_stats = self.daily_stats[self.today]
        
        start_bal = today_stats.get("start_balance", 0)
        current_bal = today_stats.get("current_balance", start_bal)
        pnl = current_bal - start_bal if start_bal else 0
        pnl_pct = (pnl / start_bal * 100) if start_bal else 0
        
        emoji = "✅" if pnl >= 0 else "🔴"
        
        summary = f"""📊 **DAILY PERFORMANCE SUMMARY**
━━━━━━━━━━━━━━━━━━
{emoji} **P&L:** ${pnl:+.2f} ({pnl_pct:+.1f}%)
💰 **Balance:** ${current_bal:,.2f}
📈 **Trades Taken:** {today_stats['trades_taken']}
🚫 **Trades Blocked:** {today_stats['trades_blocked']}
⚠️ **Limit Status:** {'HIT' if today_stats.get('limit_hit') else 'OK'}
🎯 **Max Daily Loss:** {self.max_daily_loss_pct}%"""
        
        return summary
    
    def reset_if_new_day(self):
        """Check and reset if new trading day"""
        current_date = datetime.now().date().isoformat()
        
        if current_date != self.today:
            logger.info(f"📅 New trading day: {current_date}")
            self.today = current_date
            
            if self.today not in self.daily_stats:
                self.daily_stats[self.today] = {
                    "start_balance": None,
                    "current_balance": None,
                    "trades_taken": 0,
                    "trades_blocked": 0,
                    "total_pnl": 0.0,
                    "limit_hit": False
                }
                self._save_daily_stats()
