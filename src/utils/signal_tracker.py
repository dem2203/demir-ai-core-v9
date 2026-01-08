import json
import logging
import os
from datetime import datetime
from typing import Dict, List
from src.config import Config

logger = logging.getLogger("SIGNAL_TRACKER")

class SignalRecord:
    """Individual signal record"""
    def __init__(self, timestamp: str, symbol: str, signal_type: str, 
                 entry_price: float, ai_votes: dict, confidence: int):
        self.id = f"{symbol}_{timestamp.replace(':', '').replace('-', '')}"
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type  # LONG, SHORT
        self.entry_price = entry_price
        self.ai_votes = ai_votes  # {"Macro": "BULLISH", "Gemini": "BULLISH", ...}
        self.confidence = confidence
        
        # Outcome (filled later)
        self.exit_price = None
        self.exit_timestamp = None
        self.outcome = None  # "TP" (Take Profit) or "SL" (Stop Loss)
        self.pnl = None
        self.pnl_pct = None
        self.duration_hours = None
        
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "ai_votes": self.ai_votes,
            "confidence": self.confidence,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp,
            "outcome": self.outcome,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "duration_hours": self.duration_hours
        }
    
    @staticmethod
    def from_dict(data):
        record = SignalRecord(
            data["timestamp"],
            data["symbol"],
            data["signal_type"],
            data["entry_price"],
            data["ai_votes"],
            data["confidence"]
        )
        record.id = data["id"]
        record.exit_price = data.get("exit_price")
        record.exit_timestamp = data.get("exit_timestamp")
        record.outcome = data.get("outcome")
        record.pnl = data.get("pnl")
        record.pnl_pct = data.get("pnl_pct")
        record.duration_hours = data.get("duration_hours")
        return record

class SignalPerformanceTracker:
    """
    Tracks all signals and their outcomes for AI self-learning.
    """
    def __init__(self):
        self.db_path = os.path.join(Config.DATA_DIR, "signal_history.json")
        self.signals = {}  # {signal_id: SignalRecord}
        self._load_history()
        
    def _load_history(self):
        """Load signal history from disk"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.signals = {
                        k: SignalRecord.from_dict(v) 
                        for k, v in data.items()
                    }
                logger.info(f"📊 Loaded {len(self.signals)} historical signals")
            except Exception as e:
                logger.error(f"Failed to load signal history: {e}")
                self.signals = {}
        else:
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            
    def _save_history(self):
        """Save signal history to disk"""
        try:
            data = {k: v.to_dict() for k, v in self.signals.items()}
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save signal history: {e}")
    
    def log_signal(self, symbol: str, signal_type: str, entry_price: float, 
                   ai_votes: dict, confidence: int) -> str:
        """
        Log a new trading signal.
        Returns signal_id for later reference.
        """
        timestamp = datetime.now().isoformat()
        
        record = SignalRecord(
            timestamp, symbol, signal_type, entry_price, ai_votes, confidence
        )
        
        self.signals[record.id] = record
        self._save_history()
        
        logger.info(f"📝 Signal logged: {record.id}")
        return record.id
    
    def update_outcome(self, signal_id: str, exit_price: float, outcome: str, pnl: float):
        """
        Update signal outcome when trade closes.
        """
        if signal_id not in self.signals:
            logger.warning(f"Signal {signal_id} not found")
            return
            
        record = self.signals[signal_id]
        record.exit_price = exit_price
        record.exit_timestamp = datetime.now().isoformat()
        record.outcome = outcome  # "TP" or "SL"
        record.pnl = pnl
        record.pnl_pct = (pnl / (record.entry_price * abs(pnl / (exit_price - record.entry_price)))) * 100
        
        # Calculate duration
        try:
            entry_time = datetime.fromisoformat(record.timestamp)
            exit_time = datetime.fromisoformat(record.exit_timestamp)
            duration = (exit_time - entry_time).total_seconds() / 3600
            record.duration_hours = round(duration, 2)
        except:
            record.duration_hours = 0
            
        self._save_history()
        
        logger.info(f"✅ Outcome logged: {signal_id} → {outcome} | PnL: {pnl:.2f} ({record.pnl_pct:.1f}%)")
    
    def get_performance_stats(self) -> dict:
        """
        Calculate overall performance statistics.
        """
        completed = [s for s in self.signals.values() if s.outcome is not None]
        
        if not completed:
            return {"message": "No completed trades yet"}
        
        total_trades = len(completed)
        tp_trades = [s for s in completed if s.outcome == "TP"]
        sl_trades = [s for s in completed if s.outcome == "SL"]
        
        win_rate = (len(tp_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = sum(s.pnl for s in completed if s.pnl)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_duration = sum(s.duration_hours for s in completed if s.duration_hours) / total_trades
        
        return {
            "total_trades": total_trades,
            "wins": len(tp_trades),
            "losses": len(sl_trades),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 2),
            "avg_duration_hours": round(avg_duration, 2)
        }
    
    def analyze_ai_performance(self) -> dict:
        """
        Analyze which AI combinations perform best.
        """
        completed = [s for s in self.signals.values() if s.outcome is not None]
        
        if not completed:
            return {}
        
        # Analyze by confidence level
        high_conf = [s for s in completed if s.confidence >= 8]
        mid_conf = [s for s in completed if 5 <= s.confidence < 8]
        low_conf = [s for s in completed if s.confidence < 5]
        
        def calc_wr(signals):
            if not signals:
                return 0
            wins = len([s for s in signals if s.outcome == "TP"])
            return (wins / len(signals)) * 100
        
        return {
            "high_confidence_8+": {
                "count": len(high_conf),
                "win_rate": round(calc_wr(high_conf), 1)
            },
            "mid_confidence_5-7": {
                "count": len(mid_conf),
                "win_rate": round(calc_wr(mid_conf), 1)
            },
            "low_confidence_<5": {
                "count": len(low_conf),
                "win_rate": round(calc_wr(low_conf), 1)
            }
        }
    
    def get_ai_feedback_prompt(self) -> str:
        """
        Generate feedback text to show to AIs for self-improvement.
        """
        stats = self.get_performance_stats()
        ai_perf = self.analyze_ai_performance()
        
        if "message" in stats:
            return "No historical performance data yet. This is your first trade."
        
        recent_signals = sorted(
            [s for s in self.signals.values() if s.outcome],
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        feedback = f"""📊 HISTORICAL PERFORMANCE (Self-Learning Feedback):

Overall Stats:
- Total Trades: {stats['total_trades']}
- Win Rate: {stats['win_rate']}%
- Total PnL: {stats['total_pnl']} USDT
- Avg Trade Duration: {stats['avg_duration_hours']}h

AI Confidence Analysis:
- High Confidence (8-10): {ai_perf.get('high_confidence_8+', {}).get('win_rate', 0)}% win rate
- Mid Confidence (5-7): {ai_perf.get('mid_confidence_5-7', {}).get('win_rate', 0)}% win rate
- Low Confidence (<5): {ai_perf.get('low_confidence_<5', {}).get('win_rate', 0)}% win rate

Recent Trade Outcomes:
"""
        for sig in recent_signals[:5]:
            emoji = "✅" if sig.outcome == "TP" else "❌"
            feedback += f"{emoji} {sig.symbol} {sig.signal_type} → {sig.outcome} ({sig.pnl_pct:.1f}%)\n"
        
        feedback += "\n⚠️ LEARN FROM THIS: Adjust your confidence if certain patterns keep failing."
        
        return feedback
