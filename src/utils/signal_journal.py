"""
Signal Journal - Performance Tracking System
Phase 29.5

Tracks all trading signals (sent + rejected) and calculates performance metrics:
- Win rate by coin
- Average R:R achieved
- Signal quality distribution
- Best performing setups

ALL DATA FROM REAL SIGNALS - NO MOCKS!
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import asyncio

logger = logging.getLogger("SIGNAL_JOURNAL")


@dataclass
class SignalEntry:
    """Single signal entry in the journal"""
    signal_id: str
    timestamp: str
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    urgency: str  # CRITICAL/HIGH/MEDIUM/LOW
    quality: str  # STRONG/MODERATE/WEAK
    status: str  # SENT, REJECTED, TP_HIT, SL_HIT, EXPIRED
    reason: str
    outcome_price: float = 0.0
    outcome_pnl_pct: float = 0.0
    outcome_time: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SignalJournal:
    """
    Tracks all signals and calculates performance metrics.
    
    Features:
    - Log all signals (sent + rejected)
    - Track outcomes when available
    - Calculate win rates
    - Export to CSV for analysis
    """
    
    JOURNAL_FILE = "data/signal_journal.json"
    MAX_ENTRIES = 1000  # Keep last 1000 signals
    
    def __init__(self):
        self.entries: List[SignalEntry] = []
        self._load_journal()
    
    def _load_journal(self):
        """Load existing journal from file"""
        try:
            if os.path.exists(self.JOURNAL_FILE):
                with open(self.JOURNAL_FILE, 'r') as f:
                    data = json.load(f)
                    self.entries = [SignalEntry(**e) for e in data]
                logger.info(f"📔 Loaded {len(self.entries)} journal entries")
        except Exception as e:
            logger.warning(f"Could not load journal: {e}")
            self.entries = []
    
    def _save_journal(self):
        """Save journal to file"""
        try:
            os.makedirs(os.path.dirname(self.JOURNAL_FILE), exist_ok=True)
            
            # Keep only last MAX_ENTRIES
            if len(self.entries) > self.MAX_ENTRIES:
                self.entries = self.entries[-self.MAX_ENTRIES:]
            
            with open(self.JOURNAL_FILE, 'w') as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
        except Exception as e:
            logger.error(f"Could not save journal: {e}")
    
    def log_signal(self, signal: Dict, urgency: str, status: str = "SENT") -> str:
        """
        Log a new signal to the journal.
        
        Args:
            signal: Signal dict with symbol, side, entry_price, etc.
            urgency: CRITICAL/HIGH/MEDIUM/LOW
            status: SENT or REJECTED
        
        Returns:
            signal_id for future reference
        """
        signal_id = f"{signal.get('symbol', 'UNK')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        entry = SignalEntry(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            symbol=signal.get('symbol', 'UNKNOWN'),
            side=signal.get('side', 'UNKNOWN'),
            entry_price=signal.get('entry_price', 0),
            stop_loss=signal.get('sl_price', 0),
            take_profit=signal.get('tp_price', 0),
            confidence=signal.get('confidence', 0),
            urgency=urgency,
            quality=signal.get('quality', 'UNKNOWN'),
            status=status,
            reason=signal.get('reason', '')[:100]
        )
        
        self.entries.append(entry)
        self._save_journal()
        
        logger.info(f"📔 Signal logged: {signal_id} [{status}]")
        return signal_id
    
    def update_outcome(self, signal_id: str, outcome_price: float, status: str) -> bool:
        """
        Update signal outcome when trade is closed.
        
        Args:
            signal_id: ID returned from log_signal
            outcome_price: Price at close
            status: TP_HIT, SL_HIT, or EXPIRED
        
        Returns:
            True if updated successfully
        """
        for entry in self.entries:
            if entry.signal_id == signal_id:
                entry.outcome_price = outcome_price
                entry.status = status
                entry.outcome_time = datetime.now().isoformat()
                
                # Calculate PnL percentage
                if entry.entry_price > 0:
                    if entry.side == 'BUY':
                        entry.outcome_pnl_pct = ((outcome_price - entry.entry_price) / entry.entry_price) * 100
                    else:  # SELL/SHORT
                        entry.outcome_pnl_pct = ((entry.entry_price - outcome_price) / entry.entry_price) * 100
                
                self._save_journal()
                logger.info(f"📔 Outcome updated: {signal_id} -> {status} ({entry.outcome_pnl_pct:.2f}%)")
                return True
        
        logger.warning(f"Signal not found: {signal_id}")
        return False
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """
        Get performance statistics for the last N days.
        
        Returns:
            {
                'total_signals': 100,
                'sent_signals': 80,
                'rejected_signals': 20,
                'completed_signals': 60,
                'win_count': 40,
                'loss_count': 20,
                'win_rate': 66.7,
                'avg_pnl_pct': 1.5,
                'best_trade_pct': 8.5,
                'worst_trade_pct': -3.2,
                'by_urgency': {...},
                'by_coin': {...}
            }
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in self.entries if datetime.fromisoformat(e.timestamp) > cutoff]
        
        if not recent:
            return {
                'total_signals': 0,
                'sent_signals': 0,
                'rejected_signals': 0,
                'completed_signals': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0
            }
        
        sent = [e for e in recent if e.status != 'REJECTED']
        rejected = [e for e in recent if e.status == 'REJECTED']
        completed = [e for e in recent if e.status in ['TP_HIT', 'SL_HIT', 'EXPIRED']]
        
        wins = [e for e in completed if e.outcome_pnl_pct > 0]
        losses = [e for e in completed if e.outcome_pnl_pct < 0]
        
        win_rate = (len(wins) / len(completed) * 100) if completed else 0
        avg_pnl = sum(e.outcome_pnl_pct for e in completed) / len(completed) if completed else 0
        
        best_trade = max((e.outcome_pnl_pct for e in completed), default=0)
        worst_trade = min((e.outcome_pnl_pct for e in completed), default=0)
        
        return {
            'total_signals': len(recent),
            'sent_signals': len(sent),
            'rejected_signals': len(rejected),
            'completed_signals': len(completed),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': round(win_rate, 1),
            'avg_pnl_pct': round(avg_pnl, 2),
            'best_trade_pct': round(best_trade, 2),
            'worst_trade_pct': round(worst_trade, 2),
            'by_urgency': self._stats_by_field(completed, 'urgency'),
            'by_coin': self._stats_by_field(completed, 'symbol'),
            'period_days': days
        }
    
    def _stats_by_field(self, entries: List[SignalEntry], field: str) -> Dict:
        """Calculate win rate grouped by a field"""
        groups = {}
        for e in entries:
            key = getattr(e, field, 'UNKNOWN')
            if key not in groups:
                groups[key] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            
            if e.outcome_pnl_pct > 0:
                groups[key]['wins'] += 1
            elif e.outcome_pnl_pct < 0:
                groups[key]['losses'] += 1
            groups[key]['total_pnl'] += e.outcome_pnl_pct
        
        # Calculate win rates
        result = {}
        for key, data in groups.items():
            total = data['wins'] + data['losses']
            result[key] = {
                'win_rate': round((data['wins'] / total * 100) if total > 0 else 0, 1),
                'total_trades': total,
                'avg_pnl': round(data['total_pnl'] / total if total > 0 else 0, 2)
            }
        
        return result
    
    def get_win_rate_by_coin(self, days: int = 30) -> Dict:
        """Get win rate breakdown by coin"""
        stats = self.get_performance_stats(days)
        return stats.get('by_coin', {})
    
    def get_recent_signals(self, count: int = 20) -> List[Dict]:
        """Get most recent N signals"""
        recent = self.entries[-count:] if len(self.entries) >= count else self.entries
        return [e.to_dict() for e in reversed(recent)]
    
    def export_to_csv(self, filepath: str = "data/signal_journal_export.csv") -> bool:
        """Export journal to CSV file"""
        try:
            import csv
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', newline='') as f:
                if not self.entries:
                    return False
                
                writer = csv.DictWriter(f, fieldnames=self.entries[0].to_dict().keys())
                writer.writeheader()
                for entry in self.entries:
                    writer.writerow(entry.to_dict())
            
            logger.info(f"📔 Exported {len(self.entries)} signals to {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def get_dashboard_summary(self) -> Dict:
        """
        Get summary data for dashboard display.
        """
        stats = self.get_performance_stats(30)
        recent = self.get_recent_signals(5)
        
        return {
            'stats': stats,
            'recent_signals': recent,
            'last_updated': datetime.now().isoformat()
        }


# Singleton instance
_journal_instance: Optional[SignalJournal] = None


def get_signal_journal() -> SignalJournal:
    """Get or create the singleton SignalJournal instance"""
    global _journal_instance
    if _journal_instance is None:
        _journal_instance = SignalJournal()
    return _journal_instance


# Quick test
if __name__ == "__main__":
    journal = SignalJournal()
    
    # Test log signal
    test_signal = {
        'symbol': 'BTC/USDT',
        'side': 'BUY',
        'entry_price': 97500,
        'sl_price': 95000,
        'tp_price': 102000,
        'confidence': 85,
        'quality': 'STRONG',
        'reason': 'RL+LSTM Ensemble Bullish'
    }
    
    signal_id = journal.log_signal(test_signal, urgency='HIGH', status='SENT')
    print(f"Logged signal: {signal_id}")
    
    # Simulate outcome
    journal.update_outcome(signal_id, outcome_price=101000, status='TP_HIT')
    
    # Get stats
    stats = journal.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(json.dumps(stats, indent=2))
    
    # Get recent signals
    recent = journal.get_recent_signals(5)
    print(f"\nRecent Signals: {len(recent)}")
