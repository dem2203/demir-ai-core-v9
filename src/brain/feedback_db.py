# -*- coding: utf-8 -*-
"""
DEMIR AI - Feedback Database
=============================
Trade outcome'ları kaydeder ve self-learning için temel oluşturur.

Her kapanan trade:
- Entry features (RSI, OB, Funding, etc.)
- Predicted action
- Actual PnL
- Duration
- Market regime

Bu veri ile modeller:
- Online learning yapabilir
- Performance tracking yapabilir
- Regime-specific adaptasyon yapabilir
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("FEEDBACK_DB")


class FeedbackDB:
    """Trade outcome feedback database"""
    
    def __init__(self):
        self.db_path = Path("src/brain/storage/feedback_trades.json")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
    
    def _load(self) -> List[Dict]:
        """Load existing feedback data"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load feedback DB: {e}")
                return []
        return []
    
    def _save(self):
        """Save feedback data to disk"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback DB: {e}")
    
    def save_trade_outcome(self, trade_data: Dict):
        """
        Save a completed trade outcome.
        
        Args:
            trade_data: {
                'symbol': 'BTCUSDT',
                'side': 'LONG' or 'SHORT',
                'entry_features': {
                    'rsi': 65,
                    'ob_ratio': 2.5,
                    'funding': 0.008,
                    'volatility': 1.2,
                    'regime': 'TRENDING_BULL'
                },
                'predicted_action': 'BUY',
                'actual_pnl': 125.50,
                'pnl_pct': 2.5,
                'duration_minutes': 240,
                'entry_price': 87500,
                'exit_price': 89000
            }
        """
        trade_data['timestamp'] = datetime.now().isoformat()
        trade_data['trade_id'] = len(self.data) + 1
        
        self.data.append(trade_data)
        self._save()
        
        logger.info(f"✅ Feedback saved: {trade_data['symbol']} PnL: ${trade_data['actual_pnl']:.2f}")
    
    def get_last_n(self, n: int = 100) -> List[Dict]:
        """Get last N trades"""
        return self.data[-n:] if len(self.data) >= n else self.data
    
    def get_by_symbol(self, symbol: str, n: int = 50) -> List[Dict]:
        """Get last N trades for a specific symbol"""
        symbol_trades = [t for t in self.data if t.get('symbol') == symbol]
        return symbol_trades[-n:] if len(symbol_trades) >= n else symbol_trades
    
    def get_by_regime(self, regime: str, n: int = 50) -> List[Dict]:
        """Get last N trades for a specific market regime"""
        regime_trades = [t for t in self.data if t.get('entry_features', {}).get('regime') == regime]
        return regime_trades[-n:] if len(regime_trades) >= n else regime_trades
    
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        if not self.data:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }
        
        total = len(self.data)
        wins = sum(1 for t in self.data if t.get('actual_pnl', 0) > 0)
        total_pnl = sum(t.get('actual_pnl', 0) for t in self.data)
        
        return {
            'total_trades': total,
            'win_rate': wins / total if total > 0 else 0,
            'avg_pnl': total_pnl / total if total > 0 else 0,
            'total_pnl': total_pnl,
            'best_trade': max(self.data, key=lambda t: t.get('actual_pnl', 0))['actual_pnl'] if self.data else 0,
            'worst_trade': min(self.data, key=lambda t: t.get('actual_pnl', 0))['actual_pnl'] if self.data else 0
        }
    
    def get_regime_stats(self) -> Dict:
        """Get statistics per market regime"""
        regimes = {}
        
        for trade in self.data:
            regime = trade.get('entry_features', {}).get('regime', 'UNKNOWN')
            
            if regime not in regimes:
                regimes[regime] = {'trades': [], 'wins': 0, 'total_pnl': 0}
            
            regimes[regime]['trades'].append(trade)
            if trade.get('actual_pnl', 0) > 0:
                regimes[regime]['wins'] += 1
            regimes[regime]['total_pnl'] += trade.get('actual_pnl', 0)
        
        # Calculate stats
        stats = {}
        for regime, data in regimes.items():
            total = len(data['trades'])
            stats[regime] = {
                'total_trades': total,
                'win_rate': data['wins'] / total if total > 0 else 0,
                'avg_pnl': data['total_pnl'] / total if total > 0 else 0,
                'total_pnl': data['total_pnl']
            }
        
        return stats
    
    def clear_old_trades(self, keep_last_n: int = 1000):
        """Keep only last N trades to prevent DB from growing too large"""
        if len(self.data) > keep_last_n:
            self.data = self.data[-keep_last_n:]
            self._save()
            logger.info(f"Cleared old trades, kept last {keep_last_n}")


# Global instance
_feedback_db: Optional[FeedbackDB] = None


def get_feedback_db() -> FeedbackDB:
    """Get or create FeedbackDB singleton"""
    global _feedback_db
    if _feedback_db is None:
        _feedback_db = FeedbackDB()
    return _feedback_db


# CLI for testing
if __name__ == "__main__":
    db = get_feedback_db()
    
    # Test save
    db.save_trade_outcome({
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'entry_features': {
            'rsi': 65,
            'ob_ratio': 2.5,
            'funding': 0.008,
            'regime': 'TRENDING_BULL'
        },
        'predicted_action': 'BUY',
        'actual_pnl': 50.0,
        'pnl_pct': 2.0,
        'duration_minutes': 120,
        'entry_price': 87500,
        'exit_price': 89000
    })
    
    # Test stats
    print("\n=== Overall Stats ===")
    stats = db.get_stats()
    for key, val in stats.items():
        print(f"{key}: {val}")
    
    print("\n=== Regime Stats ===")
    regime_stats = db.get_regime_stats()
    for regime, stats in regime_stats.items():
        print(f"\n{regime}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
