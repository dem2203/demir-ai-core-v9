
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

# Configure Logging
logger = logging.getLogger("FEEDBACK_LOOP")

class FeedbackLoop:
    """
    Auto-Retrain Bridge
    ===================
    Connects PaperTrader results with EarlySignalEngine features to create labeled training data.
    
    Flow:
    1. Read 'PaperTrader' history (Closed Trades).
    2. For each closed trade, find matching Feature Vector in 'src/v10/training_data'.
    3. Label the data:
       - PnL > 0 -> Label 1 (BUY Correct)
       - PnL < 0 -> Label 0 (Incorrect / SELL if opposite)
    4. Save 'Labeled Feedback Data' to 'src/v10/feedback_data'.
    """
    
    TRAINING_DATA_DIR = Path("src/v10/training_data")
    FEEDBACK_DATA_DIR = Path("src/v10/feedback_data")
    
    def __init__(self):
        self.FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    def process_closed_trades(self, trade_history: List[Dict]) -> int:
        """
        Process new closed trades and generate feedback data.
        Returns: Number of new labeled samples generated.
        """
        new_samples_count = 0
        
        # 1. Load all available feature data first (for efficient lookup)
        # This might be heavy later, but fine for now (json files are small batches)
        feature_map = self._load_feature_map() # Key: "SYMBOL_TIMESTAMP_HOUR", Value: Sample
        
        # 2. Iterate through trades
        for trade in trade_history:
            # Skip if already processed (check if we have a generated feedback file for this trade ID/Time)
            # Simple check: trade time as ID
            trade_id = f"{trade['symbol']}_{trade['time']}"
            if self._is_trade_processed(trade_id):
                continue
            
            # 3. Find matching feature vector
            # Trade 'entry' time is when signal generated.
            # Match strictly on Symbol and leniently on Time (+- 10 mins)
            matched_feature = self._find_match(trade, feature_map)
            
            if matched_feature:
                # 4. Label it
                labeled_sample = self._create_labeled_sample(trade, matched_feature)
                
                # 5. Save
                self._save_feedback_sample(labeled_sample, trade_id)
                new_samples_count += 1
                logger.info(f"✨ Feedback Generated: {trade['symbol']} ({trade['pnl']}) -> Label: {labeled_sample['label']}")
            else:
                logger.debug(f"No feature match for trade {trade_id}")
                
        return new_samples_count

    def _load_feature_map(self) -> Dict[str, Dict]:
        """Loads all raw feature samples into memory for finding matches."""
        feature_map = {} # Key: (Symbol, TimeMinute), Value: Sample
        
        files = list(self.TRAINING_DATA_DIR.glob("training_*.json"))
        for fpath in files:
            try:
                with open(fpath, 'r') as f:
                    samples = json.load(f)
                    for s in samples:
                        # Index by minute to make lookup faster
                        # timestamp format: ISO string
                        ts = datetime.fromisoformat(s['timestamp'])
                        key = (s['symbol'], ts.strftime('%Y-%m-%d %H:%M'))
                        feature_map[key] = s
            except: pass
            
        return feature_map

    def _find_match(self, trade: Dict, feature_map: Dict) -> Optional[Dict]:
        """Finds a feature sample close to trade entry time."""
        trade_ts = datetime.fromisoformat(trade['time'])
        symbol = trade['symbol']
        
        # Search window: -5 to +5 minutes
        for i in range(-5, 6):
            search_ts = trade_ts + timedelta(minutes=i)
            key = (symbol, search_ts.strftime('%Y-%m-%d %H:%M'))
            
            if key in feature_map:
                return feature_map[key]
                
        return None

    def _create_labeled_sample(self, trade: Dict, feature_sample: Dict) -> Dict:
        """
        Creates a training sample with definitive label based on PnL.
        Label Logic:
        - If Side == BUY:
            - PnL > 0: Label 2 (BUY) - Prediction was correct
            - PnL <= 0: Label 0 (SELL) - Prediction was wrong (should have sold/held)       
        """
        side = trade.get('action', 'BUY') # PaperTrader saves 'action' as 'SELL' on close, wait.
        # PaperTrader history: 'action': 'SELL' means CLOSE. 'entry' price and 'exit' price determines PnL.
        # But we need to know if it was a LONG or SHORT trade initially.
        # PaperTrader currently only does LONG trades (implied).
        # So PnL > 0 means UP move.
        
        if trade['pnl'] > 0:
            label = 2 # BUY (Price went UP)
        else:
            # If we lost on a LONG, it means price went DOWN.
            label = 0 # SELL (Price went DOWN)
            
        # Create new sample
        return {
            "features": feature_sample['features'],
            "label": label, # 0=SELL, 1=HOLD, 2=BUY
            "source": "REAL_FEEDBACK",
            "timestamp": trade['time'],
            "symbol": trade['symbol'],
            "pnl": trade['pnl']
        }

    def _is_trade_processed(self, trade_id: str) -> bool:
        """Checks if a feedback file already exists for this trade."""
        safe_id = trade_id.replace(":", "-").replace(" ", "_")
        fpath = self.FEEDBACK_DATA_DIR / f"feedback_{safe_id}.json"
        return fpath.exists()

    def _save_feedback_sample(self, sample: Dict, trade_id: str):
        """Saves individual feedback sample."""
        safe_id = trade_id.replace(":", "-").replace(" ", "_")
        fpath = self.FEEDBACK_DATA_DIR / f"feedback_{safe_id}.json"
        
        with open(fpath, 'w') as f:
            json.dump(sample, f, indent=2)

