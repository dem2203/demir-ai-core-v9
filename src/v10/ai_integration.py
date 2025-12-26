# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - AI INTEGRATION BRIDGE
====================================
Phase 2 Brain Logic Activation.
Connects the Rule-Based Engine with Deep Learning Models (RL & LSTM).
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from src.brain.rl_agent.ppo_agent import RLAgent
from src.brain.feature_engineering import FeatureEngineer
from src.v10.data_hub import MarketSnapshot

logger = logging.getLogger("AI_BRIDGE")

class AIIntegration:
    """
    Bridge between Live Market Engine and AI Brain.
    Manages model loading, state construction, and inference.
    """
    
    def __init__(self):
        self.rl_agent = RLAgent()
        self.feature_builder = FeatureEngineer() # Using the detected class name
        self.models_loaded = False
        
    async def initialize(self):
        """Load trained models"""
        try:
            # Try loading RL model
            success = self.rl_agent.load("ppo_btcusdt_v1")
            
            # Load LSTM (Placeholder logic for now if file not found)
            # self.lstm = ...
            
            if success:
                logger.info("✅ AI Integration: Models loaded successfully")
                self.models_loaded = True
            else:
                logger.warning("⚠️ AI Integration: No trained models found. Running in data collection mode.")
                
        except Exception as e:
            logger.error(f"❌ AI Init Error: {e}")

    async def get_consensus(self, symbol: str, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Get combined AI decision (RL + LSTM + Rules)
        """
        result = {
            "action": "HOLD",
            "direction": "NEUTRAL",
            "confidence": 0,
            "rl_score": 0,
            "lstm_score": 0,
            "reasoning": []
        }
        
        # DataHub NO_DATA check
        if not snapshot.raw_klines:
            return result
            
        try:
            # 1. Convert Data
            klines_dicts = self._convert_klines_to_dicts(snapshot.raw_klines)
            
            # 2. Feature Engineering
            df = self.feature_builder.process_data(klines_dicts)
            if df is None or df.empty:
                return result
                
            # 3. Build State Vector (37 dim)
            # Take last row
            last_row = df.iloc[-1]
            
            # Select numeric columns deterministic way
            numeric_cols = df.select_dtypes(include=[np.number]).columns.sort_values()
            features = last_row[numeric_cols].values
            
            # Trim or Pad to 34 (Base features)
            if len(features) > 34:
                features = features[:34]
            elif len(features) < 34:
                features = np.pad(features, (0, 34 - len(features)))
                
            # Add Position Info (3 dim) -> Assuming NO POSITION for inference since engine handles positions
            # [is_in_pos, entry_delta, unrealized_pnl]
            pos_info = np.array([0.0, 0.0, 0.0])
            
            state = np.concatenate([features, pos_info])
            
            # 4. RL Prediction
            # If model not loaded, this returns (0, 0.0)
            action_idx, conf = self.rl_agent.predict(state)
            
            # Map action
            actions = {0: "HOLD", 1: "BUY", 2: "SELL"}
            ai_action = actions.get(action_idx, "HOLD")
            
            result['action'] = ai_action
            result['rl_score'] = action_idx
            result['confidence'] = conf
            
            if ai_action == "BUY":
                result['direction'] = "BULLISH"
                result['reasoning'].append(f"RL Agent BUY Signal ({conf:.1f}%)")
            elif ai_action == "SELL":
                result['direction'] = "BEARISH"
                result['reasoning'].append(f"RL Agent SELL Signal ({conf:.1f}%)")
            
            # Add LSTM Logic here if available
            # ...
            
            return result
            
        except Exception as e:
            logger.error(f"AI Inference Error: {e}")
            return result

    def _convert_klines_to_dicts(self, raw_klines: List[Any]) -> List[Dict]:
        """Convert Binance raw klines to List of Dicts"""
        data = []
        for k in raw_klines:
            # Ensure proper types
            try:
                data.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            except:
                continue
        return data

_ai_bridge = None

def get_ai_bridge():
    global _ai_bridge
    if _ai_bridge is None:
        _ai_bridge = AIIntegration()
    return _ai_bridge
