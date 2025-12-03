import pandas as pd
import numpy as np
import logging
from typing import Dict
from src.brain.regime_classifier import RegimeClassifier

logger = logging.getLogger("STRATEGY_SELECTOR")

class StrategySelector:
    """
    DEMIR AI V20.0 - ADAPTIVE STRATEGY ENGINE
    
    Automatically switches between different trading strategies based on market regime:
    - TRENDING: Momentum Strategy (Ride the wave)
    - SIDEWAYS: Mean Reversion Strategy (Buy low, sell high)
    - VOLATILE: Conservative (Reduce position sizes)
    """
    
    STRATEGIES = {
        "MOMENTUM": {
            "name": "Trend Following",
            "description": "Follows strong trends",
            "best_regime": ["TRENDING_UP", "TRENDING_DOWN"],
            "signal_threshold": 0.60,  # Need 60%+ LSTM confidence
            "position_multiplier": 1.0
        },
        "MEAN_REVERSION": {
            "name": "Range Trading",
            "description": "Buys oversold, sells overbought",
            "best_regime": ["SIDEWAYS", "RANGING"],
            "signal_threshold": 0.65,  # Higher threshold for counter-trend
            "position_multiplier": 0.8  # Smaller positions
        },
        "CONSERVATIVE": {
            "name": "Risk-Off Mode",
            "description": "Minimal trading during uncertainty",
            "best_regime": ["HIGH_VOLATILITY", "CRASH"],
            "signal_threshold": 0.75,  # Very high threshold
            "position_multiplier": 0.5  # Half size
        }
    }
    
    def __init__(self):
        self.current_strategy = "MOMENTUM"  # Default
        self.regime_classifier = RegimeClassifier()
    
    def select_strategy(self, market_regime: str) -> str:
        """
        Selects the best strategy for the current market regime.
        """
        for strategy_name, config in self.STRATEGIES.items():
            if market_regime in config['best_regime']:
                if strategy_name != self.current_strategy:
                    logger.info(f"🔄 STRATEGY SWITCH: {self.current_strategy} → {strategy_name}")
                    logger.info(f"   Reason: Market regime is now {market_regime}")
                    self.current_strategy = strategy_name
                return strategy_name
        
        # Default: stay with current strategy
        return self.current_strategy
    
    def get_strategy_params(self, strategy_name: str = None) -> Dict:
        """
        Returns the parameters for a given strategy.
        """
        if strategy_name is None:
            strategy_name = self.current_strategy
        
        return self.STRATEGIES.get(strategy_name, self.STRATEGIES["MOMENTUM"])
    
    def should_take_signal(self, lstm_confidence: float, regime: str) -> bool:
        """
        Decides if a signal should be taken based on current strategy and confidence.
        """
        self.select_strategy(regime)  # Update strategy if regime changed
        
        params = self.get_strategy_params()
        threshold = params['signal_threshold']
        
        if lstm_confidence >= threshold:
            logger.info(f"✅ Signal ACCEPTED by {self.current_strategy} strategy")
            return True
        else:
            logger.info(f"❌ Signal REJECTED by {self.current_strategy} (Conf {lstm_confidence:.2f} < {threshold})")
            return False
    
    def adjust_position_size(self, base_kelly_size: float) -> float:
        """
        Adjusts the Kelly position size based on current strategy.
        """
        params = self.get_strategy_params()
        multiplier = params['position_multiplier']
        
        adjusted_size = base_kelly_size * multiplier
        
        logger.info(f"📊 Position Size Adjusted: {base_kelly_size:.2f}% → {adjusted_size:.2f}% ({self.current_strategy})")
        
        return adjusted_size
