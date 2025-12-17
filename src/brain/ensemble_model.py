"""
Ensemble Model - RL + LSTM Voting System
Combines predictions from multiple AI models for more robust trading signals

Professional implementation with:
- Weighted voting based on model confidence
- Dynamic weight adjustment based on recent performance
- Conflict resolution strategies
- Confidence calibration
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("ENSEMBLE_MODEL")


class VotingStrategy(Enum):
    """Voting strategies for combining model predictions."""
    MAJORITY = "majority"           # Simple majority vote
    WEIGHTED = "weighted"           # Weight by confidence
    UNANIMOUS = "unanimous"         # All must agree
    CONFIDENCE_THRESHOLD = "threshold"  # Above threshold wins

# -*- coding: utf-8 -*-
"""
DEMIR AI - Enhanced Ensemble Model
Multi-model voting ensemble for maximum accuracy

PHASE 47: ML Enhancements
- Added XGBoost (gradient boosting)
- Added LightGBM (fast gradient boosting)
- Voting ensemble strategy
- Feature importance tracking
"""
import pickle
import os
from pathlib import Path

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class EnhancedEnsembleModel:
    """
    Enhanced Ensemble Model with XGBoost + LightGBM
    
    PHASE 47 Upgrade:
    - GradientBoosting (sklearn) - baseline
    - RandomForest (sklearn) - diversity
    - XGBoost - performance
    - LightGBM - speed
    - Ridge Regression - regularization
    
    Voting: Weighted average of all models
    """
    
    def __init__(self):
        self.models = {}


@dataclass
class ModelPrediction:
    """Represents a single model's prediction."""
    model_name: str
    signal: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    raw_value: float  # Original model output


class EnsembleModel:
    """
    Professional Ensemble Model combining RL and LSTM predictions.
    
    Features:
    - Weighted voting system
    - Performance-based weight adjustment
    - Conflict resolution
    - Confidence calibration
    """
    
    def __init__(
        self,
        voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        min_agreement: float = 0.6,  # 60% agreement required
        confidence_threshold: float = 65.0,  # Min confidence to consider
        rl_base_weight: float = 0.55,  # RL slightly higher (proven sharpe)
        lstm_base_weight: float = 0.45,
        enable_dynamic_weights: bool = True
    ):
        self.voting_strategy = voting_strategy
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold
        self.rl_base_weight = rl_base_weight
        self.lstm_base_weight = lstm_base_weight
        self.enable_dynamic_weights = enable_dynamic_weights
        
        # Performance tracking for dynamic weights
        self.rl_performance = {"wins": 0, "losses": 0, "neutral": 0}
        self.lstm_performance = {"wins": 0, "losses": 0, "neutral": 0}
        
        # Current dynamic weights
        self.rl_weight = rl_base_weight
        self.lstm_weight = lstm_base_weight
        
        logger.info(f"Ensemble Model initialized: strategy={voting_strategy.value}, RL={rl_base_weight}, LSTM={lstm_base_weight}")
    
    def predict(
        self,
        rl_action: int,
        rl_confidence: float,
        lstm_prediction: float,
        lstm_confidence: float
    ) -> Tuple[str, float, str]:
        """
        Combine RL and LSTM predictions into ensemble decision.
        
        Args:
            rl_action: RL model action (0=SELL, 1=HOLD, 2=BUY)
            rl_confidence: RL model confidence (0-100)
            lstm_prediction: LSTM predicted price change (-1 to 1)
            lstm_confidence: LSTM model confidence (0-100)
            
        Returns:
            (signal, confidence, reason)
        """
        # Convert RL action to signal
        rl_signal = self._action_to_signal(rl_action)
        
        # Convert LSTM prediction to signal
        lstm_signal = self._lstm_to_signal(lstm_prediction)
        
        # Create prediction objects
        predictions = [
            ModelPrediction("RL_v5", rl_signal, rl_confidence, float(rl_action)),
            ModelPrediction("LSTM_v11", lstm_signal, lstm_confidence, lstm_prediction)
        ]
        
        # Apply voting strategy
        if self.voting_strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(predictions)
        elif self.voting_strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(predictions)
        elif self.voting_strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous_vote(predictions)
        else:
            return self._threshold_vote(predictions)
    
    def _weighted_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float, str]:
        """Weighted voting based on model confidence and performance."""
        
        # Calculate weighted scores for each signal
        signal_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0
        
        for pred in predictions:
            # Get model weight
            if "RL" in pred.model_name:
                weight = self.rl_weight
            else:
                weight = self.lstm_weight
            
            # Confidence-adjusted weight
            conf_weight = weight * (pred.confidence / 100)
            signal_scores[pred.signal] += conf_weight
            total_weight += conf_weight
        
        # Normalize scores
        if total_weight > 0:
            for signal in signal_scores:
                signal_scores[signal] /= total_weight
        
        # Get winning signal
        winning_signal = max(signal_scores, key=signal_scores.get)
        winning_score = signal_scores[winning_signal]
        
        # Calculate ensemble confidence
        ensemble_confidence = winning_score * 100
        
        # Check agreement level
        if winning_score < self.min_agreement:
            winning_signal = "HOLD"
            reason = f"Low agreement ({winning_score:.1%}), defaulting to HOLD"
        else:
            # Build reason
            model_signals = [f"{p.model_name}={p.signal}({p.confidence:.0f}%)" for p in predictions]
            reason = f"Weighted vote: {', '.join(model_signals)}"
        
        logger.info(f"Ensemble: {winning_signal} ({ensemble_confidence:.1f}%) - {reason}")
        
        return winning_signal, ensemble_confidence, reason
    
    def _majority_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float, str]:
        """Simple majority voting."""
        signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for pred in predictions:
            signal_counts[pred.signal] += 1
        
        winning_signal = max(signal_counts, key=signal_counts.get)
        agreement = signal_counts[winning_signal] / len(predictions)
        
        if agreement < self.min_agreement:
            return "HOLD", 50.0, "No majority agreement"
        
        avg_confidence = np.mean([p.confidence for p in predictions if p.signal == winning_signal])
        return winning_signal, avg_confidence, f"Majority: {signal_counts[winning_signal]}/{len(predictions)}"
    
    def _unanimous_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float, str]:
        """All models must agree."""
        signals = set(p.signal for p in predictions)
        
        if len(signals) == 1:
            signal = signals.pop()
            avg_confidence = np.mean([p.confidence for p in predictions])
            return signal, avg_confidence, "Unanimous agreement"
        
        return "HOLD", 50.0, "Models disagree, holding"
    
    def _threshold_vote(self, predictions: List[ModelPrediction]) -> Tuple[str, float, str]:
        """Only count predictions above confidence threshold."""
        valid_predictions = [p for p in predictions if p.confidence >= self.confidence_threshold]
        
        if not valid_predictions:
            return "HOLD", 50.0, "No predictions above threshold"
        
        return self._weighted_vote(valid_predictions)
    
    def _action_to_signal(self, action: int) -> str:
        """Convert RL action to signal string."""
        if action == 2:
            return "BUY"
        elif action == 0:
            return "SELL"
        return "HOLD"
    
    def _lstm_to_signal(self, prediction: float) -> str:
        """Convert LSTM prediction to signal string."""
        if prediction > 0.01:  # >1% expected increase
            return "BUY"
        elif prediction < -0.01:  # >1% expected decrease
            return "SELL"
        return "HOLD"
    
    def update_performance(self, model: str, result: str):
        """
        Update model performance for dynamic weight adjustment.
        
        Args:
            model: "RL" or "LSTM"
            result: "win", "loss", or "neutral"
        """
        if model == "RL":
            perf = self.rl_performance
        else:
            perf = self.lstm_performance
        
        if result == "win":
            perf["wins"] += 1
        elif result == "loss":
            perf["losses"] += 1
        else:
            perf["neutral"] += 1
        
        # Recalculate dynamic weights
        if self.enable_dynamic_weights:
            self._recalculate_weights()
    
    def _recalculate_weights(self):
        """Recalculate model weights based on recent performance."""
        rl_total = sum(self.rl_performance.values()) or 1
        lstm_total = sum(self.lstm_performance.values()) or 1
        
        rl_win_rate = self.rl_performance["wins"] / rl_total
        lstm_win_rate = self.lstm_performance["wins"] / lstm_total
        
        # Adjust weights (max 20% deviation from base)
        total_win_rate = rl_win_rate + lstm_win_rate or 1
        
        rl_adjusted = self.rl_base_weight * (1 + 0.2 * (rl_win_rate / total_win_rate - 0.5))
        lstm_adjusted = self.lstm_base_weight * (1 + 0.2 * (lstm_win_rate / total_win_rate - 0.5))
        
        # Normalize
        total = rl_adjusted + lstm_adjusted
        self.rl_weight = rl_adjusted / total
        self.lstm_weight = lstm_adjusted / total
        
        logger.info(f"Dynamic weights updated: RL={self.rl_weight:.2f}, LSTM={self.lstm_weight:.2f}")
    
    def get_status(self) -> Dict:
        """Get current ensemble model status."""
        return {
            "voting_strategy": self.voting_strategy.value,
            "rl_weight": self.rl_weight,
            "lstm_weight": self.lstm_weight,
            "rl_performance": self.rl_performance,
            "lstm_performance": self.lstm_performance,
            "min_agreement": self.min_agreement,
            "confidence_threshold": self.confidence_threshold
        }
