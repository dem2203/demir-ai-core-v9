"""
Predictive Volatility Scaler - Professional ML System

Forecasts short-term volatility (next 1-4 hours) using:
- Exponential smoothing on ATR
- Volatility clustering detection
- GARCH-inspired scaling

Adjusts confidence and position sizing preemptively.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple
from collections import deque

logger = logging.getLogger("VOLATILITY_PREDICTOR")

class PredictiveVolatilityScaler:
    """
    Predicts short-term volatility for proactive risk management.
    
    Features:
    - Exponential weighted moving average (EWMA) for ATR forecast
    - Volatility clustering detection
    - Dynamic confidence & position size scaling
    """
    
    def __init__(self, ewma_alpha=0.3, lookback=50):
        """
        Args:
            ewma_alpha: Smoothing factor (0.2-0.4 typical)
            lookback: Number of periods to store for clustering
        """
        self.ewma_alpha = ewma_alpha
        self.lookback = lookback
        
        # Historical ATR storage
        self.atr_history = deque(maxlen=lookback)
        
        # EWMA state
        self.ewma_atr = None
    
    def update(self, current_atr: float):
        """
        Update volatility state with new ATR reading.
        
        Args:
            current_atr: Current ATR value
        """
        self.atr_history.append(current_atr)
        
        # Update EWMA
        if self.ewma_atr is None:
            self.ewma_atr = current_atr
        else:
            self.ewma_atr = self.ewma_alpha * current_atr + (1 - self.ewma_alpha) * self.ewma_atr
    
    def predict_next_atr(self) -> float:
        """
        Forecast next-period ATR using EWMA.
        
        Returns:
            predicted_atr: Forecasted ATR value
        """
        if self.ewma_atr is None:
            if self.atr_history:
                return float(np.mean(self.atr_history))
            return 0.0
        
        # EWMA forecast is simply the current EWMA value
        # (optimal 1-step ahead forecast for exponential smoothing)
        return self.ewma_atr
    
    def detect_volatility_regime(self, current_atr: float) -> Dict:
        """
        Detect volatility clustering (GARCH-inspired).
        
        Args:
            current_atr: Current ATR
        
        Returns:
            {
                "regime": "HIGH_VOL" | "NORMAL_VOL" | "LOW_VOL",
                "clustering_score": 0.0-1.0,
                "forecast_atr": float
            }
        """
        if len(self.atr_history) < 10:
            return {
                "regime": "NORMAL_VOL",
                "clustering_score": 0.5,
                "forecast_atr": current_atr
            }
        
        # Calculate statistics
        atr_array = np.array(self.atr_history)
        mean_atr = np.mean(atr_array)
        std_atr = np.std(atr_array)
        
        # Forecast
        forecast_atr = self.predict_next_atr()
        
        # Z-score
        if std_atr > 0:
            z_score = (forecast_atr - mean_atr) / std_atr
        else:
            z_score = 0.0
        
        # Clustering score: how far from mean?
        clustering_score = min(abs(z_score) / 2.0, 1.0)  # 0-1 scale
        
        # Regime classification
        if z_score > 1.5:
            regime = "HIGH_VOL"
        elif z_score < -1.0:
            regime = "LOW_VOL"
        else:
            regime = "NORMAL_VOL"
        
        logger.info(f"📊 Volatility Regime: {regime} | Forecast ATR: {forecast_atr:.2f} | Clustering: {clustering_score:.2f}")
        
        return {
            "regime": regime,
            "clustering_score": clustering_score,
            "forecast_atr": forecast_atr,
            "z_score": z_score
        }
    
    def get_confidence_adjustment(self, current_atr: float) -> int:
        """
        Get confidence penalty based on predicted volatility.
        
        Args:
            current_atr: Current ATR
        
        Returns:
            confidence_delta: Amount to adjust confidence (-2 to +1)
        """
        vol_info = self.detect_volatility_regime(current_atr)
        
        regime = vol_info["regime"]
        z_score = vol_info["z_score"]
        
        if regime == "HIGH_VOL":
            # High predicted vol → reduce confidence
            if z_score > 2.0:
                return -2  # Very high vol
            return -1
        
        elif regime == "LOW_VOL":
            # Low predicted vol → slightly boost confidence (stable market)
            return +1
        
        else:
            # Normal vol → no adjustment
            return 0
    
    def get_position_size_multiplier(self, current_atr: float) -> float:
        """
        Get position size multiplier based on predicted volatility.
        
        Args:
            current_atr: Current ATR
        
        Returns:
            multiplier: 0.5-1.5 (scale position size)
        """
        vol_info = self.detect_volatility_regime(current_atr)
        
        regime = vol_info["regime"]
        z_score = vol_info["z_score"]
        
        if regime == "HIGH_VOL":
            # High vol → reduce position size
            if z_score > 2.5:
                return 0.5  # Extreme vol: 50% size
            elif z_score > 1.5:
                return 0.75  # High vol: 75% size
            return 0.9
        
        elif regime == "LOW_VOL":
            # Low vol → slightly increase size (better R:R)
            return 1.1  # 110% size
        
        else:
            # Normal vol → standard size
            return 1.0
    
    def get_volatility_summary(self, current_atr: float) -> str:
        """
        Get human-readable volatility summary.
        
        Args:
            current_atr: Current ATR
        
        Returns:
            summary: Text summary for logging/debugging
        """
        vol_info = self.detect_volatility_regime(current_atr)
        
        regime = vol_info["regime"]
        forecast = vol_info["forecast_atr"]
        conf_adj = self.get_confidence_adjustment(current_atr)
        size_mult = self.get_position_size_multiplier(current_atr)
        
        summary = f"""
📈 Volatility Forecast:
- Current ATR: {current_atr:.2f}
- Forecast ATR: {forecast:.2f}
- Regime: {regime}
- Confidence Adjustment: {conf_adj:+d}
- Position Size Multiplier: {size_mult:.2f}x
"""
        return summary.strip()
