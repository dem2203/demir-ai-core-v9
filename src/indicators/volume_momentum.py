"""
Volume Momentum Detector - Professional Pre-Pump Indicator

Tracks volume acceleration to predict price spikes BEFORE they happen.

Key Features:
- 5-minute rolling window
- Volume acceleration (1st derivative)
- Momentum scoring (0-10)
- Early warning: 30s-2min lead time

Algorithm:
- Track volume changes per minute
- Calculate acceleration: Δvolume / Δtime
- Threshold: 3x acceleration = PRE-PUMP signal
"""

import numpy as np
import pandas as pd
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Tuple

logger = logging.getLogger("VOLUME_MOMENTUM")

class VolumeAccelerationDetector:
    """
    Detects volume acceleration patterns that precede price spikes.
    
    Theory:
    - Volume typically spikes BEFORE price
    - Acceleration (2nd derivative) is leading indicator
    - 3x acceleration within 5 minutes = high probability pump
    """
    
    def __init__(self, lookback_minutes=5, acceleration_threshold=3.0):
        """
        Args:
            lookback_minutes: Rolling window for volume tracking
            acceleration_threshold: Multiplier for acceleration alert (3x = 300%)
        """
        self.lookback_minutes = lookback_minutes
        self.acceleration_threshold = acceleration_threshold
        
        # Historical volume storage
        # Format: deque of (timestamp, volume) tuples
        self.volume_history = deque(maxlen=lookback_minutes * 60)  # 1 per second max
        
        # Baseline volume (20-minute average)
        self.baseline_volumes = deque(maxlen=20)
    
    def update(self, current_volume: float, timestamp: datetime = None):
        """
        Update volume history with new data point.
        
        Args:
            current_volume: Current 1-minute candle volume
            timestamp: Timestamp of data (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.volume_history.append((timestamp, current_volume))
        self.baseline_volumes.append(current_volume)
    
    def calculate_acceleration(self) -> float:
        """
        Calculate volume acceleration (rate of change of rate of change).
        
        Returns:
            acceleration: Volume acceleration factor (1.0 = no change, 3.0 = 3x acceleration)
        """
        if len(self.volume_history) < 3:
            return 1.0  # Not enough data
        
        # Get last 5 minutes
        cutoff_time = datetime.now() - timedelta(minutes=self.lookback_minutes)
        recent_data = [(t, v) for t, v in self.volume_history if t >= cutoff_time]
        
        if len(recent_data) < 3:
            return 1.0
        
        # Extract volumes
        volumes = [v for _, v in recent_data]
        
        # Calculate 1st derivative (velocity)
        velocities = np.diff(volumes)
        
        # Calculate 2nd derivative (acceleration)
        if len(velocities) < 2:
            return 1.0
        
        accelerations = np.diff(velocities)
        
        # Average acceleration in recent period
        avg_acceleration = np.mean(accelerations) if len(accelerations) > 0 else 0
        
        # Normalize by baseline
        baseline = np.mean(self.baseline_volumes) if len(self.baseline_volumes) > 0 else 1
        
        if baseline == 0:
            return 1.0
        
        acceleration_factor = 1 + (avg_acceleration / baseline)
        
        return max(acceleration_factor, 0.1)  # Prevent negative
    
    def calculate_momentum_score(self) -> Dict:
        """
        Calculate momentum score (0-10) based on volume acceleration.
        
        Returns:
            {
                "score": 0-10,
                "acceleration": factor,
                "signal": "PRE_PUMP" | "NORMAL" | "DECLINING",
                "confidence": 0-10,
                "lead_time_estimate": "30s-2min" | None
            }
        """
        acceleration = self.calculate_acceleration()
        
        # Score calculation
        if acceleration >= self.acceleration_threshold:
            # High acceleration = PRE-PUMP
            score = min(10, int(acceleration * 2))
            signal = "PRE_PUMP"
            confidence = min(10, int((acceleration / self.acceleration_threshold) * 8))
            lead_time = "30s-2min"
            
        elif acceleration >= 2.0:
            # Moderate acceleration = BUILDING
            score = 7
            signal = "BUILDING"
            confidence = 6
            lead_time = "2-5min"
            
        elif acceleration >= 1.5:
            # Slight increase = NORMAL
            score = 5
            signal = "NORMAL"
            confidence = 5
            lead_time = None
            
        else:
            # Declining or no acceleration
            score = max(0, int(acceleration * 4))
            signal = "DECLINING" if acceleration < 1.0 else "NORMAL"
            confidence = 3
            lead_time = None
        
        logger.info(f"📊 Volume Momentum: Acceleration {acceleration:.2f}x | Score: {score}/10 | Signal: {signal}")
        
        return {
            "score": score,
            "acceleration": acceleration,
            "signal": signal,
            "confidence": confidence,
            "lead_time_estimate": lead_time,
            "raw_volumes": list(self.volume_history)[-10:] if len(self.volume_history) >= 10 else list(self.volume_history)
        }
    
    def detect_volume_spike_pattern(self) -> bool:
        """
        Detect classic volume spike pattern: gradual increase → sharp spike.
        
        Returns:
            is_pre_spike: True if pattern detected
        """
        if len(self.volume_history) < 5:
            return False
        
        recent_volumes = [v for _, v in list(self.volume_history)[-5:]]
        
        # Check if volumes are increasing
        increasing = all(recent_volumes[i] < recent_volumes[i+1] for i in range(len(recent_volumes)-1))
        
        # Check if rate of increase is accelerating
        if increasing and len(recent_volumes) >= 3:
            growth_rate_1 = recent_volumes[-2] / recent_volumes[-3] if recent_volumes[-3] > 0 else 1
            growth_rate_2 = recent_volumes[-1] / recent_volumes[-2] if recent_volumes[-2] > 0 else 1
            
            accelerating = growth_rate_2 > growth_rate_1 * 1.2
            
            return accelerating
        
        return False
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        result = self.calculate_momentum_score()
        
        summary = f"""
📈 Volume Momentum Analysis:
- Acceleration: {result['acceleration']:.2f}x
- Score: {result['score']}/10
- Signal: {result['signal']}
- Confidence: {result['confidence']}/10
- Lead Time: {result['lead_time_estimate'] or 'N/A'}
"""
        return summary.strip()
