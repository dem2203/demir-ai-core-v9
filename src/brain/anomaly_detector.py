import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from sklearn.ensemble import IsolationForest
from collections import deque

logger = logging.getLogger("ANOMALY_DETECTOR")

class AnomalyDetector:
    """
    DEMIR AI V22.0 - REAL-TIME ANOMALY DETECTOR
    
    Detects unusual market behavior (price spikes, volume surges) in real-time.
    Uses online learning to adapt to changing market conditions.
    """
    
    HISTORY_SIZE = 500  # Keep last 500 data points
    ANOMALY_THRESHOLD = 0.95  # 95th percentile = anomaly
    RETRAIN_INTERVAL = 100  # Retrain model every 100 new data points
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.data_count = 0
        self.is_trained = False
    
    def add_data_point(self, price: float, volume: float, volatility: float):
        """
        Adds a new data point to history.
        """
        self.history.append({
            'price_change': 0,  # Will calculate on next point
            'volume': volume,
            'volatility': volatility
        })
        self.data_count += 1
        
        # Calculate price change if we have previous data
        if len(self.history) >= 2:
            prev = list(self.history)[-2]
            curr = list(self.history)[-1]
            if 'price' in prev:
                curr['price_change'] = ((price - prev.get('price', price)) / prev.get('price', 1)) * 100
            curr['price'] = price
    
    def train(self):
        """
        Trains the anomaly detection model on historical data.
        """
        if len(self.history) < 50:
            logger.warning("Not enough data to train anomaly detector")
            return False
        
        try:
            # Convert history to DataFrame
            df = pd.DataFrame(list(self.history))
            
            # Features for anomaly detection
            features = df[['price_change', 'volume', 'volatility']].fillna(0).values
            
            # Fit model
            self.model.fit(features)
            self.is_trained = True
            logger.info(f"Anomaly detector trained on {len(features)} samples")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def detect_anomaly(self, price: float, volume: float, volatility: float) -> Optional[Dict]:
        """
        Checks if current market state is anomalous.
        
        Returns:
            {
                'is_anomaly': True/False,
                'anomaly_score': -0.15 (negative = anomaly),
                'metrics': {...}
            }
        """
        # Add to history
        self.add_data_point(price, volume, volatility)
        
        # Retrain periodically
        if self.data_count % self.RETRAIN_INTERVAL == 0:
            self.train()
        
        if not self.is_trained:
            if len(self.history) >= 50:
                self.train()
            else:
                return None
        
        try:
            # Calculate features
            if len(self.history) < 2:
                return None
            
            recent = list(self.history)[-10:]  # Last 10 data points
            
            # Average volume surge
            avg_volume = np.mean([d.get('volume', 0) for d in list(self.history)[:-10]])
            volume_surge = volume / avg_volume if avg_volume > 0 else 1
            
            # Price change
            price_change = recent[-1].get('price_change', 0)
            
            # Predict
            features = np.array([[price_change, volume, volatility]])
            anomaly_score = self.model.decision_function(features)[0]
            is_anomaly = self.model.predict(features)[0] == -1
            
            # Enhanced detection: also check extreme values
            extreme_volume = volume_surge > 3.0
            extreme_price = abs(price_change) > 2.0
            
            is_anomaly = is_anomaly or extreme_volume or extreme_price
            
            if is_anomaly:
                logger.warning(f"🚨 ANOMALY DETECTED: Score={anomaly_score:.3f}, Volume={volume_surge:.1f}x, Price={price_change:+.2f}%")
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'volume_surge': float(volume_surge),
                'price_change_pct': float(price_change),
                'volatility': float(volatility)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return None
