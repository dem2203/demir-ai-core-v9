# -*- coding: utf-8 -*-
"""
DEMIR AI - LSTM Trend Model
Deep Learning ile 1-2 saat fiyat yönü tahmini.

PHASE 48: Advanced Prediction
- LSTM (Long Short-Term Memory) neural network
- Son 24 saatlik veriyi input olarak alır
- 1-2 saat sonrasını tahmin eder
- Teknik göstergelerle zenginleştirilmiş özellikler
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("LSTM_TREND")

# TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - LSTM predictions disabled")


class LSTMTrendPredictor:
    """
    LSTM-based Price Direction Predictor
    
    Input Features:
    - Close price (normalized)
    - Volume (normalized)
    - RSI
    - MACD
    - Bollinger Band position
    - Price change velocity
    
    Output:
    - Direction: UP (1) / NEUTRAL (0) / DOWN (-1)
    - Confidence: 0-100%
    """
    
    def __init__(self, lookback_hours: int = 24, predict_hours: int = 2, symbol: str = 'BTCUSDT'):
        self.lookback_hours = lookback_hours
        self.predict_hours = predict_hours
        self.model = None
        self.scaler = MinMaxScaler() if TF_AVAILABLE else None
        self.feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'velocity']
        self.trained = False
        self.last_train = None
        self.symbol = symbol
        
        # Auto-load trained model from storage
        if TF_AVAILABLE:
            self._auto_load_model()
    
    def _auto_load_model(self):
        """Auto-load pre-trained model from storage."""
        import os
        import joblib
        
        # Try v12 first (newly trained with current TensorFlow)
        model_path_v12 = f"src/brain/models/storage/lstm_v12_{self.symbol}.h5"
        model_path_v11 = f"src/brain/models/storage/lstm_v11_{self.symbol}.h5"
        scaler_path = f"src/brain/models/storage/scaler_{self.symbol}.pkl"
        
        model_path = model_path_v12 if os.path.exists(model_path_v12) else model_path_v11
        
        try:
            if os.path.exists(model_path):
                # Try loading with compile=False to avoid optimizer issues
                try:
                    self.model = load_model(model_path, compile=False)
                except Exception as e1:
                    # Fallback: try without compile argument
                    try:
                        self.model = load_model(model_path)
                    except Exception as e2:
                        logger.warning(f"Model load failed with both methods: {e1}, {e2}")
                        self.model = None
                        return
                
                if self.model:
                    logger.info(f"LSTM model loaded: {model_path}")
                    self.trained = True
                
                if os.path.exists(scaler_path):
                    try:
                        self.scaler = joblib.load(scaler_path)
                        logger.info(f"Scaler loaded: {scaler_path}")
                    except Exception as e:
                        logger.warning(f"Scaler load failed: {e}")
                        self.scaler = MinMaxScaler()
                
                self.trained = True
            else:
                logger.warning(f"⚠️ No trained model found at {model_path}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
    
    def build_model(self, input_shape: tuple):
        """Build LSTM model with specified input shape."""
        if not TF_AVAILABLE:
            logger.warning("Cannot build model - TensorFlow not available")
            return
            
        self.model = Sequential([
            # First LSTM layer with return sequences
            LSTM(64, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense layers
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            # Output layer - 3 classes (UP, NEUTRAL, DOWN)
            Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"LSTM model built: {input_shape} input shape")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LSTM input."""
        features = pd.DataFrame()
        
        # Price
        features['close'] = df['close']
        
        # Volume
        features['volume'] = df['volume'] if 'volume' in df else 0
        
        # RSI (14-period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        
        # Bollinger Band position (-1 to 1)
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        features['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10) * 2 - 1
        
        # Velocity (rate of change)
        features['velocity'] = df['close'].pct_change(periods=4) * 100
        
        return features.dropna()
    
    def create_label(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """Create labels: UP (1), NEUTRAL (0), DOWN (-1)."""
        future_pct = df['close'].shift(-self.predict_hours).pct_change(periods=self.predict_hours) * 100
        
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_pct > threshold] = 1   # UP
        labels[future_pct < -threshold] = -1  # DOWN
        labels[(future_pct >= -threshold) & (future_pct <= threshold)] = 0  # NEUTRAL
        
        return labels
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(self.lookback_hours, len(features) - self.predict_hours):
            X.append(features[i - self.lookback_hours:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, price_data: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model on historical data.
        
        Args:
            price_data: DataFrame with 'close', 'volume' columns (hourly data)
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Training metrics
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        if len(price_data) < self.lookback_hours * 3:
            return {'error': 'Insufficient data'}
        
        try:
            # Prepare features
            features = self.prepare_features(price_data)
            labels = self.create_label(price_data.loc[features.index])
            
            # Align and drop NaN
            common_idx = features.dropna().index.intersection(labels.dropna().index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features.values)
            
            # Convert labels to one-hot
            label_values = labels.values
            label_values[labels == -1] = 2  # Convert -1 to 2 for one-hot
            labels_onehot = tf.keras.utils.to_categorical(label_values, 3)
            
            # Create sequences
            X, y = self.create_sequences(scaled_features, labels_onehot)
            
            if len(X) < 100:
                return {'error': 'Not enough sequences for training'}
            
            # Build model
            if self.model is None:
                self.build_model((self.lookback_hours, len(self.feature_columns)))
            
            # Train with early stopping
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            self.trained = True
            self.last_train = datetime.now()
            
            return {
                'success': True,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'epochs_trained': len(history.history['loss']),
                'samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'error': str(e)}
    
    def predict(self, recent_data: pd.DataFrame) -> Dict:
        """
        Predict next 1-2 hours direction.
        
        Args:
            recent_data: Last 24+ hours of price data
        
        Returns:
            {
                'direction': 'UP' / 'DOWN' / 'NEUTRAL',
                'confidence': 0-100,
                'probabilities': {'UP': X, 'NEUTRAL': Y, 'DOWN': Z}
            }
        """
        # NO FALLBACK - Real data only
        if not TF_AVAILABLE:
            return self._no_model_response("TensorFlow not available")
        
        if not self.trained or self.model is None:
            return self._no_model_response("Model not trained/loaded")
        
        try:
            # Prepare features
            features = self.prepare_features(recent_data)
            
            if len(features) < self.lookback_hours:
                return self._no_model_response("Insufficient data for prediction")
            
            # Scale and create sequence
            scaled = self.scaler.transform(features.values[-self.lookback_hours:])
            X = np.array([scaled])
            
            # Predict
            probs = self.model.predict(X, verbose=0)[0]
            
            # Interpret
            directions = ['UP', 'NEUTRAL', 'DOWN']
            max_idx = np.argmax(probs)
            direction = directions[max_idx]
            confidence = probs[max_idx] * 100
            
            return {
                'direction': direction,
                'confidence': round(confidence, 1),
                'probabilities': {
                    'UP': round(probs[0] * 100, 1),
                    'NEUTRAL': round(probs[1] * 100, 1),
                    'DOWN': round(probs[2] * 100, 1)
                },
                'model': 'LSTM_TRAINED',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._no_model_response(f"Prediction failed: {e}")
    
    def _no_model_response(self, reason: str) -> Dict:
        """Return NO SIGNAL when model unavailable - NO FALLBACK DATA."""
        logger.warning(f"LSTM NO SIGNAL: {reason}")
        return {
            'direction': 'NEUTRAL',
            'confidence': 0,  # ZERO confidence = no signal
            'probabilities': {'UP': 0, 'NEUTRAL': 0, 'DOWN': 0},
            'model': 'NO_MODEL',
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    def save_model(self, path: str) -> bool:
        """Save trained model."""
        if self.model is None:
            return False
        try:
            self.model.save(path)
            return True
        except:
            return False
    
    def load_model(self, path: str) -> bool:
        """Load pre-trained model."""
        if not TF_AVAILABLE:
            return False
        try:
            self.model = load_model(path)
            self.trained = True
            return True
        except:
            return False


# Convenience functions
def predict_direction(recent_prices: pd.DataFrame) -> Dict:
    """Quick direction prediction."""
    predictor = LSTMTrendPredictor()
    return predictor.predict(recent_prices)


def get_lstm_signal(recent_prices: pd.DataFrame) -> str:
    """Get signal: UP/DOWN/NEUTRAL."""
    result = predict_direction(recent_prices)
    return result['direction']
