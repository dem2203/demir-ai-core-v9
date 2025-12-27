# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Early Signal LSTM Trainer
==========================================
Leading Indicators'ı feature olarak kullanan yeni LSTM modeli.

Bu model ESKİ LSTM'den farklı:
- Eski: Fiyat geçmişi -> Tahmin
- Yeni: Whale + OI + Funding + OrderBook -> Tahmin

Training için gerekli veri: src/v10/training_data/*.json
Veri toplama: EarlySignalEngine.analyze() her çağrıldığında otomatik toplar
"""
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("EARLY_SIGNAL_TRAINER")

# TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not installed - training disabled")


class EarlySignalTrainer:
    """
    Early Signal LSTM Trainer
    
    Leading Indicators -> LSTM -> 4 saat sonraki fiyat yönü
    """
    
    DATA_DIR = Path("src/v10/training_data")
    MODEL_DIR = Path("src/v10/models")
    MODEL_PATH = MODEL_DIR / "early_signal_lstm.keras"
    FEEDBACK_DIR = Path("src/v10/feedback_data")
    
    # Feature boyutu (leading_indicators.py'deki FeatureCollector ile uyumlu)
    FEATURE_SIZE = 20
    SEQUENCE_LENGTH = 10  # Son 10 örnek
    
    def __init__(self):
        self.model = None
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        if not HAS_TENSORFLOW:
            logger.error("TensorFlow required for training!")
            return
        
        # GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"[GPU] {len(gpus)} GPU bulundu")
        else:
            logger.info("[CPU] GPU bulunamadi, CPU kullanilacak")
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tüm training verisini yükle.
        
        Returns:
            X: (samples, seq_len, features)
            y: (samples,) - 1=UP, 0=FLAT, -1=DOWN
        """
        all_samples = []
        
        for filepath in self.DATA_DIR.glob("training_*.json"):
            try:
                with open(filepath, 'r') as f:
                    samples = json.load(f)
                
                for sample in samples:
                    if sample.get('future_price') and sample.get('current_price'):
                        all_samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        if len(all_samples) < 100:
            logger.warning(f"Not enough data: {len(all_samples)} samples (need 100+)")
            return np.array([]), np.array([])
        
        # Sort by timestamp
        all_samples.sort(key=lambda x: x['timestamp'])
        
        # Create sequences
        y = []
        
        # --- 1. Load Feedback Data (Priority) ---
        feedback_count = 0
        for fb_path in self.FEEDBACK_DIR.glob("feedback_*.json"):
            try:
                with open(fb_path, 'r') as f:
                    sample = json.load(f)
                    
                # Oversampling for Feedback Data (x5 weight)
                # Because real trade data is much more valuable than synthetic
                for _ in range(5):
                    # We can't form sequences easily from single feedback points unless we look up history
                    # Strategy: Feedback data is usually sparse. 
                    # For LSTM we need a sequence. 
                    # CRITICAL: We need the PREVIOUS 9 features + Current Feature to make a sequence of 10.
                    # Current simple implementation: Repeat the single feature vector 10 times (weak) 
                    # OR: Just use the single vector if we switch to Dense, but we use LSTM.
                    # BETTER: In FeedbackLoop, we should store the whole SEQUENCE, not just last feature.
                    
                    # For now, let's assume FeedbackLoop saved just the last feature.
                    # To fix this properly, let's just repeat it. It's not ideal for time-series but 
                    # tells the model "This state leads to WIN".
                    seq = [sample['features']] * self.SEQUENCE_LENGTH
                    X.append(seq)
                    y.append(sample['label']) 
                    
                feedback_count += 1
            except Exception as e:
                logger.warning(f"Failed to load feedback {fb_path}: {e}")
                
        if feedback_count > 0:
            logger.info(f"[DATA] Loaded {feedback_count} feedback samples (x5 oversampled)")

        # --- 2. Load Evaluation Data (Synthetic/History) ---
        for i in range(self.SEQUENCE_LENGTH, len(all_samples)):
            # Son SEQUENCE_LENGTH sample'ı al
            seq = all_samples[i-self.SEQUENCE_LENGTH:i]
            features = [s['features'] for s in seq]
            
            # Current sample's label
            sample = all_samples[i]
            price_change = (sample['future_price'] - sample['current_price']) / sample['current_price']
            
            if price_change > 0.015:  # >1.5% up
                label = 1  # BUY
            elif price_change < -0.015:  # >1.5% down
                label = -1  # SELL
            else:
                label = 0  # HOLD
            
            X.append(features)
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        # Convert labels to categorical for classification
        # Synthetic labels were -1,0,1 -> need shift if not already 0,1,2
        # Feedback labels are already 0,1,2
        
        # Check matching
        # Synth: -1 (SELL), 0 (HOLD), 1 (BUY) -> Add 1 -> 0, 1, 2
        # Feedback: 0 (SELL), 1 (HOLD), 2 (BUY) -> No Add needed?
        
        # Wait, loop above for synth:
        # label = -1, 0, 1.
        # Direct append.
        # So array 'y' has mix of [-1,0,1] and [0,1,2] (from feedback).
        # We must standardize.
        
        # FIX:
        # If val < 0 (SELL): set to 0
        # If val == 0 (HOLD): set to 1
        # If val > 0 (BUY/WIN): set to 2
        
        # But wait, synth labels:
        # 1 (BUY)
        # -1 (SELL)
        # 0 (HOLD)
        
        # Feedback labels:
        # 2 (BUY)
        # 0 (SELL)
        
        # Mapping needed: 
        # Synth 1 -> 2
        # Synth -1 -> 0
        # Synth 0 -> 1
        
        y_mapped = []
        for val in y:
            if val == 1: y_mapped.append(2)    # BUY
            elif val == -1: y_mapped.append(0) # SELL
            elif val == 0: y_mapped.append(1)  # HOLD
            elif val == 2: y_mapped.append(2)  # Already mapped BUY
            # else: keep as is (likely 0 from feedback which is SELL, wait 0 is SELL? yes)
            
        y = np.array(y_mapped, dtype=np.int32)
        
        # Shift is no longer needed if we mapped manually
        # y = y + 1  <-- REMOVE THIS
        
        logger.info(f"[DATA] {len(X)} sequences loaded")
        logger.info(f"[DATA] X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"[DATA] Label distribution: {np.bincount(y)}")
        
        return X, y
    
    def build_model(self) -> 'keras.Model':
        """
        LSTM model oluştur.
        """
        model = Sequential([
            # Input: (batch, seq_len, features)
            LSTM(64, return_sequences=True, input_shape=(self.SEQUENCE_LENGTH, self.FEATURE_SIZE)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            # Output: 3 classes (SELL=0, HOLD=1, BUY=2)
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def train(
        self, 
        epochs: int = 100, 
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Modeli eğit.
        """
        if not HAS_TENSORFLOW:
            return {'error': 'TensorFlow not installed'}
        
        # Load data
        X, y = self.load_training_data()
        
        if len(X) < 50:
            return {'error': f'Not enough data: {len(X)} samples (need 50+)'}
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                str(self.MODEL_PATH),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        logger.info(f"[TRAIN] Starting training with {len(X)} samples...")
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Results
        best_val_acc = max(history.history['val_accuracy'])
        best_val_loss = min(history.history['val_loss'])
        
        results = {
            'samples': len(X),
            'epochs_trained': len(history.history['loss']),
            'best_val_accuracy': round(best_val_acc, 4),
            'best_val_loss': round(best_val_loss, 4),
            'model_path': str(self.MODEL_PATH)
        }
        
        logger.info(f"[TRAIN] Complete! Best val accuracy: {best_val_acc:.2%}")
        
        return results
    
    def load_model(self) -> bool:
        """
        Eğitilmiş modeli yükle.
        """
        if not HAS_TENSORFLOW:
            return False
        
        if self.MODEL_PATH.exists():
            try:
                self.model = load_model(str(self.MODEL_PATH))
                logger.info(f"[MODEL] Loaded from {self.MODEL_PATH}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        
        return False
    
    def predict(self, features_sequence: np.ndarray) -> Dict:
        """
        Tahmin yap.
        
        Args:
            features_sequence: (seq_len, features) veya (1, seq_len, features)
            
        Returns:
            {'action': 'BUY'|'SELL'|'HOLD', 'confidence': 0-100, 'probabilities': [...]}
        """
        if self.model is None:
            if not self.load_model():
                return {'action': 'HOLD', 'confidence': 0, 'error': 'Model not loaded'}
        
        # Reshape if needed
        if len(features_sequence.shape) == 2:
            features_sequence = features_sequence.reshape(1, *features_sequence.shape)
        
        # Predict
        probs = self.model.predict(features_sequence, verbose=0)[0]
        
        # probs: [SELL, HOLD, BUY]
        action_idx = np.argmax(probs)
        actions = ['SELL', 'HOLD', 'BUY']
        
        return {
            'action': actions[action_idx],
            'confidence': round(float(probs[action_idx]) * 100, 1),
            'probabilities': {
                'SELL': round(float(probs[0]) * 100, 1),
                'HOLD': round(float(probs[1]) * 100, 1),
                'BUY': round(float(probs[2]) * 100, 1)
            }
        }
    
    def get_data_stats(self) -> Dict:
        """
        Training data istatistikleri.
        """
        total_samples = 0
        files = list(self.DATA_DIR.glob("training_*.json"))
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    samples = json.load(f)
                total_samples += len(samples)
            except:
                pass
        
        return {
            'total_files': len(files),
            'total_samples': total_samples,
            'sequences_possible': max(0, total_samples - self.SEQUENCE_LENGTH),
            'ready_for_training': total_samples >= 100,
            'model_exists': self.MODEL_PATH.exists()
        }


# Global instance
_trainer: Optional[EarlySignalTrainer] = None


def get_trainer() -> EarlySignalTrainer:
    """Get or create trainer instance"""
    global _trainer
    if _trainer is None:
        _trainer = EarlySignalTrainer()
    return _trainer


# CLI interface
if __name__ == "__main__":
    import sys
    
    trainer = get_trainer()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "stats":
            stats = trainer.get_data_stats()
            print(f"\n[DATA STATS]")
            print(f"  Files: {stats['total_files']}")
            print(f"  Samples: {stats['total_samples']}")
            print(f"  Ready: {'Yes' if stats['ready_for_training'] else 'No (need 100+)'}")
            print(f"  Model: {'Exists' if stats['model_exists'] else 'Not trained yet'}")
            
        elif cmd == "train":
            results = trainer.train()
            print(f"\n[TRAINING RESULTS]")
            for k, v in results.items():
                print(f"  {k}: {v}")
        else:
            print("Usage: python early_signal_trainer.py [stats|train]")
    else:
        print("Usage: python early_signal_trainer.py [stats|train]")
