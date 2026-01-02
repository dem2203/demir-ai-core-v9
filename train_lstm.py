# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - LSTM Model Trainer
=================================
Train the LSTM predictor on historical data.

USAGE:
    python train_lstm.py --symbol BTCUSDT --days 365

OUTPUT:
    - Trained model saved to models/lstm_btcusdt.keras
    - Training metrics logged
"""
import asyncio
import argparse
import sys
import os

# Force UTF-8 on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import aiohttp
from datetime import datetime
from pathlib import Path


async def fetch_historical_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical klines from Binance Futures."""
    print(f"[INFO] Fetching {days} days of {symbol} data...")
    
    all_klines = []
    end_time = int(datetime.now().timestamp() * 1000)
    batch_size = 1000
    
    # 1h candles
    total_candles = days * 24
    
    async with aiohttp.ClientSession() as session:
        while len(all_klines) < total_candles:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': batch_size,
                'endTime': end_time
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    klines = await resp.json()
                    if not klines:
                        break
                    all_klines = klines + all_klines
                    end_time = klines[0][0] - 1
                    print(f"  Loaded {len(all_klines)}/{total_candles} candles...")
                else:
                    print(f"[ERROR] API error: {resp.status}")
                    break
            
            await asyncio.sleep(0.1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    print(f"[SUCCESS] Loaded {len(df)} candles")
    return df


def prepare_lstm_features(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """Prepare features for LSTM training."""
    print("[INFO] Preparing features...")
    
    # Calculate technical indicators
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume MA
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # EMAs
    df['ema9'] = df['close'].ewm(span=9).mean()
    df['ema21'] = df['close'].ewm(span=21).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    # Normalize prices relative to current
    df['price_norm'] = df['close'] / df['close'].rolling(50).mean()
    
    # Drop NaN
    df = df.dropna()
    
    # Feature columns
    feature_cols = [
        'returns', 'rsi', 'macd', 'macd_signal', 'bb_position',
        'volume_ratio', 'price_norm'
    ]
    
    # Create sequences
    X, y = [], []
    data = df[feature_cols].values
    targets = df['returns'].shift(-1).values  # Predict next candle return
    
    for i in range(len(data) - lookback - 1):
        X.append(data[i:i+lookback])
        # Classification: UP (>0.5%), DOWN (<-0.5%), NEUTRAL
        next_return = targets[i + lookback]
        if next_return > 0.005:
            y.append([1, 0, 0])  # UP
        elif next_return < -0.005:
            y.append([0, 1, 0])  # DOWN
        else:
            y.append([0, 0, 1])  # NEUTRAL
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"[SUCCESS] Created {len(X)} sequences with {len(feature_cols)} features")
    return X, y, feature_cols


def build_lstm_model(input_shape: tuple, num_classes: int = 3):
    """Build LSTM model architecture."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
        return None
    
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


async def train_lstm(symbol: str = "BTCUSDT", days: int = 365):
    """Main training function."""
    print("=" * 60)
    print("[TRAIN] DEMIR AI v10 - LSTM Model Training")
    print("=" * 60)
    
    # Fetch data
    df = await fetch_historical_data(symbol, days)
    
    # Prepare features
    X, y, features = prepare_lstm_features(df)
    
    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\n[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Test set: {len(X_test)} samples")
    
    # Build model
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    if model is None:
        return
    
    print("\n[INFO] Model architecture:")
    model.summary()
    
    # Train
    print("\n[INFO] Training...")
    
    try:
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[RESULT] Test Accuracy: {accuracy * 100:.1f}%")
        print(f"[RESULT] Test Loss: {loss:.4f}")
        
        # Save model
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"lstm_{symbol.lower()}.keras"
        model.save(model_path)
        print(f"\n[SUCCESS] Model saved to: {model_path}")
        
        # Also save feature info
        import json
        meta = {
            'symbol': symbol,
            'features': features,
            'lookback': X.shape[1],
            'accuracy': float(accuracy),
            'trained_at': datetime.now().isoformat()
        }
        with open(models_dir / f"lstm_{symbol.lower()}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        print("[SUCCESS] Training complete!")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Train LSTM predictor')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    args = parser.parse_args()
    
    asyncio.run(train_lstm(args.symbol, args.days))


if __name__ == "__main__":
    main()
