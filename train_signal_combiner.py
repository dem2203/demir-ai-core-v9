# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Combiner Training Script
Geçmiş verilerle Signal Combiner modelini eğitir.

Bu script:
1. Binance'den geçmiş fiyat verilerini çeker
2. Her nokta için feature vektörü oluşturur
3. Gerçek getirileri hesaplar
4. Modeli eğitir ve kaydeder
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.brain.signal_combiner import SignalCombinerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_historical_klines(symbol: str = 'BTCUSDT', interval: str = '4h', limit: int = 1000):
    """Binance'den geçmiş mum verilerini çek"""
    logger.info(f"Fetching {limit} {interval} candles for {symbol}...")
    
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to proper types
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        logger.info(f"Fetched {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    else:
        logger.error(f"Failed to fetch data: {response.status_code}")
        return None


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Teknik indikatörleri hesapla"""
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Volume SMA
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price change
    df['return_1'] = df['close'].pct_change(1)
    df['return_5'] = df['close'].pct_change(5)
    df['return_20'] = df['close'].pct_change(20)
    
    # Volatility
    df['volatility'] = df['return_1'].rolling(window=20).std() * 100
    
    return df


def generate_synthetic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her zaman noktası için sinyal feature'ları oluştur.
    
    Gerçek dünyada bunlar web scraping'den gelir, burada teknik
    indikatörlerden simüle ediyoruz.
    """
    
    # 1. Fear & Greed simülasyonu (RSI bazlı)
    # RSI düşük = Fear, RSI yüksek = Greed
    df['fear_greed'] = df['rsi'].fillna(50)
    
    # 2. TradingView simülasyonu (MACD + SMA cross bazlı)
    df['tradingview'] = 0.0
    df.loc[df['macd'] > df['macd_signal'], 'tradingview'] = 0.5
    df.loc[df['macd'] < df['macd_signal'], 'tradingview'] = -0.5
    df.loc[df['close'] > df['sma_50'], 'tradingview'] += 0.3
    df.loc[df['close'] < df['sma_50'], 'tradingview'] -= 0.3
    df['tradingview'] = df['tradingview'].clip(-1, 1)
    
    # 3. Stablecoin Flow simülasyonu (Volume spike bazlı)
    df['stablecoin_flow'] = (df['volume_ratio'] - 1) * 500_000_000  # Normalize to USD
    
    # 4. DeFi TVL Change simülasyonu (Volatility inverse)
    df['defi_tvl_change'] = -df['volatility'] + 2  # Less volatility = stable TVL
    
    # 5. CME Gap simülasyonu (Weekend returns)
    df['cme_gap'] = 0.0
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    # Monday (0) - Calculate gap from Friday
    monday_mask = df['day_of_week'] == 0
    df.loc[monday_mask, 'cme_gap'] = df['return_5'] * 100  # Approximate gap
    
    # 6. News Sentiment simülasyonu (Momentum bazlı)
    df['news_sentiment'] = 50 + (df['return_5'] * 500).clip(-30, 30)
    
    # 7. Bullish Patterns (Price above key MAs)
    df['bullish_patterns'] = 0
    df.loc[df['close'] > df['sma_20'], 'bullish_patterns'] += 1
    df.loc[df['close'] > df['sma_50'], 'bullish_patterns'] += 1
    df.loc[df['macd'] > 0, 'bullish_patterns'] += 1
    df.loc[df['rsi'] < 40, 'bullish_patterns'] += 1  # Oversold = bullish
    
    # 8. Bearish Patterns
    df['bearish_patterns'] = 0
    df.loc[df['close'] < df['sma_20'], 'bearish_patterns'] += 1
    df.loc[df['close'] < df['sma_50'], 'bearish_patterns'] += 1
    df.loc[df['macd'] < 0, 'bearish_patterns'] += 1
    df.loc[df['rsi'] > 70, 'bearish_patterns'] += 1  # Overbought = bearish
    
    # 9. Funding Rate simülasyonu (RSI extreme bazlı)
    df['funding_rate'] = (df['rsi'] - 50) / 1000  # -0.05 to +0.05
    
    # 10. OI Velocity simülasyonu (Volume change)
    df['oi_velocity'] = df['volume_ratio'].pct_change(5) * 100
    
    # 11. Whale Ratio simülasyonu (Large volume candles)
    df['whale_ratio'] = 0.5  # Default
    df.loc[df['volume_ratio'] > 2, 'whale_ratio'] = 0.7  # High volume = whale activity
    df.loc[df['volume_ratio'] < 0.5, 'whale_ratio'] = 0.3
    
    # === PHASE 42 FEATURES ===
    
    # 12. Liquidation Risk simülasyonu (High OI + Price extremes)
    # When price is near highs/lows + high volume = cascade risk
    df['price_norm'] = (df['close'] - df['close'].rolling(50).min()) / (df['close'].rolling(50).max() - df['close'].rolling(50).min())
    df['liquidation_risk'] = 0.0
    # High risk when price near bottom (0-0.2) or top (0.8-1.0) with high volume
    df.loc[(df['price_norm'] < 0.2) & (df['volume_ratio'] > 1.5), 'liquidation_risk'] = 0.7
    df.loc[(df['price_norm'] > 0.8) & (df['volume_ratio'] > 1.5), 'liquidation_risk'] = 0.6
    
    # 13. Whale Flow simülasyonu (Volume direction)
    # Positive = outflow (bullish), Negative = inflow (bearish)
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['whale_flow'] = (df['volume'] - df['volume_ma']) * 50_000_000  # Normalize to USD
    # Bias based on price action
    df.loc[df['close'] > df['open'], 'whale_flow'] = df['whale_flow'].abs()  # Green candle = outflow
    df.loc[df['close'] < df['open'], 'whale_flow'] = -df['whale_flow'].abs()  # Red candle = inflow
    
    # 14. Reddit Sentiment simülasyonu (Community mood based on returns)
    # Positive returns = bullish sentiment
    df['reddit_sentiment'] = 50 + (df['return_5'] * 1000).clip(-45, 45)
    
    return df


def calculate_future_returns(df: pd.DataFrame, lookahead: int = 6) -> pd.DataFrame:
    """
    Gelecek getiriyi hesapla (eğitim hedefi).
    
    lookahead: Kaç bar sonraki getiri
    """
    df['future_close'] = df['close'].shift(-lookahead)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    # Normalize to -1 to +1 (5% = max)
    df['target'] = (df['future_return'] * 20).clip(-1, 1)
    
    return df


def prepare_training_data(df: pd.DataFrame) -> list:
    """Training data formatına dönüştür"""
    
    # Fill NaN values for Phase 42 features
    df['liquidation_risk'] = df['liquidation_risk'].fillna(0)
    df['whale_flow'] = df['whale_flow'].fillna(0)
    df['reddit_sentiment'] = df['reddit_sentiment'].fillna(50)
    df['price_norm'] = df['price_norm'].fillna(0.5)
    
    training_data = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['target']) or pd.isna(row['rsi']):
            continue
        
        features = {
            'fear_greed': row['fear_greed'],
            'tradingview': row['tradingview'],
            'stablecoin_flow': row['stablecoin_flow'],
            'defi_tvl_change': row['defi_tvl_change'],
            'cme_gap': row['cme_gap'],
            'news_sentiment': row['news_sentiment'],
            'bullish_patterns': row['bullish_patterns'],
            'bearish_patterns': row['bearish_patterns'],
            'funding_rate': row['funding_rate'],
            'oi_velocity': row['oi_velocity'] if not pd.isna(row['oi_velocity']) else 0,
            'whale_ratio': row['whale_ratio'],
            'rsi': row['rsi'],
            # Phase 42 features
            'liquidation_risk': row.get('liquidation_risk', 0),
            'whale_flow': row.get('whale_flow', 0),
            'reddit_sentiment': row.get('reddit_sentiment', 50)
        }
        
        target = row['target']
        training_data.append((features, target))
    
    return training_data


def main():
    """Ana eğitim fonksiyonu"""
    
    print("=" * 60)
    print("DEMIR AI - Signal Combiner Model Training (Phase 44)")
    print("   15 Features (12 original + 3 Phase 42)")
    print("=" * 60)
    
    # 1. Fetch historical data
    print("\n📊 Step 1: Fetching historical data from Binance...")
    df = fetch_historical_klines('BTCUSDT', '4h', 1000)
    
    if df is None:
        print("❌ Failed to fetch data!")
        return
    
    # 2. Calculate technical indicators
    print("\n📈 Step 2: Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    # 3. Generate synthetic signals
    print("\n🔮 Step 3: Generating signal features...")
    df = generate_synthetic_signals(df)
    
    # 4. Calculate future returns (target)
    print("\n🎯 Step 4: Calculating target returns (6 bars ahead)...")
    df = calculate_future_returns(df, lookahead=6)
    
    # 5. Prepare training data
    print("\n📦 Step 5: Preparing training data...")
    training_data = prepare_training_data(df)
    print(f"   Total samples: {len(training_data)}")
    
    # 6. Train the model
    print("\n🏋️ Step 6: Training Signal Combiner Model...")
    combiner = SignalCombinerModel()
    
    try:
        result = combiner.train(training_data, epochs=100)
        print(f"\n✅ Training complete!")
        print(f"   Train R²: {result.get('train_score', 0):.4f}")
        print(f"   Test R²: {result.get('test_score', 0):.4f}")
    except Exception as e:
        print(f"   sklearn not available, using weight optimization...")
        result = combiner._train_weights(training_data, epochs=200)
        print(f"\n✅ Weight optimization complete!")
        print(f"   MSE: {result.get('mse', 0):.6f}")
    
    # 7. Test prediction
    print("\n🧪 Step 7: Testing prediction...")
    
    # Get latest data point
    latest = training_data[-1][0]
    signal = combiner.predict(latest)
    
    print(f"\n   Latest Signal:")
    print(f"   Action: {signal.action}")
    print(f"   Confidence: {signal.confidence:.1f}%")
    print(f"   Raw Score: {signal.raw_score:.3f}")
    print(f"   Reasoning: {signal.reasoning}")
    
    # 8. Show learned weights
    print("\n📊 Learned Feature Weights:")
    sorted_weights = sorted(combiner.weights.items(), key=lambda x: x[1], reverse=True)
    for name, weight in sorted_weights:
        bar = "█" * int(weight * 50)
        print(f"   {name:20s}: {bar} {weight:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Model saved to: src/brain/models/storage/signal_combiner_v1.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
