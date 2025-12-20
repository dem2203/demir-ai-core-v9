# -*- coding: utf-8 -*-
"""
DEMIR AI - Auto Training Pipeline
==================================
LSTM ve RL modellerini otomatik eğiten sistem.
Railway'de background task olarak çalışır.

Features:
1. LSTM model eğitimi (1000+ mum verisi ile)
2. Periyodik yeniden eğitim (her 24 saat)
3. Model versiyonlama
4. Performance tracking
"""
import logging
import asyncio
import numpy as np
import pandas as pd
import requests
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("AUTO_TRAINER")

# Paths
MODEL_DIR = Path("src/brain/models/storage")
TRAINING_LOG = MODEL_DIR / "training_log.json"


class AutoTrainer:
    """
    Otomatik Model Eğitim Sistemi
    
    - LSTM: 1000+ saatlik veri ile eğitir
    - RL: Simülasyon ortamında eğitir
    - Her 24 saatte bir günceller
    - Telegram bildirimi gönderir
    """
    
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT']
    TRAINING_INTERVAL = 86400  # 24 saat
    MIN_CANDLES = 500  # Minimum eğitim verisi
    
    def __init__(self, notify_callback=None):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.training_log = self._load_training_log()
        self.is_training = False
        self.notify_callback = notify_callback  # Telegram notification function
    
    def _load_training_log(self) -> Dict:
        """Eğitim logunu yükle."""
        if TRAINING_LOG.exists():
            try:
                with open(TRAINING_LOG, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'last_train': {}, 'models': {}, 'performance': {}}
    
    def _save_training_log(self):
        """Eğitim logunu kaydet."""
        try:
            with open(TRAINING_LOG, 'w') as f:
                json.dump(self.training_log, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Training log save failed: {e}")
    
    async def run_training_loop(self):
        """
        Sürekli eğitim döngüsü.
        Railway'de background task olarak çalışır.
        """
        logger.info("🎓 Auto Trainer started")
        
        while True:
            try:
                for symbol in self.SYMBOLS:
                    if self._should_train(symbol):
                        await self._train_lstm(symbol)
                
                # RL eğitimi (sadece BTC için)
                if self._should_train_rl():
                    await self._train_rl('BTCUSDT')
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
            
            # 1 saat bekle, sonra tekrar kontrol et
            await asyncio.sleep(3600)
    
    def _should_train(self, symbol: str) -> bool:
        """Bu coin için eğitim gerekli mi?"""
        last_train = self.training_log.get('last_train', {}).get(symbol)
        
        if not last_train:
            return True
        
        try:
            last_dt = datetime.fromisoformat(last_train)
            elapsed = (datetime.now() - last_dt).total_seconds()
            return elapsed >= self.TRAINING_INTERVAL
        except:
            return True
    
    def _should_train_rl(self) -> bool:
        """RL eğitimi gerekli mi?"""
        last_train = self.training_log.get('last_train', {}).get('rl_agent')
        
        if not last_train:
            return True
        
        try:
            last_dt = datetime.fromisoformat(last_train)
            elapsed = (datetime.now() - last_dt).total_seconds()
            return elapsed >= self.TRAINING_INTERVAL * 7  # Haftada bir
        except:
            return True
    
    async def _train_lstm(self, symbol: str):
        """LSTM modelini eğit."""
        if self.is_training:
            logger.info(f"Training already in progress, skipping {symbol}")
            return
        
        self.is_training = True
        logger.info(f"🧠 Starting LSTM training for {symbol}...")
        
        try:
            # 1. Veri topla (1000+ mum)
            df = await self._fetch_training_data(symbol, limit=1000)
            
            if df is None or len(df) < self.MIN_CANDLES:
                logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0}")
                return
            
            # 2. Model yükle veya oluştur
            from src.brain.models.lstm_trend import LSTMTrendPredictor
            
            model = LSTMTrendPredictor(symbol=symbol)
            
            # 3. Eğit
            result = model.train(df, epochs=30, batch_size=32)
            
            if result.get('success'):
                # 4. Modeli kaydet
                model_path = MODEL_DIR / f"lstm_v12_{symbol}.h5"
                model.save_model(str(model_path))
                
                # Scaler'ı da kaydet
                import joblib
                scaler_path = MODEL_DIR / f"scaler_{symbol}.pkl"
                joblib.dump(model.scaler, scaler_path)
                
                # 5. Log güncelle
                self.training_log['last_train'][symbol] = datetime.now().isoformat()
                self.training_log['models'][symbol] = {
                    'version': 'v12',
                    'accuracy': result.get('final_accuracy', 0),
                    'loss': result.get('final_loss', 0),
                    'samples': result.get('samples', 0),
                    'trained_at': datetime.now().isoformat()
                }
                self._save_training_log()
                
                logger.info(f"✅ LSTM {symbol} trained: acc={result.get('final_accuracy', 0):.2%}, loss={result.get('final_loss', 0):.4f}")
                
                # Telegram Notification
                if self.notify_callback:
                    try:
                        msg = f"🧠 **MODEL UPDATED**\n" \
                              f"• Coin: {symbol}\n" \
                              f"• Accuracy: {result.get('final_accuracy', 0):.1%}\n" \
                              f"• Loss: {result.get('final_loss', 0):.4f}\n" \
                              f"• Samples: {result.get('samples', 0)}\n" \
                              f"• Next Train: 24 hours"
                        asyncio.create_task(self.notify_callback(msg))
                    except Exception as notify_err:
                        logger.debug(f"Notification failed: {notify_err}")
            else:
                logger.warning(f"LSTM training failed for {symbol}: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"LSTM training error for {symbol}: {e}")
        finally:
            self.is_training = False
    
    async def _train_rl(self, symbol: str):
        """RL agent'ı eğit."""
        logger.info(f"🎮 Starting RL training for {symbol}...")
        
        try:
            # 1. Eğitim verisi topla
            df = await self._fetch_training_data(symbol, limit=2000)
            
            if df is None or len(df) < 500:
                logger.warning("Insufficient data for RL training")
                return
            
            # 2. Feature engineering
            features = self._prepare_rl_features(df)
            
            if features is None:
                return
            
            # 3. Environment oluştur
            from src.brain.rl_agent.trading_env import TradingEnv
            from src.brain.rl_agent.ppo_agent import RLAgent
            
            env = TradingEnv(data=features)
            agent = RLAgent()
            agent.create_model(env)
            
            # 4. Eğit (10k step - kısa tutuyoruz, Railway timeout için)
            agent.train(total_timesteps=10000)
            
            # 5. Kaydet
            agent.save("ppo_btcusdt_v2")
            
            # 6. Log güncelle
            self.training_log['last_train']['rl_agent'] = datetime.now().isoformat()
            self.training_log['models']['rl_agent'] = {
                'version': 'v2',
                'timesteps': 10000,
                'trained_at': datetime.now().isoformat()
            }
            self._save_training_log()
            
            logger.info(f"✅ RL Agent trained and saved")
            
        except Exception as e:
            logger.error(f"RL training error: {e}")
    
    async def _fetch_training_data(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Binance'dan eğitim verisi çek."""
        try:
            # Binance max 1000 per request, need multiple calls
            all_klines = []
            remaining = limit
            end_time = None
            
            while remaining > 0:
                batch_size = min(remaining, 1000)
                params = {
                    'symbol': symbol,
                    'interval': '1h',
                    'limit': batch_size
                }
                if end_time:
                    params['endTime'] = end_time - 1
                
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params=params,
                    timeout=30
                )
                
                if resp.status_code != 200:
                    break
                
                klines = resp.json()
                if not klines:
                    break
                
                all_klines = klines + all_klines  # Prepend older data
                remaining -= len(klines)
                end_time = klines[0][0]  # Oldest timestamp
                
                await asyncio.sleep(0.5)  # Rate limit
            
            if not all_klines:
                return None
            
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None
    
    def _prepare_rl_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """RL için feature matrix hazırla (37 boyut)."""
        try:
            from src.brain.state_builder import StateVectorBuilder
            
            builder = StateVectorBuilder()
            
            # Her satır için state vector oluştur
            features = []
            
            for i in range(50, len(df)):
                window = df.iloc[i-50:i]
                
                # Basit feature extraction
                close = window['close'].values
                volume = window['volume'].values
                
                # 37 feature
                state = np.zeros(37)
                
                # Price features
                state[0] = (close[-1] / close[-2] - 1) * 100  # 1-bar return
                state[1] = (close[-1] / close[-5] - 1) * 100  # 5-bar return
                state[2] = (close[-1] / close[-20] - 1) * 100  # 20-bar return
                
                # Volume features
                avg_vol = np.mean(volume[:-1])
                state[3] = volume[-1] / avg_vol if avg_vol > 0 else 1
                
                # Volatility
                returns = np.diff(close) / close[:-1]
                state[4] = np.std(returns) * 100
                
                # RSI
                delta = np.diff(close)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / (avg_loss + 1e-10)
                state[5] = 100 - (100 / (1 + rs))
                
                # EMA crossovers
                ema9 = self._ema(close, 9)
                ema21 = self._ema(close, 21)
                state[6] = (ema9 / ema21 - 1) * 100
                
                # Bollinger position
                sma = np.mean(close[-20:])
                std = np.std(close[-20:])
                state[7] = (close[-1] - sma) / (2 * std + 1e-10)
                
                # --- Zero-Mock Features (Replaces random noise) ---
                
                # 8-10: MACD
                ema12 = self._ema(close, 12)
                ema26 = self._ema(close, 26)
                macd_line = ema12 - ema26
                signal_line = self._ema(np.array([macd_line]), 9) if len(close) > 9 else macd_line
                state[8] = macd_line
                state[9] = signal_line
                state[10] = macd_line - signal_line
                
                # 11: Momentum (10-bar)
                state[11] = close[-1] - close[-11] if len(close) > 11 else 0
                
                # 12: ROC (Rate of Change)
                state[12] = ((close[-1] - close[-10]) / close[-10]) * 100 if len(close) > 10 else 0
                
                # 13: ATR (Approximate Trur Range) normalized
                high = df['high'].values
                low = df['low'].values
                tr = np.maximum(high[i] - low[i], np.abs(high[i] - close[i-1]))
                atr = np.mean(tr[-14:]) if len(tr) > 14 else tr[-1]
                state[13] = atr / close[-1] * 100
                
                # 14-16: Lagged Returns (t-2, t-3, t-4)
                state[14] = (close[-2] / close[-3] - 1) * 100
                state[15] = (close[-3] / close[-4] - 1) * 100
                state[16] = (close[-4] / close[-5] - 1) * 100
                
                # 17-19: Lagged Volume
                state[17] = volume[-2] / avg_vol if avg_vol > 0 else 1
                state[18] = volume[-3] / avg_vol if avg_vol > 0 else 1
                state[19] = volume[-4] / avg_vol if avg_vol > 0 else 1
                
                # 20: High/Low relative position
                hl_range = high[i] - low[i]
                state[20] = (close[i] - low[i]) / hl_range if hl_range > 0 else 0.5
                
                # Fill remaining with 0 (better than random noise for model stability)
                for j in range(21, 37):
                    state[j] = 0.0
                
                features.append(state)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature prep error: {e}")
            return None
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """EMA hesapla."""
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def get_training_status(self) -> Dict:
        """Eğitim durumunu döndür."""
        return {
            'is_training': self.is_training,
            'last_train': self.training_log.get('last_train', {}),
            'models': self.training_log.get('models', {})
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_trainer: Optional[AutoTrainer] = None

def get_trainer(notify_callback=None) -> AutoTrainer:
    """Get or create trainer instance."""
    global _trainer
    if _trainer is None:
        _trainer = AutoTrainer(notify_callback=notify_callback)
    return _trainer


async def start_training_background(notify_callback=None):
    """Background training task başlat."""
    trainer = get_trainer(notify_callback=notify_callback)
    await trainer.run_training_loop()


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        trainer = get_trainer()
        
        # Test data fetch
        df = await trainer._fetch_training_data('BTCUSDT', limit=100)
        if df is not None:
            print(f"Fetched {len(df)} candles")
            
            # Test LSTM training
            await trainer._train_lstm('BTCUSDT')
    
    asyncio.run(test())
