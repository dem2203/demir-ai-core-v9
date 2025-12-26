# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - LSTM PREDICTOR
==============================
LSTM tabanlı fiyat tahmini modeli.
Binance Futures kline verilerinden otomatik eğitim ve tahmin.

ÖZELLİKLER:
- Online learning: Her 1 saatte yeni veri ile güncelleme
- Multi-feature: OHLCV + RSI + EMA
- 4 saat sonrası fiyat tahmini
- Confidence score

KULLANIM:
    predictor = LSTMPredictor()
    await predictor.train(symbol)
    prediction = await predictor.predict(symbol)
"""
import logging
import numpy as np
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os
import json

logger = logging.getLogger("LSTM_PREDICTOR")


@dataclass
class PricePrediction:
    """LSTM tahmin sonucu"""
    symbol: str
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    direction: str  # "UP", "DOWN", "FLAT"
    confidence: float  # 0-100
    horizon: str  # "4h"
    model_accuracy: float  # Historical accuracy
    timestamp: datetime


class LSTMPredictor:
    """
    Basit LSTM benzeri tahmin modeli.
    TensorFlow/PyTorch gerektirmez - saf NumPy implementasyonu.
    
    Not: Bu basitleştirilmiş bir versiyon. Gerçek LSTM için
    TensorFlow/PyTorch kullanılmalıdır.
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    
    LOOKBACK = 48  # 48 bar (48 saat @ 1h)
    PREDICTION_HORIZON = 4  # 4 bar (4 saat) ilerisi
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: Dict[str, Dict] = {}  # symbol -> weights
        self._accuracy: Dict[str, float] = {}  # symbol -> accuracy
        self._predictions_history: Dict[str, List] = {}  # Geçmiş tahminler (doğrulama için)
        
        # Model storage path
        self._storage_path = "src/v10/models"
        os.makedirs(self._storage_path, exist_ok=True)
        
        logger.info("🧠 LSTM Predictor initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=self.HEADERS
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def train(self, symbol: str) -> Dict:
        """
        Modeli eğit veya güncelle.
        """
        logger.info(f"🎓 Training model for {symbol}...")
        
        try:
            # Veri çek
            klines = await self._fetch_klines(symbol, limit=500)
            if len(klines) < 200:
                return {'error': 'Insufficient data', 'success': False}
            
            # Feature extraction
            X, y = self._prepare_training_data(klines)
            
            # Simple linear regression (LSTM yerine - dependency-free)
            # y = X * weights + bias
            # En küçük kareler ile çözüm
            
            # Normalize
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-8
            X_norm = (X - X_mean) / X_std
            
            y_mean = np.mean(y)
            y_std = np.std(y) + 1e-8
            y_norm = (y - y_mean) / y_std
            
            # Pseudo-inverse solution: weights = (X^T X)^-1 X^T y
            XtX = np.dot(X_norm.T, X_norm)
            XtX_inv = np.linalg.pinv(XtX + 0.01 * np.eye(XtX.shape[0]))  # Regularization
            Xty = np.dot(X_norm.T, y_norm)
            weights = np.dot(XtX_inv, Xty)
            
            # Store model
            self._models[symbol] = {
                'weights': weights.tolist(),
                'X_mean': X_mean.tolist(),
                'X_std': X_std.tolist(),
                'y_mean': float(y_mean),
                'y_std': float(y_std),
                'trained_at': datetime.now().isoformat()
            }
            
            # Calculate training accuracy
            y_pred_norm = np.dot(X_norm, weights)
            y_pred = y_pred_norm * y_std + y_mean
            
            # Direction accuracy
            actual_direction = np.sign(y - klines[self.LOOKBACK:-self.PREDICTION_HORIZON, 4])
            predicted_direction = np.sign(y_pred - klines[self.LOOKBACK:-self.PREDICTION_HORIZON, 4])
            direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            
            self._accuracy[symbol] = direction_accuracy
            
            # Save model
            self._save_model(symbol)
            
            logger.info(f"✅ {symbol} model trained. Direction accuracy: {direction_accuracy:.1f}%")
            
            return {
                'success': True,
                'accuracy': direction_accuracy,
                'samples': len(y)
            }
            
        except Exception as e:
            logger.error(f"❌ Training error {symbol}: {e}")
            return {'error': str(e), 'success': False}
    
    async def predict(self, symbol: str) -> Optional[PricePrediction]:
        """
        Gelecek fiyat tahmini yap.
        """
        try:
            # Ensure model exists
            if symbol not in self._models:
                await self.train(symbol)
            
            if symbol not in self._models:
                return None
            
            model = self._models[symbol]
            
            # Fetch recent data
            klines = await self._fetch_klines(symbol, limit=self.LOOKBACK + 10)
            if len(klines) < self.LOOKBACK:
                return None
            
            current_price = float(klines[-1][4])
            
            # Extract features for prediction
            features = self._extract_features(klines[-self.LOOKBACK:])
            
            # Normalize
            X_mean = np.array(model['X_mean'])
            X_std = np.array(model['X_std'])
            X_norm = (features - X_mean) / X_std
            
            # Predict
            weights = np.array(model['weights'])
            y_pred_norm = np.dot(X_norm, weights)
            predicted_price = y_pred_norm * model['y_std'] + model['y_mean']
            
            # Change percentage
            change_pct = (predicted_price - current_price) / current_price * 100
            
            # Direction
            if change_pct > 0.5:
                direction = "UP"
            elif change_pct < -0.5:
                direction = "DOWN"
            else:
                direction = "FLAT"
            
            # Confidence based on historical accuracy and prediction magnitude
            base_confidence = self._accuracy.get(symbol, 50)
            magnitude_factor = min(1.0, abs(change_pct) / 5)  # Larger moves = more confidence
            confidence = base_confidence * (0.7 + 0.3 * magnitude_factor)
            
            prediction = PricePrediction(
                symbol=symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                predicted_change_pct=change_pct,
                direction=direction,
                confidence=confidence,
                horizon="4h",
                model_accuracy=self._accuracy.get(symbol, 0),
                timestamp=datetime.now()
            )
            
            # Store for validation
            if symbol not in self._predictions_history:
                self._predictions_history[symbol] = []
            self._predictions_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'current': current_price,
                'predicted': predicted_price,
                'direction': direction
            })
            
            # Keep last 100 predictions
            self._predictions_history[symbol] = self._predictions_history[symbol][-100:]
            
            logger.info(f"📈 {symbol} prediction: {direction} {change_pct:+.2f}% (conf: {confidence:.0f}%)")
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction error {symbol}: {e}")
            return None
    
    def _prepare_training_data(self, klines: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        OHLCV + teknik göstergelerden eğitim verisi hazırla.
        """
        arr = np.array(klines, dtype=float)
        
        X_list = []
        y_list = []
        
        for i in range(self.LOOKBACK, len(arr) - self.PREDICTION_HORIZON):
            features = self._extract_features(arr[i-self.LOOKBACK:i])
            target = arr[i + self.PREDICTION_HORIZON, 4]  # Future close price
            
            X_list.append(features)
            y_list.append(target)
        
        return np.array(X_list), np.array(y_list)
    
    def _extract_features(self, klines: np.ndarray) -> np.ndarray:
        """
        Kline verilerinden feature vektörü çıkar.
        """
        closes = klines[:, 4]
        volumes = klines[:, 5]
        highs = klines[:, 2]
        lows = klines[:, 3]
        
        # Price features
        current = closes[-1]
        price_change_1 = (closes[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] > 0 else 0
        price_change_5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] > 0 else 0
        price_change_24 = (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 and closes[-24] > 0 else 0
        
        # RSI
        rsi = self._calc_rsi(closes.tolist())
        
        # EMA
        ema_20 = self._calc_ema(closes.tolist(), 20)
        ema_50 = self._calc_ema(closes.tolist(), 50) if len(closes) >= 50 else ema_20
        
        # Price position relative to EMAs
        price_vs_ema20 = (current - ema_20) / ema_20 * 100 if ema_20 > 0 else 0
        price_vs_ema50 = (current - ema_50) / ema_50 * 100 if ema_50 > 0 else 0
        
        # Volatility
        atr = np.mean([highs[i] - lows[i] for i in range(-14, 0)])
        volatility = atr / current * 100 if current > 0 else 0
        
        # Volume
        vol_avg = np.mean(volumes[-14:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1
        
        # Momentum
        momentum = sum([1 if closes[i] > closes[i-1] else -1 for i in range(-5, 0)])
        
        return np.array([
            price_change_1,
            price_change_5,
            price_change_24,
            rsi,
            price_vs_ema20,
            price_vs_ema50,
            volatility,
            vol_ratio,
            momentum
        ])
    
    def _calc_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calc_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    async def _fetch_klines(self, symbol: str, limit: int = 500) -> np.ndarray:
        """
        Binance Futures'tan kline çek.
        """
        session = await self._get_session()
        
        url = f"{self.FUTURES_BASE}/fapi/v1/klines?symbol={symbol}&interval=1h&limit={limit}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Klines API error: {resp.status}")
            data = await resp.json()
        
        return np.array(data, dtype=float)
    
    def _save_model(self, symbol: str):
        """Modeli dosyaya kaydet."""
        path = os.path.join(self._storage_path, f"{symbol}_model.json")
        with open(path, 'w') as f:
            json.dump(self._models[symbol], f)
    
    def _load_model(self, symbol: str) -> bool:
        """Modeli dosyadan yükle."""
        path = os.path.join(self._storage_path, f"{symbol}_model.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                self._models[symbol] = json.load(f)
            return True
        return False
    
    def get_predictions_history(self, symbol: str) -> List:
        """Geçmiş tahminleri döndür."""
        return self._predictions_history.get(symbol, [])


# Singleton
_lstm_predictor: Optional[LSTMPredictor] = None

def get_lstm_predictor() -> LSTMPredictor:
    global _lstm_predictor
    if _lstm_predictor is None:
        _lstm_predictor = LSTMPredictor()
    return _lstm_predictor
