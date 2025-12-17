"""
DEMIR AI - Signal Combiner Model
Tüm web scraping ve analiz sinyallerini ML ile birleştirir.

PHASE 39: Intelligent Signal Fusion
- Tüm sinyalleri tek bir confidence score'a dönüştürür
- Hangi sinyalin daha önemli olduğunu öğrenir
- Yanlış pozitif sayısını azaltır

Input Features:
1. Fear & Greed Index (0-100)
2. TradingView Signal (-1 to +1)
3. Stablecoin Flow (normalized)
4. DeFi TVL Change (percent)
5. CME Gap (percent)
6. News Sentiment (0-100)
7. Bullish Pattern Count (0-5)
8. Bearish Pattern Count (0-5)
9. Funding Rate (normalized)
10. OI Velocity (normalized)
11. Whale Buy/Sell Ratio (0-1)
12. RSI (0-100)

Output: Combined Signal (-1 to +1) and Confidence (0-100)
"""
import logging
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("SIGNAL_COMBINER")


@dataclass
class CombinedSignal:
    """Birleştirilmiş sinyal çıktısı"""
    action: str           # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence: float     # 0-100
    raw_score: float      # -1 to +1
    top_bullish: List[str]   # En güçlü bullish sinyaller
    top_bearish: List[str]   # En güçlü bearish sinyaller
    reasoning: str        # Türkçe açıklama
    timestamp: datetime


class SignalCombinerModel:
    """
    Sinyal Birleştirici Model
    
    Tüm web scraping ve analiz sinyallerini ML ile birleştirip
    tek bir tahmin üretir.
    
    İki mod:
    1. Rule-based (başlangıç) - Ağırlıklı ortalama
    2. ML-trained - Gradient Boosting / Neural Network
    """
    
    # Feature names for interpretability
    # UPDATED: Phase 42 - Added liquidation_risk, whale_flow, reddit_sentiment
    FEATURE_NAMES = [
        'fear_greed',
        'tradingview',
        'stablecoin_flow',
        'defi_tvl_change',
        'cme_gap',
        'news_sentiment',
        'bullish_patterns',
        'bearish_patterns',
        'funding_rate',
        'oi_velocity',
        'whale_ratio',
        'rsi',
        'liquidation_risk',  # NEW: Phase 42
        'whale_flow',         # NEW: Phase 42
        'reddit_sentiment'    # NEW: Phase 42
    ]
    
    # Initial weights (will be learned)
    # UPDATED: Phase 42 - Adjusted weights for new features
    DEFAULT_WEIGHTS = {
        'fear_greed': 0.10,       # Inverse - low fear = bullish
        'tradingview': 0.12,      # Direct signal
        'stablecoin_flow': 0.08,  # Mint = bullish
        'defi_tvl_change': 0.06,  # TVL up = bullish
        'cme_gap': 0.06,          # Gap direction
        'news_sentiment': 0.08,   # High = bullish
        'bullish_patterns': 0.10, # Count of bullish patterns
        'bearish_patterns': 0.08, # Count of bearish patterns (inverse)
        'funding_rate': 0.04,     # Extreme = contrarian
        'oi_velocity': 0.03,      # Volatility indicator
        'whale_ratio': 0.04,      # Whale buy ratio
        'rsi': 0.02,              # Overbought/oversold
        #  Phase 42 features (higher weights - critical data!)
        'liquidation_risk': 0.08, # Cascade risk (HIGH impact!)
        'whale_flow': 0.06,       # Exchange flow direction
        'reddit_sentiment': 0.05  # Social sentiment
    }
    
    def __init__(self):
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.model_path = Path("src/brain/models/storage/signal_combiner_v1.pkl")
        
        # Try to load pre-trained model
        self._load_model()
    
    def normalize_inputs(self, raw_data: Dict) -> np.ndarray:
        """
        Ham verileri normalize et (0-1 veya -1 to +1).
        """
        features = []
        
        # 1. Fear & Greed (0-100 → 0-1, inversed: low fear = bullish)
        fear_greed = raw_data.get('fear_greed', 50)
        features.append((100 - fear_greed) / 100)  # Inverse
        
        # 2. TradingView (-1 to +1)
        tv = raw_data.get('tradingview', 0)
        features.append(np.clip(tv, -1, 1))
        
        # 3. Stablecoin Flow (normalized to -1 to +1)
        stablecoin = raw_data.get('stablecoin_flow', 0)
        # $1B mint = +1, $1B burn = -1
        features.append(np.clip(stablecoin / 1e9, -1, 1))
        
        # 4. DeFi TVL Change (percent, -10 to +10 → -1 to +1)
        tvl_change = raw_data.get('defi_tvl_change', 0)
        features.append(np.clip(tvl_change / 10, -1, 1))
        
        # 5. CME Gap (percent, inversed because price tends to fill gap)
        cme_gap = raw_data.get('cme_gap', 0)
        features.append(np.clip(-cme_gap / 5, -1, 1))  # Inverse
        
        # 6. News Sentiment (0-100 → 0-1)
        news = raw_data.get('news_sentiment', 50)
        features.append((news - 50) / 50)  # Center around 0
        
        # 7. Bullish Pattern Count (0-5 → 0-1)
        bullish_count = raw_data.get('bullish_patterns', 0)
        features.append(min(bullish_count / 3, 1))  # 3+ patterns = max
        
        # 8. Bearish Pattern Count (0-5 → 0-1, inversed)
        bearish_count = raw_data.get('bearish_patterns', 0)
        features.append(-min(bearish_count / 3, 1))  # Inverse
        
        # 9. Funding Rate (extreme = contrarian)
        funding = raw_data.get('funding_rate', 0)
        # High positive funding = bearish (crowded long), vice versa
        features.append(np.clip(-funding * 20, -1, 1))  # Inverse
        
        # 10. OI Velocity (high = volatility coming)
        oi_velocity = raw_data.get('oi_velocity', 0)
        features.append(np.clip(oi_velocity / 10, -1, 1))
        
        # 11. Whale Buy Ratio (0-1)
        whale_ratio = raw_data.get('whale_ratio', 0.5)
        features.append((whale_ratio - 0.5) * 2)  # Center around 0
        
        # 12. RSI (contrarian at extremes)
        rsi = raw_data.get('rsi', 50)
        if rsi < 30:
            features.append(0.5)  # Oversold = bullish
        elif rsi > 70:
            features.append(-0.5)  # Overbought = bearish
        else:
            features.append(0)
        
        # === PHASE 42 FEATURES (TEMPORARILY DISABLED) ===
        # Model needs retraining for 15 features. Current model trained with 12.
        # TODO: Retrain model with train_signal_combiner.py, then uncomment below
        
        # # 13. Liquidation Risk (0-1 scale, HIGH = bearish cascade risk)
        # liq_risk = raw_data.get('liquidation_risk', 0)
        # # Inverse: high risk = bearish
        # features.append(-np.clip(liq_risk, 0, 1))
        # 
        # # 14. Whale Flow (-1 to +1: negative=inflow/bearish, positive=outflow/bullish)
        # whale_flow = raw_data.get('whale_flow', 0)
        # # Normalize $100M flow = +/-1
        # features.append(np.clip(whale_flow / 100_000_000, -1, 1))
        # 
        # # 15. Reddit Sentiment (0-100 → -1 to +1)
        # reddit_sent = raw_data.get('reddit_sentiment', 50)
        # features.append((reddit_sent - 50) / 50)  # Center around 0
        
        return np.array(features)
    
    def predict(self, raw_data: Dict) -> CombinedSignal:
        """
        Ham sinyallerden birleştirilmiş tahmin üret.
        """
        # Normalize inputs
        features = self.normalize_inputs(raw_data)
        
        # Calculate weighted score
        if self.is_trained and self.model is not None:
            # Use trained model
            raw_score = self._predict_with_model(features)
        else:
            # Use weighted average (rule-based)
            raw_score = self._predict_weighted(features)
        
        # Convert to action and confidence
        confidence = abs(raw_score) * 100
        
        if raw_score > 0.6:
            action = 'STRONG_BUY'
        elif raw_score > 0.2:
            action = 'BUY'
        elif raw_score < -0.6:
            action = 'STRONG_SELL'
        elif raw_score < -0.2:
            action = 'SELL'
        else:
            action = 'NEUTRAL'
        
        # Find top contributors
        top_bullish, top_bearish = self._get_top_contributors(features, raw_data)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(action, confidence, top_bullish, top_bearish, raw_data)
        
        return CombinedSignal(
            action=action,
            confidence=confidence,
            raw_score=raw_score,
            top_bullish=top_bullish,
            top_bearish=top_bearish,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def _predict_weighted(self, features: np.ndarray) -> float:
        """Ağırlıklı ortalama ile tahmin"""
        weights = np.array(list(self.weights.values()))
        weighted_sum = np.sum(features * weights)
        return np.clip(weighted_sum, -1, 1)
    
    def _predict_with_model(self, features: np.ndarray) -> float:
        """Eğitilmiş model ile tahmin"""
        try:
            prediction = self.model.predict(features.reshape(1, -1))[0]
            return np.clip(prediction, -1, 1)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}")
            return self._predict_weighted(features)
    
    def _get_top_contributors(self, features: np.ndarray, raw_data: Dict) -> Tuple[List[str], List[str]]:
        """En çok katkı sağlayan sinyalleri bul"""
        contributions = {}
        weights = list(self.weights.values())
        
        for i, (name, weight) in enumerate(self.weights.items()):
            contributions[name] = features[i] * weight
        
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        top_bullish = []
        top_bearish = []
        
        for name, contrib in sorted_contributions:
            if contrib > 0.02:
                if name == 'fear_greed':
                    top_bullish.append(f"Fear & Greed düşük ({raw_data.get('fear_greed', 'N/A')})")
                elif name == 'tradingview':
                    top_bullish.append(f"TradingView: BUY sinyali")
                elif name == 'stablecoin_flow':
                    top_bullish.append(f"Stablecoin mint: ${raw_data.get('stablecoin_flow', 0)/1e6:+.0f}M")
                elif name == 'bullish_patterns':
                    top_bullish.append(f"{raw_data.get('bullish_patterns', 0)} bullish pattern")
                elif name == 'news_sentiment':
                    top_bullish.append(f"Haberler pozitif ({raw_data.get('news_sentiment', 'N/A')})")
                else:
                    top_bullish.append(name)
            elif contrib < -0.02:
                if name == 'fear_greed':
                    top_bearish.append(f"Fear & Greed yüksek ({raw_data.get('fear_greed', 'N/A')})")
                elif name == 'tradingview':
                    top_bearish.append(f"TradingView: SELL sinyali")
                elif name == 'bearish_patterns':
                    top_bearish.append(f"{raw_data.get('bearish_patterns', 0)} bearish pattern")
                elif name == 'funding_rate':
                    top_bearish.append(f"Extreme funding ({raw_data.get('funding_rate', 0):.3f}%)")
                elif name == 'cme_gap':
                    top_bearish.append(f"CME gap kapatılacak ({raw_data.get('cme_gap', 0):+.1f}%)")
                else:
                    top_bearish.append(name)
        
        return top_bullish[:3], top_bearish[:3]
    
    def _generate_reasoning(self, action: str, confidence: float, 
                           top_bullish: List[str], top_bearish: List[str],
                           raw_data: Dict) -> str:
        """Türkçe açıklama oluştur"""
        
        reasoning = f"AI {action} sinyali veriyor (güven: {confidence:.0f}%). "
        
        if action in ['STRONG_BUY', 'BUY']:
            reasoning += "Bullish faktörler: "
            reasoning += ", ".join(top_bullish[:3]) if top_bullish else "Genel trend pozitif"
            reasoning += ". "
        elif action in ['STRONG_SELL', 'SELL']:
            reasoning += "Bearish faktörler: "
            reasoning += ", ".join(top_bearish[:3]) if top_bearish else "Genel trend negatif"
            reasoning += ". "
        else:
            reasoning += "Sinyaller karışık, net yön yok. "
        
        # Add warning if low confidence
        if confidence < 40:
            reasoning += "⚠️ Düşük güven - dikkatli ol!"
        
        return reasoning
    
    # =========================================
    # TRAINING
    # =========================================
    def train(self, training_data: List[Tuple[Dict, float]], epochs: int = 100):
        """
        Modeli eğit.
        
        Args:
            training_data: [(raw_features, actual_return), ...]
            epochs: Eğitim iterasyonu
        """
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import train_test_split
            
            logger.info(f"Training Signal Combiner with {len(training_data)} samples...")
            
            # Prepare data
            X = np.array([self.normalize_inputs(d[0]) for d in training_data])
            y = np.array([d[1] for d in training_data])  # -1 to +1 returns
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Gradient Boosting
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Training complete! Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
            
            # Update feature importance as weights
            importances = self.model.feature_importances_
            for i, name in enumerate(self.FEATURE_NAMES):
                self.weights[name] = importances[i]
            
            self.is_trained = True
            self._save_model()
            
            return {'train_score': train_score, 'test_score': test_score}
            
        except ImportError:
            logger.warning("sklearn not installed, falling back to weight optimization")
            return self._train_weights(training_data, epochs)
    
    def _train_weights(self, training_data: List[Tuple[Dict, float]], epochs: int):
        """Basit ağırlık optimizasyonu"""
        
        best_weights = self.weights.copy()
        best_error = float('inf')
        
        for epoch in range(epochs):
            # Random perturbation
            new_weights = {}
            for k, v in self.weights.items():
                new_weights[k] = max(0.01, v + np.random.uniform(-0.02, 0.02))
            
            # Normalize weights
            total = sum(new_weights.values())
            new_weights = {k: v/total for k, v in new_weights.items()}
            
            # Calculate error
            self.weights = new_weights
            error = 0
            for raw_data, actual in training_data:
                features = self.normalize_inputs(raw_data)
                pred = self._predict_weighted(features)
                error += (pred - actual) ** 2
            
            if error < best_error:
                best_error = error
                best_weights = new_weights.copy()
        
        self.weights = best_weights
        self.is_trained = True
        self._save_model()
        
        logger.info(f"Weight optimization complete! MSE: {best_error/len(training_data):.4f}")
        return {'mse': best_error / len(training_data)}
    
    def generate_training_data(self, historical_signals: List[Dict], 
                               price_data: List[Dict], lookahead: int = 24) -> List[Tuple[Dict, float]]:
        """
        Geriye dönük training data oluştur.
        
        Her sinyal noktası için lookahead saat sonraki getiriyi hesapla.
        """
        training_data = []
        
        for i, signals in enumerate(historical_signals):
            if i + lookahead >= len(price_data):
                break
            
            current_price = price_data[i].get('close', 0)
            future_price = price_data[i + lookahead].get('close', 0)
            
            if current_price > 0:
                # Calculate return (-1 to +1 scale)
                raw_return = (future_price - current_price) / current_price
                normalized_return = np.clip(raw_return * 10, -1, 1)  # 10% = max
                
                training_data.append((signals, normalized_return))
        
        return training_data
    
    # =========================================
    # SAVE/LOAD
    # =========================================
    def _save_model(self):
        """Modeli kaydet"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'weights': self.weights,
                'model': self.model,
                'is_trained': self.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Signal Combiner model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    state = pickle.load(f)
                
                self.weights = state.get('weights', self.DEFAULT_WEIGHTS)
                self.model = state.get('model')
                self.is_trained = state.get('is_trained', False)
                
                logger.info(f"Signal Combiner model loaded (trained: {self.is_trained})")
                
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
    
    # =========================================
    # FORMAT FOR TELEGRAM
    # =========================================
    def format_for_telegram(self, signal: CombinedSignal) -> str:
        """Telegram için formatla"""
        
        # Action emoji
        if signal.action == 'STRONG_BUY':
            emoji = "🟢🟢"
        elif signal.action == 'BUY':
            emoji = "🟢"
        elif signal.action == 'STRONG_SELL':
            emoji = "🔴🔴"
        elif signal.action == 'SELL':
            emoji = "🔴"
        else:
            emoji = "⚪"
        
        # Confidence bar
        conf_bars = int(signal.confidence / 20)
        conf_bar = "█" * conf_bars + "░" * (5 - conf_bars)
        
        msg = "🤖 *AI Birleşik Sinyal*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"{emoji} *{signal.action}*\n"
        msg += f"📊 Güven: [{conf_bar}] {signal.confidence:.0f}%\n\n"
        
        if signal.top_bullish:
            msg += "🟢 *Bullish faktörler:*\n"
            for factor in signal.top_bullish:
                msg += f"• {factor}\n"
            msg += "\n"
        
        if signal.top_bearish:
            msg += "🔴 *Bearish faktörler:*\n"
            for factor in signal.top_bearish:
                msg += f"• {factor}\n"
            msg += "\n"
        
        msg += f"💡 _{signal.reasoning}_\n"
        msg += f"\n⏰ _{signal.timestamp.strftime('%H:%M:%S')}_"
        
        return msg
