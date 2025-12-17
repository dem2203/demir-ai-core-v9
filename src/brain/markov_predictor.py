# -*- coding: utf-8 -*-
"""
DEMIR AI - Markov Chain Predictor
Durum geçiş olasılıklarına dayalı 1-2 saat tahmin modeli.

PHASE 48: Advanced Prediction
- Piyasa durumlarını tanımlar (STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN)
- Geçiş matrisini tarihsel veriden öğrenir
- Gelecek durumu olasılıklarla tahmin eder
- Trend süresi tahmini yapar
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("MARKOV_PREDICTOR")


class MarkovPredictor:
    """
    Markov Zinciri Tahmin Modeli
    
    Piyasa durumları:
    - STRONG_UP:   > +2% değişim
    - UP:          +0.5% to +2%
    - NEUTRAL:     -0.5% to +0.5%
    - DOWN:        -2% to -0.5%
    - STRONG_DOWN: < -2%
    
    Her saat için durum geçiş olasılıkları hesaplanır.
    """
    
    # Piyasa durumları
    STATES = ['STRONG_UP', 'UP', 'NEUTRAL', 'DOWN', 'STRONG_DOWN']
    
    # Durum eşikleri (yüzde değişim)
    THRESHOLDS = {
        'STRONG_UP': 2.0,
        'UP': 0.5,
        'NEUTRAL': 0.0,  # -0.5 to +0.5
        'DOWN': -0.5,
        'STRONG_DOWN': -2.0
    }
    
    def __init__(self):
        # Geçiş matrisi - varsayılan değerler (uniform olmayan, gerçekçi)
        self.transition_matrix = self._initialize_default_matrix()
        self.state_history = []
        self.trained = False
        self.last_train = None
    
    def _initialize_default_matrix(self) -> np.ndarray:
        """
        Varsayılan geçiş matrisi.
        
        Gerçek piyasa davranışına yakın:
        - Momentum devam etme eğilimi
        - Extreme durumlardan geri dönüş eğilimi
        """
        # [STRONG_UP, UP, NEUTRAL, DOWN, STRONG_DOWN]
        matrix = np.array([
            # STRONG_UP'tan:
            [0.35, 0.40, 0.15, 0.08, 0.02],
            # UP'tan:
            [0.20, 0.40, 0.25, 0.12, 0.03],
            # NEUTRAL'dan:
            [0.10, 0.25, 0.35, 0.22, 0.08],
            # DOWN'dan:
            [0.05, 0.12, 0.23, 0.40, 0.20],
            # STRONG_DOWN'dan:
            [0.02, 0.08, 0.15, 0.35, 0.40],
        ])
        return matrix
    
    def classify_state(self, pct_change: float) -> str:
        """Yüzde değişimi duruma dönüştür."""
        if pct_change >= 2.0:
            return 'STRONG_UP'
        elif pct_change >= 0.5:
            return 'UP'
        elif pct_change >= -0.5:
            return 'NEUTRAL'
        elif pct_change >= -2.0:
            return 'DOWN'
        else:
            return 'STRONG_DOWN'
    
    def train(self, price_data: pd.DataFrame, interval_hours: int = 1):
        """
        Tarihsel veriden geçiş matrisini öğren.
        
        Args:
            price_data: DataFrame with 'close' column
            interval_hours: Durum aralığı (1 veya 2 saat)
        """
        if price_data.empty or len(price_data) < 24:
            logger.warning("Insufficient data for training")
            return
        
        # Saatlik değişimleri hesapla
        df = price_data.copy()
        df['pct_change'] = df['close'].pct_change(periods=interval_hours) * 100
        df['state'] = df['pct_change'].apply(self.classify_state)
        
        # Geçişleri say
        transition_counts = defaultdict(lambda: defaultdict(int))
        states = df['state'].dropna().tolist()
        
        for i in range(len(states) - 1):
            current = states[i]
            next_state = states[i + 1]
            transition_counts[current][next_state] += 1
        
        # Sayıları olasılıklara dönüştür
        matrix = np.zeros((5, 5))
        for i, from_state in enumerate(self.STATES):
            total = sum(transition_counts[from_state].values())
            if total > 0:
                for j, to_state in enumerate(self.STATES):
                    matrix[i, j] = transition_counts[from_state][to_state] / total
            else:
                # Veri yoksa varsayılan kullan
                matrix[i] = self.transition_matrix[i]
        
        self.transition_matrix = matrix
        self.state_history = states[-100:]  # Son 100 durumu sakla
        self.trained = True
        self.last_train = datetime.now()
        
        logger.info(f"Markov model trained on {len(states)} states")
    
    def predict_next_state(self, current_state: str, steps: int = 1) -> Dict:
        """
        Sonraki durumu tahmin et.
        
        Args:
            current_state: Mevcut piyasa durumu
            steps: Kaç adım sonrasını tahmin et (1=1 saat, 2=2 saat)
        
        Returns:
            {
                'most_likely_state': 'UP',
                'probability': 0.42,
                'all_probabilities': {...},
                'confidence': 'HIGH' / 'MEDIUM' / 'LOW',
                'direction': 'BULLISH' / 'BEARISH' / 'NEUTRAL'
            }
        """
        if current_state not in self.STATES:
            current_state = 'NEUTRAL'
        
        state_idx = self.STATES.index(current_state)
        
        # Çoklu adım için matris üssü
        if steps > 1:
            prob_matrix = np.linalg.matrix_power(self.transition_matrix, steps)
            probs = prob_matrix[state_idx]
        else:
            probs = self.transition_matrix[state_idx]
        
        # En olası durumu bul
        most_likely_idx = np.argmax(probs)
        most_likely_state = self.STATES[most_likely_idx]
        probability = probs[most_likely_idx]
        
        # Güven seviyesi
        if probability > 0.5:
            confidence = 'HIGH'
        elif probability > 0.35:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Yön belirleme
        bullish_prob = probs[0] + probs[1]  # STRONG_UP + UP
        bearish_prob = probs[3] + probs[4]  # DOWN + STRONG_DOWN
        
        if bullish_prob > bearish_prob + 0.15:
            direction = 'BULLISH'
        elif bearish_prob > bullish_prob + 0.15:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'most_likely_state': most_likely_state,
            'probability': round(probability * 100, 1),
            'all_probabilities': {
                state: round(probs[i] * 100, 1)
                for i, state in enumerate(self.STATES)
            },
            'confidence': confidence,
            'direction': direction,
            'bullish_probability': round(bullish_prob * 100, 1),
            'bearish_probability': round(bearish_prob * 100, 1),
            'steps_ahead': steps,
            'timestamp': datetime.now()
        }
    
    def predict_1_2_hours(self, current_pct_change: float) -> Dict:
        """
        1 ve 2 saat sonrasını tahmin et.
        
        Args:
            current_pct_change: Son saatteki yüzde değişim
        
        Returns:
            {
                '1_hour': {...},
                '2_hour': {...},
                'combined_signal': 'LONG' / 'SHORT' / 'WAIT',
                'signal_strength': 0-100
            }
        """
        current_state = self.classify_state(current_pct_change)
        
        pred_1h = self.predict_next_state(current_state, steps=1)
        pred_2h = self.predict_next_state(current_state, steps=2)
        
        # Kombine sinyal
        score = 0
        
        # 1 saat tahmini
        if pred_1h['direction'] == 'BULLISH':
            score += pred_1h['bullish_probability'] - 50
        elif pred_1h['direction'] == 'BEARISH':
            score -= pred_1h['bearish_probability'] - 50
        
        # 2 saat tahmini (daha az ağırlık)
        if pred_2h['direction'] == 'BULLISH':
            score += (pred_2h['bullish_probability'] - 50) * 0.5
        elif pred_2h['direction'] == 'BEARISH':
            score -= (pred_2h['bearish_probability'] - 50) * 0.5
        
        # Sinyal belirleme
        if score > 15:
            combined_signal = 'LONG'
        elif score < -15:
            combined_signal = 'SHORT'
        else:
            combined_signal = 'WAIT'
        
        signal_strength = min(100, abs(score) * 2)
        
        return {
            '1_hour': pred_1h,
            '2_hour': pred_2h,
            'current_state': current_state,
            'combined_signal': combined_signal,
            'signal_strength': round(signal_strength, 1),
            'score': round(score, 1),
            'timestamp': datetime.now()
        }
    
    def get_trend_duration_estimate(self, current_state: str) -> Dict:
        """
        Mevcut trendin tahmini süresini hesapla.
        
        Returns:
            {
                'expected_duration_hours': 3.5,
                'reversal_probability': 0.25,
                'momentum_strength': 'STRONG' / 'MODERATE' / 'WEAK'
            }
        """
        state_idx = self.STATES.index(current_state) if current_state in self.STATES else 2
        
        # Aynı durumda kalma olasılığı
        stay_prob = self.transition_matrix[state_idx, state_idx]
        
        # Beklenen süre (geometrik dağılım)
        if stay_prob < 0.99:
            expected_duration = 1 / (1 - stay_prob)
        else:
            expected_duration = 10
        
        # Tersine dönüş olasılığı
        if state_idx <= 1:  # Bullish states
            reversal_prob = self.transition_matrix[state_idx, 3] + self.transition_matrix[state_idx, 4]
        elif state_idx >= 3:  # Bearish states
            reversal_prob = self.transition_matrix[state_idx, 0] + self.transition_matrix[state_idx, 1]
        else:
            reversal_prob = 0.5
        
        # Momentum gücü
        if stay_prob > 0.45:
            momentum = 'STRONG'
        elif stay_prob > 0.30:
            momentum = 'MODERATE'
        else:
            momentum = 'WEAK'
        
        return {
            'expected_duration_hours': round(expected_duration, 1),
            'reversal_probability': round(reversal_prob * 100, 1),
            'momentum_strength': momentum,
            'stay_probability': round(stay_prob * 100, 1)
        }
    
    def format_for_dashboard(self, prediction: Dict) -> str:
        """Dashboard için formatlanmış tahmin."""
        signal = prediction.get('combined_signal', 'WAIT')
        strength = prediction.get('signal_strength', 0)
        
        emoji = "🟢" if signal == "LONG" else "🔴" if signal == "SHORT" else "⚪"
        
        lines = [
            f"{emoji} **Markov Tahmin:** {signal}",
            f"📊 Güç: {strength:.0f}%",
            f"⏰ 1 Saat: {prediction['1_hour']['direction']} ({prediction['1_hour']['probability']:.0f}%)",
            f"⏰ 2 Saat: {prediction['2_hour']['direction']} ({prediction['2_hour']['probability']:.0f}%)"
        ]
        
        return "\n".join(lines)


# Convenience functions
def predict_short_term(current_change: float) -> Dict:
    """Hızlı 1-2 saat tahmini."""
    predictor = MarkovPredictor()
    return predictor.predict_1_2_hours(current_change)


def get_markov_signal(current_change: float) -> str:
    """Sadece sinyal al: LONG/SHORT/WAIT."""
    result = predict_short_term(current_change)
    return result['combined_signal']
