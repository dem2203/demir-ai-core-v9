# -*- coding: utf-8 -*-
"""
DEMIR AI - Regime Adaptive Weights
Piyasa rejimine göre modül ağırlıklarını otomatik değiştirir.

PHASE 96: True AI - Market Regime Adaptation
- Piyasa rejimini tespit et (BULL/BEAR/RANGE/VOLATILE)
- Her rejim için farklı ağırlık profili
- Otomatik rejim geçişi
- Rejime göre sinyal güvenilirliği
"""
import json
import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

logger = logging.getLogger("REGIME_ADAPTER")


class RegimeAdaptiveWeights:
    """
    Rejim Adaptif Ağırlıklar
    
    Piyasa koşullarına göre modül ağırlıklarını dinamik ayarlar.
    Bull'da trend modülleri, bear'da reversal modülleri, range'de mean-reversion.
    """
    
    REGIME_FILE = "regime_data.json"
    
    # Regime profiles - hangi modül hangi rejimde daha iyi?
    REGIME_MULTIPLIERS = {
        'BULL': {
            # Trend-following modüller boost
            'LSTMTrend': 1.5,
            'TradingViewTA': 1.4,
            'MultiTimeframe': 1.3,
            'MarkovPredictor': 1.2,
            'OnChainIntel': 1.3,
            # Contrarian modüller azalt
            'CGTopTraderLS': 0.7,
            'BollingerSqueeze': 0.8,
        },
        'BEAR': {
            # Reversal ve short modüller boost
            'SMCAnalyzer': 1.4,
            'LiquidationHunter': 1.5,
            'LiquidationCascade': 1.5,
            'CGLiquidationMap': 1.4,
            'VolumeSpike': 1.3,
            # Trend modüller azalt
            'LSTMTrend': 0.8,
            'OnChainIntel': 0.7,
        },
        'RANGE': {
            # Mean-reversion modüller boost
            'BollingerSqueeze': 1.5,
            'CGOrderbookDelta': 1.4,
            'TakerFlowDelta': 1.3,
            'OptionsFlow': 1.3,
            'SMCAnalyzer': 1.2,
            # Trend modüller azalt
            'LSTMTrend': 0.6,
            'MultiTimeframe': 0.7,
        },
        'VOLATILE': {
            # Risk-aware modüller boost
            'VolatilityPredictor': 1.6,
            'LiquidationCascade': 1.5,
            'VolumeSpike': 1.4,
            'CGFundingExtreme': 1.4,
            'ExchangeDivergence': 1.3,
            # Long-term modüller azalt
            'MarkovPredictor': 0.7,
            'OnChainIntel': 0.6,
        }
    }
    
    def __init__(self):
        self.current_regime: str = 'UNKNOWN'
        self.regime_confidence: float = 0
        self.regime_history: list = []
        self.last_detection: Optional[datetime] = None
        self._load_data()
        logger.info("✅ Regime Adaptive Weights initialized")
    
    def _load_data(self):
        """Mevcut verileri yükle."""
        try:
            if os.path.exists(self.REGIME_FILE):
                with open(self.REGIME_FILE, 'r') as f:
                    data = json.load(f)
                    self.current_regime = data.get('current_regime', 'UNKNOWN')
                    self.regime_confidence = data.get('regime_confidence', 0)
                    self.regime_history = data.get('regime_history', [])[-50:]  # Last 50
        except Exception as e:
            logger.warning(f"Regime data load failed: {e}")
    
    def _save_data(self):
        """Verileri kaydet."""
        try:
            with open(self.REGIME_FILE, 'w') as f:
                json.dump({
                    'current_regime': self.current_regime,
                    'regime_confidence': self.regime_confidence,
                    'regime_history': self.regime_history[-50:],
                    'last_detection': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Regime data save failed: {e}")
    
    def detect_regime(self, symbol: str = 'BTCUSDT') -> Tuple[str, float]:
        """
        Mevcut piyasa rejimini tespit et.
        
        Returns:
            (regime, confidence)
        """
        try:
            # 4 saatlik mumları al
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '4h', 'limit': 30},
                timeout=10
            )
            
            if resp.status_code != 200:
                return self.current_regime, self.regime_confidence
            
            klines = resp.json()
            
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            
            # Trend analizi
            sma_short = sum(closes[-7:]) / 7
            sma_long = sum(closes[-20:]) / 20
            current_price = closes[-1]
            
            # Volatilite analizi
            avg_range = sum([(h - l) for h, l in zip(highs, lows)]) / len(highs)
            recent_range = sum([(h - l) for h, l in zip(highs[-7:], lows[-7:])]) / 7
            volatility_ratio = recent_range / avg_range if avg_range > 0 else 1
            
            # Trend strength
            price_change_pct = ((current_price - closes[0]) / closes[0]) * 100
            
            # Regime determination
            regime = 'UNKNOWN'
            confidence = 50
            
            # VOLATILE: Yüksek volatilite
            if volatility_ratio > 1.5:
                regime = 'VOLATILE'
                confidence = min(90, 50 + (volatility_ratio - 1) * 40)
            
            # BULL: Güçlü yükseliş trendi
            elif price_change_pct > 5 and sma_short > sma_long:
                regime = 'BULL'
                confidence = min(90, 50 + price_change_pct * 4)
            
            # BEAR: Güçlü düşüş trendi
            elif price_change_pct < -5 and sma_short < sma_long:
                regime = 'BEAR'
                confidence = min(90, 50 + abs(price_change_pct) * 4)
            
            # RANGE: Yatay piyasa
            elif abs(price_change_pct) < 3 and volatility_ratio < 1.2:
                regime = 'RANGE'
                confidence = 70
            
            # Orta durumlar
            elif sma_short > sma_long:
                regime = 'BULL'
                confidence = 60
            else:
                regime = 'BEAR'
                confidence = 60
            
            # Güncelle
            self.current_regime = regime
            self.regime_confidence = confidence
            self.regime_history.append({
                'regime': regime,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'price': current_price
            })
            self.last_detection = datetime.now()
            self._save_data()
            
            logger.info(f"📊 Regime detected: {regime} ({confidence:.0f}% confidence)")
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return self.current_regime, self.regime_confidence
    
    def apply_regime_weights(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Mevcut rejime göre ağırlıkları ayarla.
        
        Args:
            base_weights: Temel modül ağırlıkları
            
        Returns:
            Rejime göre ayarlanmış ağırlıklar
        """
        if self.current_regime not in self.REGIME_MULTIPLIERS:
            return base_weights
        
        multipliers = self.REGIME_MULTIPLIERS[self.current_regime]
        new_weights = {}
        
        for module, weight in base_weights.items():
            multiplier = multipliers.get(module, 1.0)
            
            # Confidence'a göre multiplier'ı yumuşat
            # %100 confidence = full multiplier, %50 = half effect
            confidence_factor = self.regime_confidence / 100
            adjusted_multiplier = 1 + (multiplier - 1) * confidence_factor
            
            new_weights[module] = weight * adjusted_multiplier
        
        # Normalize
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v/total for k, v in new_weights.items()}
        
        return new_weights
    
    def get_regime_info(self) -> Dict:
        """Rejim bilgisi getir."""
        return {
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'last_detection': self.last_detection.isoformat() if self.last_detection else None,
            'history_count': len(self.regime_history)
        }


# Global instance
_adapter = None

def get_adapter() -> RegimeAdaptiveWeights:
    """Get or create adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = RegimeAdaptiveWeights()
    return _adapter
