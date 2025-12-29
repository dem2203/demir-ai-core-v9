# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - Advanced Pattern Detection
=========================================
Elliott Wave ve Harmonic Pattern tespiti.

Features:
1. Elliott Wave Detector (Impulse & Corrective)
2. Harmonic Pattern Scanner (Gartley, Bat, Butterfly, Crab)
3. Market Structure Analyzer (BOS, CHoCH)
4. Fibonacci Levels
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("ADVANCED_PATTERNS")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WavePoint:
    """Elliott Wave noktası"""
    index: int
    price: float
    wave_label: str  # 1, 2, 3, 4, 5, A, B, C
    timestamp: datetime = None


@dataclass
class ElliottWaveResult:
    """Elliott Wave analiz sonucu"""
    is_valid: bool = False
    wave_type: str = ""  # "IMPULSE" veya "CORRECTIVE"
    current_wave: str = ""  # "1", "2", "3", "4", "5", "A", "B", "C"
    wave_direction: str = "NEUTRAL"  # "UP", "DOWN"
    confidence: float = 0.0
    next_target: float = 0.0
    invalidation_level: float = 0.0
    wave_points: List[WavePoint] = field(default_factory=list)
    fib_levels: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'wave_type': self.wave_type,
            'current_wave': self.current_wave,
            'wave_direction': self.wave_direction,
            'confidence': self.confidence,
            'next_target': self.next_target,
            'invalidation_level': self.invalidation_level,
            'fib_levels': self.fib_levels
        }


@dataclass
class HarmonicPattern:
    """Harmonic Pattern sonucu"""
    pattern_name: str = ""  # Gartley, Bat, Butterfly, Crab, Shark
    is_bullish: bool = True
    confidence: float = 0.0
    prz_low: float = 0.0  # Potential Reversal Zone
    prz_high: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    xabcd_points: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'pattern_name': self.pattern_name,
            'is_bullish': self.is_bullish,
            'confidence': self.confidence,
            'prz_low': self.prz_low,
            'prz_high': self.prz_high,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2
        }


@dataclass
class MarketStructure:
    """Market Structure analizi"""
    trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    bos_detected: bool = False  # Break of Structure
    bos_direction: str = ""
    choch_detected: bool = False  # Change of Character
    key_levels: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'trend': self.trend,
            'last_swing_high': self.last_swing_high,
            'last_swing_low': self.last_swing_low,
            'bos_detected': self.bos_detected,
            'bos_direction': self.bos_direction,
            'choch_detected': self.choch_detected,
            'key_levels': self.key_levels
        }


# =============================================================================
# ELLIOTT WAVE DETECTOR
# =============================================================================

class ElliottWaveDetector:
    """
    Elliott Wave Pattern Dedektörü
    
    Kurallar:
    - Wave 2, Wave 1'in başlangıcının altına inemez (genellikle %38.2-%78.6 retrace)
    - Wave 3 genellikle en uzun dalgadır (%161.8-%261.8 extension)
    - Wave 4, Wave 1'in zirvesine giremez
    - Wave 5 genellikle Wave 3'ün %61.8-%100'ü kadar uzar
    """
    
    # Fibonacci seviyeleri
    FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0, 2.618]
    
    def __init__(self):
        self.min_wave_size = 0.02  # Minimum %2 hareket
    
    def analyze(self, prices: List[float], highs: List[float] = None, lows: List[float] = None) -> ElliottWaveResult:
        """
        Fiyat serisinden Elliott Wave tespiti yap.
        
        Args:
            prices: Close fiyatları (en az 50 bar)
            highs: High fiyatları (opsiyonel, daha doğru pivot tespiti için)
            lows: Low fiyatları (opsiyonel)
        """
        if len(prices) < 50:
            return ElliottWaveResult()
        
        prices = np.array(prices)
        
        # Swing noktalarını bul
        pivots = self._find_pivot_points(prices, highs, lows)
        
        if len(pivots) < 6:
            return ElliottWaveResult()
        
        # Impulse wave kontrolü (5 dalga)
        impulse_result = self._check_impulse_wave(pivots, prices)
        if impulse_result.is_valid:
            return impulse_result
        
        # Corrective wave kontrolü (ABC)
        corrective_result = self._check_corrective_wave(pivots, prices)
        if corrective_result.is_valid:
            return corrective_result
        
        return ElliottWaveResult()
    
    def _find_pivot_points(self, prices: np.ndarray, highs: np.ndarray = None, lows: np.ndarray = None) -> List[WavePoint]:
        """Swing high/low noktalarını bul."""
        pivots = []
        lookback = 5
        
        use_hl = highs is not None and lows is not None
        
        for i in range(lookback, len(prices) - lookback):
            # Swing High
            if use_hl:
                is_high = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                          all(highs[i] > highs[i+j] for j in range(1, lookback+1))
                is_low = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                         all(lows[i] < lows[i+j] for j in range(1, lookback+1))
                price = highs[i] if is_high else (lows[i] if is_low else 0)
            else:
                is_high = all(prices[i] > prices[i-j] for j in range(1, lookback+1)) and \
                          all(prices[i] > prices[i+j] for j in range(1, lookback+1))
                is_low = all(prices[i] < prices[i-j] for j in range(1, lookback+1)) and \
                         all(prices[i] < prices[i+j] for j in range(1, lookback+1))
                price = prices[i]
            
            if is_high:
                pivots.append(WavePoint(index=i, price=price, wave_label="H"))
            elif is_low:
                pivots.append(WavePoint(index=i, price=price, wave_label="L"))
        
        return pivots
    
    def _check_impulse_wave(self, pivots: List[WavePoint], prices: np.ndarray) -> ElliottWaveResult:
        """5-wave impulse pattern kontrolü."""
        if len(pivots) < 6:
            return ElliottWaveResult()
        
        # Son 6 pivot'u al (5 dalga için)
        recent = pivots[-6:]
        
        # Yükseliş trendi kontrolü (H, L, H, L, H, L pattern)
        is_bullish = recent[0].price < recent[2].price < recent[4].price
        is_bearish = recent[0].price > recent[2].price > recent[4].price
        
        if not (is_bullish or is_bearish):
            return ElliottWaveResult()
        
        # Elliott kurallarını kontrol et
        if is_bullish:
            wave1 = recent[0].price
            wave2 = recent[1].price
            wave3_end = recent[2].price
            wave4 = recent[3].price
            wave5_end = recent[4].price
            
            # Kural 1: Wave 2, Wave 1'in başlangıcının altına inemez
            # (Bu örnekte basitleştirilmiş)
            
            # Kural 2: Wave 4, Wave 1'in zirvesine giremez
            if wave4 < wave3_end * 0.618:  # Basit kontrol
                wave_valid = True
            else:
                wave_valid = False
            
            if wave_valid:
                # Fibonacci hedefleri hesapla
                wave1_size = wave3_end - wave1
                fib_levels = {
                    '38.2%': wave5_end - wave1_size * 0.382,
                    '50.0%': wave5_end - wave1_size * 0.5,
                    '61.8%': wave5_end - wave1_size * 0.618,
                    '100%': wave5_end - wave1_size,
                    '161.8%': wave5_end + wave1_size * 0.618
                }
                
                return ElliottWaveResult(
                    is_valid=True,
                    wave_type="IMPULSE",
                    current_wave="5",
                    wave_direction="UP",
                    confidence=70.0,
                    next_target=fib_levels['161.8%'],
                    invalidation_level=wave4,
                    fib_levels=fib_levels
                )
        
        return ElliottWaveResult()
    
    def _check_corrective_wave(self, pivots: List[WavePoint], prices: np.ndarray) -> ElliottWaveResult:
        """ABC corrective pattern kontrolü."""
        if len(pivots) < 4:
            return ElliottWaveResult()
        
        recent = pivots[-4:]
        
        # Basit ABC pattern: A-down, B-up (retest), C-down (final)
        is_abc_down = recent[0].price > recent[1].price and \
                      recent[1].price < recent[2].price and \
                      recent[2].price > recent[3].price
        
        is_abc_up = recent[0].price < recent[1].price and \
                    recent[1].price > recent[2].price and \
                    recent[2].price < recent[3].price
        
        if is_abc_down:
            a_start = recent[0].price
            a_end = recent[1].price
            b_end = recent[2].price
            c_end = recent[3].price
            
            # B genellikle A'nın %38.2-%61.8'ini retrace eder
            a_size = a_start - a_end
            b_retrace = (b_end - a_end) / a_size if a_size != 0 else 0
            
            if 0.382 <= b_retrace <= 0.786:
                return ElliottWaveResult(
                    is_valid=True,
                    wave_type="CORRECTIVE",
                    current_wave="C",
                    wave_direction="DOWN",
                    confidence=65.0,
                    next_target=a_end - a_size * 0.618,
                    invalidation_level=b_end,
                    fib_levels={
                        'A_end': a_end,
                        'B_end': b_end,
                        'C_target': a_end - a_size * 0.618
                    }
                )
        
        return ElliottWaveResult()
    
    def calculate_fib_levels(self, high: float, low: float) -> Dict[str, float]:
        """Fibonacci retracement seviyeleri hesapla."""
        diff = high - low
        return {
            f'{int(level*100)}%': low + diff * level for level in self.FIB_LEVELS
        }


# =============================================================================
# HARMONIC PATTERN SCANNER
# =============================================================================

class HarmonicPatternScanner:
    """
    Harmonic Pattern Tarayıcı
    
    Desteklenen Patternler:
    - Gartley (XA: 0.618, AB: 0.382-0.886, BC: 0.382-0.886, CD: 1.27-1.618)
    - Bat (XA: 0.382-0.5, AB: 0.382-0.886, BC: 0.382-0.886, CD: 1.618-2.618)
    - Butterfly (XA: 0.786, AB: 0.382-0.886, BC: 0.382-0.886, CD: 1.618-2.618)
    - Crab (XA: 0.382-0.618, AB: 0.382-0.886, BC: 0.382-0.886, CD: 2.618-3.618)
    """
    
    # Pattern kuralları (min, max ratios)
    PATTERNS = {
        'Gartley': {
            'XA': (0.618, 0.618),
            'AB': (0.382, 0.886),
            'BC': (0.382, 0.886),
            'CD': (1.27, 1.618)
        },
        'Bat': {
            'XA': (0.382, 0.5),
            'AB': (0.382, 0.886),
            'BC': (0.382, 0.886),
            'CD': (1.618, 2.618)
        },
        'Butterfly': {
            'XA': (0.786, 0.786),
            'AB': (0.382, 0.886),
            'BC': (0.382, 0.886),
            'CD': (1.618, 2.618)
        },
        'Crab': {
            'XA': (0.382, 0.618),
            'AB': (0.382, 0.886),
            'BC': (0.382, 0.886),
            'CD': (2.618, 3.618)
        }
    }
    
    def __init__(self):
        self.tolerance = 0.05  # %5 tolerans
    
    def scan(self, prices: List[float], highs: List[float] = None, lows: List[float] = None) -> List[HarmonicPattern]:
        """
        Harmonic pattern taraması yap.
        
        Returns:
            Tespit edilen pattern'lerin listesi
        """
        patterns = []
        
        if len(prices) < 30:
            return patterns
        
        # Pivot noktalarını bul
        pivots = self._find_pivots(prices, highs, lows)
        
        if len(pivots) < 5:
            return patterns
        
        # Her 5-nokta kombinasyonunu kontrol et
        for i in range(len(pivots) - 4):
            xabcd = pivots[i:i+5]
            
            # Her pattern tipini kontrol et
            for pattern_name, rules in self.PATTERNS.items():
                result = self._check_pattern(xabcd, pattern_name, rules)
                if result:
                    patterns.append(result)
        
        return patterns
    
    def _find_pivots(self, prices: List[float], highs: List[float] = None, lows: List[float] = None) -> List[Dict]:
        """Pivot noktalarını bul."""
        pivots = []
        lookback = 3
        
        data = prices
        
        for i in range(lookback, len(data) - lookback):
            is_high = all(data[i] > data[i-j] for j in range(1, lookback+1)) and \
                      all(data[i] > data[i+j] for j in range(1, lookback+1))
            is_low = all(data[i] < data[i-j] for j in range(1, lookback+1)) and \
                     all(data[i] < data[i+j] for j in range(1, lookback+1))
            
            if is_high or is_low:
                pivots.append({
                    'index': i,
                    'price': data[i],
                    'type': 'H' if is_high else 'L'
                })
        
        return pivots
    
    def _check_pattern(self, points: List[Dict], pattern_name: str, rules: Dict) -> Optional[HarmonicPattern]:
        """Belirli bir pattern'i kontrol et."""
        if len(points) < 5:
            return None
        
        x, a, b, c, d = [p['price'] for p in points]
        
        # XABCD oranlarını hesapla
        xa_leg = abs(a - x)
        ab_leg = abs(b - a)
        bc_leg = abs(c - b)
        cd_leg = abs(d - c)
        
        if xa_leg == 0:
            return None
        
        ab_ratio = ab_leg / xa_leg
        bc_ratio = bc_leg / ab_leg if ab_leg != 0 else 0
        cd_ratio = cd_leg / bc_leg if bc_leg != 0 else 0
        
        # Kuralları kontrol et
        ab_min, ab_max = rules['AB']
        bc_min, bc_max = rules['BC']
        cd_min, cd_max = rules['CD']
        
        ab_valid = ab_min - self.tolerance <= ab_ratio <= ab_max + self.tolerance
        bc_valid = bc_min - self.tolerance <= bc_ratio <= bc_max + self.tolerance
        cd_valid = cd_min - self.tolerance <= cd_ratio <= cd_max + self.tolerance
        
        if ab_valid and bc_valid and cd_valid:
            is_bullish = x < a  # X'ten A'ya yükseliş = Bullish pattern
            
            # PRZ (Potential Reversal Zone) hesapla
            prz_center = d
            prz_range = abs(d - c) * 0.1
            
            return HarmonicPattern(
                pattern_name=pattern_name,
                is_bullish=is_bullish,
                confidence=self._calculate_confidence(ab_ratio, bc_ratio, cd_ratio, rules),
                prz_low=prz_center - prz_range,
                prz_high=prz_center + prz_range,
                entry_price=prz_center,
                stop_loss=x if is_bullish else a,
                take_profit_1=b,
                take_profit_2=a if is_bullish else x,
                xabcd_points={'X': x, 'A': a, 'B': b, 'C': c, 'D': d}
            )
        
        return None
    
    def _calculate_confidence(self, ab: float, bc: float, cd: float, rules: Dict) -> float:
        """Pattern güvenilirlik skoru hesapla."""
        ab_mid = (rules['AB'][0] + rules['AB'][1]) / 2
        bc_mid = (rules['BC'][0] + rules['BC'][1]) / 2
        cd_mid = (rules['CD'][0] + rules['CD'][1]) / 2
        
        ab_dev = abs(ab - ab_mid) / ab_mid if ab_mid != 0 else 1
        bc_dev = abs(bc - bc_mid) / bc_mid if bc_mid != 0 else 1
        cd_dev = abs(cd - cd_mid) / cd_mid if cd_mid != 0 else 1
        
        avg_dev = (ab_dev + bc_dev + cd_dev) / 3
        confidence = max(0, 100 - avg_dev * 100)
        
        return round(confidence, 1)


# =============================================================================
# MARKET STRUCTURE ANALYZER
# =============================================================================

class MarketStructureAnalyzer:
    """
    Market Structure Analizi (SMC/ICT Style)
    
    - Higher Highs / Higher Lows (Bullish)
    - Lower Highs / Lower Lows (Bearish)
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    """
    
    def __init__(self):
        self.lookback = 20
    
    def analyze(self, prices: List[float], highs: List[float] = None, lows: List[float] = None) -> MarketStructure:
        """Market structure analizi yap."""
        if len(prices) < self.lookback:
            return MarketStructure()
        
        prices = np.array(prices)
        
        # Swing noktalarını bul
        swing_highs = self._find_swing_highs(prices)
        swing_lows = self._find_swing_lows(prices)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return MarketStructure()
        
        # Trend belirleme
        trend = self._determine_trend(swing_highs, swing_lows)
        
        # BOS kontrolü
        bos_detected, bos_direction = self._check_bos(prices, swing_highs, swing_lows)
        
        # CHoCH kontrolü
        choch_detected = self._check_choch(swing_highs, swing_lows, trend)
        
        # Key levels
        key_levels = self._find_key_levels(swing_highs, swing_lows)
        
        return MarketStructure(
            trend=trend,
            last_swing_high=swing_highs[-1] if swing_highs else 0,
            last_swing_low=swing_lows[-1] if swing_lows else 0,
            bos_detected=bos_detected,
            bos_direction=bos_direction,
            choch_detected=choch_detected,
            key_levels=key_levels
        )
    
    def _find_swing_highs(self, prices: np.ndarray) -> List[float]:
        """Swing high noktalarını bul."""
        swings = []
        for i in range(3, len(prices) - 3):
            if prices[i] == max(prices[i-3:i+4]):
                swings.append(prices[i])
        return swings[-5:] if len(swings) > 5 else swings
    
    def _find_swing_lows(self, prices: np.ndarray) -> List[float]:
        """Swing low noktalarını bul."""
        swings = []
        for i in range(3, len(prices) - 3):
            if prices[i] == min(prices[i-3:i+4]):
                swings.append(prices[i])
        return swings[-5:] if len(swings) > 5 else swings
    
    def _determine_trend(self, highs: List[float], lows: List[float]) -> str:
        """Trend belirleme."""
        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"
        
        hh = highs[-1] > highs[-2]  # Higher High
        hl = lows[-1] > lows[-2]   # Higher Low
        lh = highs[-1] < highs[-2]  # Lower High
        ll = lows[-1] < lows[-2]   # Lower Low
        
        if hh and hl:
            return "BULLISH"
        elif lh and ll:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _check_bos(self, prices: np.ndarray, highs: List[float], lows: List[float]) -> Tuple[bool, str]:
        """Break of Structure kontrolü."""
        current_price = prices[-1]
        
        if highs and current_price > highs[-1]:
            return True, "BULLISH"
        elif lows and current_price < lows[-1]:
            return True, "BEARISH"
        
        return False, ""
    
    def _check_choch(self, highs: List[float], lows: List[float], current_trend: str) -> bool:
        """Change of Character kontrolü."""
        if len(highs) < 3 or len(lows) < 3:
            return False
        
        # Trend değişimi kontrolü
        if current_trend == "BULLISH" and highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            return True  # Bearish CHoCH
        elif current_trend == "BEARISH" and highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            return True  # Bullish CHoCH
        
        return False
    
    def _find_key_levels(self, highs: List[float], lows: List[float]) -> List[float]:
        """Önemli seviyeleri bul."""
        levels = list(set(highs + lows))
        levels.sort()
        return levels[-5:] if len(levels) > 5 else levels


# =============================================================================
# UNIFIED PATTERN ANALYZER
# =============================================================================

class AdvancedPatternAnalyzer:
    """Tüm pattern analizlerini birleştiren sınıf."""
    
    def __init__(self):
        self.elliott = ElliottWaveDetector()
        self.harmonic = HarmonicPatternScanner()
        self.structure = MarketStructureAnalyzer()
    
    async def analyze(self, prices: List[float], highs: List[float] = None, lows: List[float] = None) -> Dict:
        """
        Kapsamlı pattern analizi.
        
        Returns:
            {
                'elliott_wave': ElliottWaveResult,
                'harmonic_patterns': [HarmonicPattern, ...],
                'market_structure': MarketStructure,
                'combined_signal': str,
                'confidence': float
            }
        """
        # Elliott Wave
        elliott_result = self.elliott.analyze(prices, highs, lows)
        
        # Harmonic Patterns
        harmonic_patterns = self.harmonic.scan(prices, highs, lows)
        
        # Market Structure
        structure_result = self.structure.analyze(prices, highs, lows)
        
        # Combined signal
        combined_signal, confidence = self._combine_signals(
            elliott_result, harmonic_patterns, structure_result
        )
        
        return {
            'elliott_wave': elliott_result.to_dict(),
            'harmonic_patterns': [p.to_dict() for p in harmonic_patterns],
            'market_structure': structure_result.to_dict(),
            'combined_signal': combined_signal,
            'confidence': confidence
        }
    
    def _combine_signals(self, elliott: ElliottWaveResult, harmonics: List[HarmonicPattern], structure: MarketStructure) -> Tuple[str, float]:
        """Sinyalleri birleştir."""
        signals = []
        
        # Elliott Wave sinyali
        if elliott.is_valid:
            if elliott.wave_direction == "UP" and elliott.current_wave in ["1", "3", "5"]:
                signals.append(("BULLISH", elliott.confidence))
            elif elliott.wave_direction == "DOWN":
                signals.append(("BEARISH", elliott.confidence))
        
        # Harmonic pattern sinyalleri
        for pattern in harmonics:
            direction = "BULLISH" if pattern.is_bullish else "BEARISH"
            signals.append((direction, pattern.confidence))
        
        # Market Structure sinyali
        if structure.trend == "BULLISH":
            signals.append(("BULLISH", 60))
        elif structure.trend == "BEARISH":
            signals.append(("BEARISH", 60))
        
        # BOS/CHoCH boost
        if structure.bos_detected:
            signals.append((structure.bos_direction, 70))
        
        if not signals:
            return "NEUTRAL", 0.0
        
        # Çoğunluk kararı
        bullish_count = sum(1 for s, _ in signals if s == "BULLISH")
        bearish_count = sum(1 for s, _ in signals if s == "BEARISH")
        
        if bullish_count > bearish_count:
            avg_conf = sum(c for s, c in signals if s == "BULLISH") / bullish_count
            return "BULLISH", avg_conf
        elif bearish_count > bullish_count:
            avg_conf = sum(c for s, c in signals if s == "BEARISH") / bearish_count
            return "BEARISH", avg_conf
        else:
            return "NEUTRAL", 50.0


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_analyzer: Optional[AdvancedPatternAnalyzer] = None

def get_advanced_pattern_analyzer() -> AdvancedPatternAnalyzer:
    """Get or create pattern analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AdvancedPatternAnalyzer()
    return _analyzer


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    prices = [100, 105, 102, 110, 107, 115, 112, 120, 116, 125, 
              121, 118, 115, 120, 117, 122, 119, 125, 121, 128,
              123, 130, 126, 132, 128, 135, 130, 138, 133, 140]
    
    analyzer = get_advanced_pattern_analyzer()
    result = asyncio.run(analyzer.analyze(prices))
    
    print("Elliott Wave:", result['elliott_wave'])
    print("Market Structure:", result['market_structure'])
    print("Combined Signal:", result['combined_signal'])
    print("Confidence:", result['confidence'])
