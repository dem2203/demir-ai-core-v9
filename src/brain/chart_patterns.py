"""
DEMIR AI - Advanced Chart Pattern Recognition
Grafik formasyonlarını otomatik tespit eder ve fırsat/risk bildirir.

PHASE 38: Visual Pattern Analysis
1. Wedges (Rising/Falling)
2. Triangles (Ascending/Descending/Symmetrical)
3. Double Top/Bottom
4. Head & Shoulders
5. Flags & Pennants
6. Cup & Handle
7. Channel Detection
"""
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.signal import argrelextrema

logger = logging.getLogger("CHART_PATTERNS")


@dataclass
class ChartPattern:
    """Grafik formasyonu"""
    pattern_type: str      # WEDGE, TRIANGLE, DOUBLE_TOP, HEAD_SHOULDERS, FLAG, CHANNEL
    sub_type: str          # RISING, FALLING, ASCENDING, etc.
    direction: str         # BULLISH, BEARISH
    reliability: float     # 0-100 güvenilirlik skoru
    target_price: float    # Hedef fiyat
    stop_loss: float       # Stop loss seviyesi
    description: str       # Türkçe açıklama
    action: str            # Önerilen aksiyon
    detected_at: datetime


class ChartPatternAnalyzer:
    """
    Grafik Formasyonu Analizörü
    
    Mum verilerinden otomatik olarak:
    - Destek/Direnç çizer
    - Trend çizgileri oluşturur
    - Formasyonları tespit eder
    - Fırsat/Risk bildirimi yapar
    """
    
    def __init__(self):
        self.min_pattern_bars = 10  # Minimum bar sayısı
        self.lookback = 100  # Analiz için geriye bakış
        
    def analyze_all_patterns(self, ohlcv_data: List[Dict], symbol: str = 'BTC/USDT') -> List[ChartPattern]:
        """
        Tüm grafik formasyonlarını analiz et.
        
        Args:
            ohlcv_data: [{'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}, ...]
        """
        patterns = []
        
        if len(ohlcv_data) < self.min_pattern_bars:
            return patterns
        
        # Extract price arrays
        highs = np.array([d['high'] for d in ohlcv_data])
        lows = np.array([d['low'] for d in ohlcv_data])
        closes = np.array([d['close'] for d in ohlcv_data])
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(highs, order=5, mode='high')
        swing_lows = self._find_swing_points(lows, order=5, mode='low')
        
        # 1. Wedge Detection
        wedge = self._detect_wedge(highs, lows, closes, swing_highs, swing_lows, symbol)
        if wedge:
            patterns.append(wedge)
        
        # 2. Triangle Detection
        triangle = self._detect_triangle(highs, lows, closes, swing_highs, swing_lows, symbol)
        if triangle:
            patterns.append(triangle)
        
        # 3. Double Top/Bottom
        double = self._detect_double_pattern(highs, lows, closes, swing_highs, swing_lows, symbol)
        if double:
            patterns.append(double)
        
        # 4. Head & Shoulders
        hs = self._detect_head_shoulders(highs, lows, closes, swing_highs, swing_lows, symbol)
        if hs:
            patterns.append(hs)
        
        # 5. Channel Detection
        channel = self._detect_channel(highs, lows, closes, symbol)
        if channel:
            patterns.append(channel)
        
        return patterns
    
    def _find_swing_points(self, data: np.ndarray, order: int = 5, mode: str = 'high') -> List[Tuple[int, float]]:
        """Swing high/low noktalarını bul"""
        if mode == 'high':
            indices = argrelextrema(data, np.greater, order=order)[0]
        else:
            indices = argrelextrema(data, np.less, order=order)[0]
        
        return [(int(i), float(data[i])) for i in indices]
    
    # =========================================
    # 1. WEDGE DETECTION
    # =========================================
    def _detect_wedge(self, highs, lows, closes, swing_highs, swing_lows, symbol) -> Optional[ChartPattern]:
        """
        Kama formasyonu tespiti.
        
        Rising Wedge: Yükselen kama - BEARISH (düşüş sinyali)
        Falling Wedge: Düşen kama - BULLISH (yükseliş sinyali)
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return None
        
        try:
            # Get recent swing points
            recent_highs = swing_highs[-5:]
            recent_lows = swing_lows[-5:]
            
            # Fit trend lines
            high_indices = [h[0] for h in recent_highs]
            high_values = [h[1] for h in recent_highs]
            low_indices = [l[0] for l in recent_lows]
            low_values = [l[1] for l in recent_lows]
            
            if len(high_indices) < 2 or len(low_indices) < 2:
                return None
            
            # Calculate slopes
            high_slope, _, _, _, _ = stats.linregress(high_indices, high_values)
            low_slope, _, _, _, _ = stats.linregress(low_indices, low_values)
            
            current_price = closes[-1]
            
            # Rising Wedge: Both slopes positive, converging
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                height = max(high_values) - min(low_values)
                target = current_price - height * 0.618  # Fibonacci retracement
                
                return ChartPattern(
                    pattern_type='WEDGE',
                    sub_type='RISING',
                    direction='BEARISH',
                    reliability=65,
                    target_price=target,
                    stop_loss=max(high_values) * 1.02,  # 2% above highest high
                    description=f"🔻 {symbol} Yükselen Kama (Rising Wedge) tespit edildi! Bu pattern genellikle düşüşle sonuçlanır.",
                    action="⚠️ Short pozisyon düşün veya long'lardan kar al. Kırılımı bekle!",
                    detected_at=datetime.now()
                )
            
            # Falling Wedge: Both slopes negative, converging
            elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
                height = max(high_values) - min(low_values)
                target = current_price + height * 0.618
                
                return ChartPattern(
                    pattern_type='WEDGE',
                    sub_type='FALLING',
                    direction='BULLISH',
                    reliability=70,
                    target_price=target,
                    stop_loss=min(low_values) * 0.98,  # 2% below lowest low
                    description=f"🔺 {symbol} Düşen Kama (Falling Wedge) tespit edildi! Bu pattern genellikle yükselişle sonuçlanır.",
                    action="✅ Long pozisyon için hazırlan. Yukarı kırılımı bekle!",
                    detected_at=datetime.now()
                )
                
        except Exception as e:
            logger.debug(f"Wedge detection error: {e}")
        
        return None
    
    # =========================================
    # 2. TRIANGLE DETECTION
    # =========================================
    def _detect_triangle(self, highs, lows, closes, swing_highs, swing_lows, symbol) -> Optional[ChartPattern]:
        """
        Üçgen formasyonu tespiti.
        
        Ascending Triangle: Düz üst, yükselen alt - BULLISH
        Descending Triangle: Yükselen üst, düz alt - BEARISH
        Symmetrical Triangle: Daralan - Yön belirsiz
        """
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return None
        
        try:
            recent_highs = swing_highs[-5:]
            recent_lows = swing_lows[-5:]
            
            high_indices = [h[0] for h in recent_highs]
            high_values = [h[1] for h in recent_highs]
            low_indices = [l[0] for l in recent_lows]
            low_values = [l[1] for l in recent_lows]
            
            high_slope, _, r_high, _, _ = stats.linregress(high_indices, high_values)
            low_slope, _, r_low, _, _ = stats.linregress(low_indices, low_values)
            
            current_price = closes[-1]
            resistance = max(high_values)
            support = min(low_values)
            height = resistance - support
            
            # Ascending Triangle: Flat top, rising bottom
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                target = resistance + height * 0.75
                
                return ChartPattern(
                    pattern_type='TRIANGLE',
                    sub_type='ASCENDING',
                    direction='BULLISH',
                    reliability=75,
                    target_price=target,
                    stop_loss=current_price - height * 0.5,
                    description=f"📐 {symbol} Yükselen Üçgen (Ascending Triangle)! Direnç: ${resistance:,.0f}",
                    action=f"✅ ${resistance:,.0f} üzerine kırılımda LONG! Hedef: ${target:,.0f}",
                    detected_at=datetime.now()
                )
            
            # Descending Triangle: Flat bottom, falling top
            elif abs(low_slope) < 0.001 and high_slope < -0.001:
                target = support - height * 0.75
                
                return ChartPattern(
                    pattern_type='TRIANGLE',
                    sub_type='DESCENDING',
                    direction='BEARISH',
                    reliability=75,
                    target_price=target,
                    stop_loss=current_price + height * 0.5,
                    description=f"📐 {symbol} Düşen Üçgen (Descending Triangle)! Destek: ${support:,.0f}",
                    action=f"⚠️ ${support:,.0f} altına kırılımda SHORT! Hedef: ${target:,.0f}",
                    detected_at=datetime.now()
                )
            
            # Symmetrical Triangle: Converging
            elif high_slope < 0 and low_slope > 0:
                # Direction uncertain, wait for breakout
                return ChartPattern(
                    pattern_type='TRIANGLE',
                    sub_type='SYMMETRICAL',
                    direction='NEUTRAL',
                    reliability=60,
                    target_price=current_price,  # Unknown until breakout
                    stop_loss=current_price,
                    description=f"📐 {symbol} Simetrik Üçgen! Kırılım bekleniyor.",
                    action="⏳ Yön belirsiz - Kırılımı bekle! Büyük hareket yakın.",
                    detected_at=datetime.now()
                )
                
        except Exception as e:
            logger.debug(f"Triangle detection error: {e}")
        
        return None
    
    # =========================================
    # 3. DOUBLE TOP/BOTTOM
    # =========================================
    def _detect_double_pattern(self, highs, lows, closes, swing_highs, swing_lows, symbol) -> Optional[ChartPattern]:
        """
        Çift tepe/dip formasyonu.
        
        Double Top: M şekli - BEARISH
        Double Bottom: W şekli - BULLISH
        """
        try:
            current_price = closes[-1]
            
            # Double Top: Look for two similar highs
            if len(swing_highs) >= 2:
                last_highs = swing_highs[-4:]
                for i in range(len(last_highs) - 1):
                    h1, h2 = last_highs[i], last_highs[i + 1]
                    # Check if highs are within 1% of each other
                    if abs(h1[1] - h2[1]) / h1[1] < 0.01 and h2[0] - h1[0] > 5:
                        # Find neckline (low between the two peaks)
                        neckline_lows = [l[1] for l in swing_lows if h1[0] < l[0] < h2[0]]
                        if neckline_lows:
                            neckline = min(neckline_lows)
                            height = h1[1] - neckline
                            target = neckline - height
                            
                            if current_price < neckline * 1.02:  # Near or below neckline
                                return ChartPattern(
                                    pattern_type='DOUBLE_TOP',
                                    sub_type='M_PATTERN',
                                    direction='BEARISH',
                                    reliability=70,
                                    target_price=target,
                                    stop_loss=max(h1[1], h2[1]) * 1.01,
                                    description=f"📉 {symbol} Çift Tepe (Double Top)! Boyun çizgisi: ${neckline:,.0f}",
                                    action=f"⚠️ BEARISH! ${neckline:,.0f} altına kırılımda hedef: ${target:,.0f}",
                                    detected_at=datetime.now()
                                )
            
            # Double Bottom: Look for two similar lows
            if len(swing_lows) >= 2:
                last_lows = swing_lows[-4:]
                for i in range(len(last_lows) - 1):
                    l1, l2 = last_lows[i], last_lows[i + 1]
                    if abs(l1[1] - l2[1]) / l1[1] < 0.01 and l2[0] - l1[0] > 5:
                        # Find neckline (high between the two bottoms)
                        neckline_highs = [h[1] for h in swing_highs if l1[0] < h[0] < l2[0]]
                        if neckline_highs:
                            neckline = max(neckline_highs)
                            height = neckline - l1[1]
                            target = neckline + height
                            
                            if current_price > neckline * 0.98:  # Near or above neckline
                                return ChartPattern(
                                    pattern_type='DOUBLE_BOTTOM',
                                    sub_type='W_PATTERN',
                                    direction='BULLISH',
                                    reliability=70,
                                    target_price=target,
                                    stop_loss=min(l1[1], l2[1]) * 0.99,
                                    description=f"📈 {symbol} Çift Dip (Double Bottom)! Boyun çizgisi: ${neckline:,.0f}",
                                    action=f"✅ BULLISH! ${neckline:,.0f} üstüne kırılımda hedef: ${target:,.0f}",
                                    detected_at=datetime.now()
                                )
                                
        except Exception as e:
            logger.debug(f"Double pattern detection error: {e}")
        
        return None
    
    # =========================================
    # 4. HEAD & SHOULDERS
    # =========================================
    def _detect_head_shoulders(self, highs, lows, closes, swing_highs, swing_lows, symbol) -> Optional[ChartPattern]:
        """
        Baş-Omuz formasyonu.
        
        Regular H&S: BEARISH (düşüş)
        Inverse H&S: BULLISH (yükseliş)
        """
        try:
            current_price = closes[-1]
            
            # Regular Head & Shoulders (bearish)
            if len(swing_highs) >= 3:
                last_highs = swing_highs[-5:]
                for i in range(len(last_highs) - 2):
                    left_shoulder = last_highs[i]
                    head = last_highs[i + 1]
                    right_shoulder = last_highs[i + 2]
                    
                    # Head should be higher than both shoulders
                    # Shoulders should be roughly equal
                    if (head[1] > left_shoulder[1] and head[1] > right_shoulder[1] and
                        abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.03):
                        
                        # Find neckline
                        neckline_points = [l[1] for l in swing_lows 
                                         if left_shoulder[0] < l[0] < right_shoulder[0]]
                        if neckline_points:
                            neckline = min(neckline_points)
                            height = head[1] - neckline
                            target = neckline - height
                            
                            if current_price < neckline * 1.05:  # Near neckline
                                return ChartPattern(
                                    pattern_type='HEAD_SHOULDERS',
                                    sub_type='REGULAR',
                                    direction='BEARISH',
                                    reliability=80,
                                    target_price=target,
                                    stop_loss=head[1] * 1.01,
                                    description=f"👤 {symbol} Baş-Omuz Formasyonu! Çok güçlü düşüş sinyali!",
                                    action=f"🔴 ${neckline:,.0f} altına kırılırsa hedef: ${target:,.0f}",
                                    detected_at=datetime.now()
                                )
            
            # Inverse Head & Shoulders (bullish) - Check swing lows
            if len(swing_lows) >= 3:
                last_lows = swing_lows[-5:]
                for i in range(len(last_lows) - 2):
                    left_shoulder = last_lows[i]
                    head = last_lows[i + 1]
                    right_shoulder = last_lows[i + 2]
                    
                    # Head should be lower than both shoulders
                    if (head[1] < left_shoulder[1] and head[1] < right_shoulder[1] and
                        abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.03):
                        
                        neckline_points = [h[1] for h in swing_highs 
                                         if left_shoulder[0] < h[0] < right_shoulder[0]]
                        if neckline_points:
                            neckline = max(neckline_points)
                            height = neckline - head[1]
                            target = neckline + height
                            
                            if current_price > neckline * 0.95:
                                return ChartPattern(
                                    pattern_type='HEAD_SHOULDERS',
                                    sub_type='INVERSE',
                                    direction='BULLISH',
                                    reliability=80,
                                    target_price=target,
                                    stop_loss=head[1] * 0.99,
                                    description=f"👤 {symbol} Ters Baş-Omuz! Çok güçlü yükseliş sinyali!",
                                    action=f"🟢 ${neckline:,.0f} üstüne kırılırsa hedef: ${target:,.0f}",
                                    detected_at=datetime.now()
                                )
                                
        except Exception as e:
            logger.debug(f"Head & Shoulders detection error: {e}")
        
        return None
    
    # =========================================
    # 5. CHANNEL DETECTION
    # =========================================
    def _detect_channel(self, highs, lows, closes, symbol) -> Optional[ChartPattern]:
        """
        Kanal formasyonu tespiti.
        
        Ascending Channel: BULLISH trend
        Descending Channel: BEARISH trend
        Horizontal Channel: Range bound
        """
        try:
            n = min(50, len(closes))
            indices = np.arange(n)
            recent_highs = highs[-n:]
            recent_lows = lows[-n:]
            recent_closes = closes[-n:]
            current_price = closes[-1]
            
            # Fit trend lines to highs and lows
            high_slope, high_intercept, r_high, _, _ = stats.linregress(indices, recent_highs)
            low_slope, low_intercept, r_low, _, _ = stats.linregress(indices, recent_lows)
            
            # Check if slopes are parallel (channel)
            if abs(r_high) > 0.7 and abs(r_low) > 0.7:  # Strong correlation
                slope_diff = abs(high_slope - low_slope)
                avg_slope = (high_slope + low_slope) / 2
                
                if slope_diff / abs(avg_slope + 0.001) < 0.3:  # Parallel lines
                    channel_width = (recent_highs[-1] - recent_lows[-1])
                    channel_mid = (recent_highs[-1] + recent_lows[-1]) / 2
                    
                    # Ascending Channel
                    if avg_slope > 0.001:
                        return ChartPattern(
                            pattern_type='CHANNEL',
                            sub_type='ASCENDING',
                            direction='BULLISH',
                            reliability=65,
                            target_price=recent_highs[-1] + high_slope * 10,  # Project 10 bars
                            stop_loss=recent_lows[-1] * 0.98,
                            description=f"📊 {symbol} Yükselen Kanal! Trend yukarı.",
                            action="✅ Kanal altında alım, üstünde satım yap. Trend devam ediyor.",
                            detected_at=datetime.now()
                        )
                    
                    # Descending Channel
                    elif avg_slope < -0.001:
                        return ChartPattern(
                            pattern_type='CHANNEL',
                            sub_type='DESCENDING',
                            direction='BEARISH',
                            reliability=65,
                            target_price=recent_lows[-1] + low_slope * 10,
                            stop_loss=recent_highs[-1] * 1.02,
                            description=f"📊 {symbol} Düşen Kanal! Trend aşağı.",
                            action="⚠️ Kanal üstünde short, altında cover yap. Trend devam ediyor.",
                            detected_at=datetime.now()
                        )
                    
                    # Horizontal Channel (Range)
                    else:
                        return ChartPattern(
                            pattern_type='CHANNEL',
                            sub_type='HORIZONTAL',
                            direction='NEUTRAL',
                            reliability=60,
                            target_price=channel_mid,
                            stop_loss=recent_lows[-1] * 0.97,
                            description=f"📊 {symbol} Yatay Kanal! Range: ${recent_lows[-1]:,.0f} - ${recent_highs[-1]:,.0f}",
                            action="↔️ Destek/dirençte işlem yap. Kırılımı bekle!",
                            detected_at=datetime.now()
                        )
                        
        except Exception as e:
            logger.debug(f"Channel detection error: {e}")
        
        return None
    
    # =========================================
    # FORMAT FOR TELEGRAM
    # =========================================
    def format_patterns_for_telegram(self, patterns: List[ChartPattern]) -> str:
        """Telegram için pattern'ları formatla"""
        if not patterns:
            return ""
        
        msg = "📐 *Grafik Formasyonları*\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for p in patterns:
            emoji = "🟢" if p.direction == 'BULLISH' else "🔴" if p.direction == 'BEARISH' else "⚪"
            reliability_bar = "█" * int(p.reliability / 20) + "░" * (5 - int(p.reliability / 20))
            
            msg += f"{emoji} *{p.pattern_type}* ({p.sub_type})\n"
            msg += f"_{p.description}_\n"
            msg += f"📊 Güvenilirlik: [{reliability_bar}] {p.reliability}%\n"
            msg += f"🎯 Hedef: `${p.target_price:,.0f}` | Stop: `${p.stop_loss:,.0f}`\n"
            msg += f"💡 {p.action}\n\n"
        
        msg += f"⏰ _{datetime.now().strftime('%H:%M:%S')}_"
        return msg
