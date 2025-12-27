"""
DEMIR AI - PATTERN RECOGNITION ENGINE
Wyckoff, Smart Money Concepts, Elliott Wave Detection

AI-destekli price action analizi
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("PATTERN_ENGINE")

class WyckoffPhase(Enum):
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "UNKNOWN"

class PatternRecognition:
    """
    PATTERN TANIMA MOTORU
    
    1. Wyckoff Phase Detection
    2. Smart Money Concepts (Order Blocks, FVG)
    3. Support/Resistance Clustering
    4. Trend Structure Analysis
    """
    
    def __init__(self):
        self.swing_threshold = 0.02  # %2 swing için minimum hareket

    async def analyze(self, symbol: str = "BTCUSDT") -> Dict:
        """Standard interface for Brain Modules (Async wrapper)"""
        # Pattern engine is synchronous but we wrap it for asyncio.gather
        # Need to fetch data first or assume data is passed?
        # IMPORTANT: Pattern engine needs DATAFRAME, not symbol.
        # So we must fetch data here or refactor.
        # For robustness, we will fetch 1h candles here.
        import aiohttp
        import pandas as pd
        
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200: return {}
                    data = await resp.json()
                    
            df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            return self.get_full_pattern_analysis(df)
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {}

    def detect_wyckoff_phase(self, df: pd.DataFrame) -> Dict:
        """
        Wyckoff Fazı Tespiti
        
        Accumulation: Düşük volatilite, düşen hacim, range içinde
        Markup: Yükselen fiyat, artan hacim
        Distribution: Yüksek fiyatta range, düşen hacim
        Markdown: Düşen fiyat, artan hacim
        """
        if len(df) < 50:
            return {'phase': WyckoffPhase.UNKNOWN.value, 'confidence': 0}
        
        # Son 50 bar
        recent = df.tail(50).copy()
        older = df.tail(100).head(50).copy() if len(df) >= 100 else recent
        
        # Fiyat trendi
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        
        # Volatilite (ATR/Price)
        if 'atr' in recent.columns:
            volatility = recent['atr'].mean() / recent['close'].mean()
        else:
            volatility = recent['close'].std() / recent['close'].mean()
        
        # Hacim trendi
        recent_vol = recent['volume'].mean()
        older_vol = older['volume'].mean() if len(older) > 0 else recent_vol
        volume_change = (recent_vol - older_vol) / max(older_vol, 1)
        
        # Range detection
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean()
        is_ranging = price_range < 0.12  # Relaxed to 12%
        
        # Faz belirleme
        confidence = 0
        
        if is_ranging and volume_change < 0.0 and abs(price_change) < 0.05:
            # Düşük volatilite, düşen/yatay hacim, yatay = Accumulation veya Distribution
            if recent['close'].mean() < older['close'].mean():
                phase = WyckoffPhase.ACCUMULATION
                confidence = 0.65
            else:
                phase = WyckoffPhase.DISTRIBUTION
                confidence = 0.65
                
        elif price_change > 0.03 and volume_change > -0.1: # Relaxed Markup
            phase = WyckoffPhase.MARKUP
            confidence = min(0.9, 0.5 + price_change + (volume_change * 0.2))
            
        elif price_change < -0.03 and volume_change > -0.1: # Relaxed Markdown
            phase = WyckoffPhase.MARKDOWN
            confidence = min(0.9, 0.5 + abs(price_change) + (volume_change * 0.2))
            
        else:
            phase = WyckoffPhase.UNKNOWN
            # Fallback: Simple Trend Check if unknown
            if price_change > 0.05: phase = WyckoffPhase.MARKUP 
            elif price_change < -0.05: phase = WyckoffPhase.MARKDOWN
            confidence = 0.4
        
        # Trading implication
        if phase == WyckoffPhase.ACCUMULATION:
            implication = "DIP_BUY_ZONE"
            bias = "BULLISH"
        elif phase == WyckoffPhase.MARKUP:
            implication = "TREND_FOLLOW_LONG"
            bias = "BULLISH"
        elif phase == WyckoffPhase.DISTRIBUTION:
            implication = "TAKE_PROFIT_ZONE"
            bias = "BEARISH"
        elif phase == WyckoffPhase.MARKDOWN:
            implication = "AVOID_OR_SHORT"
            bias = "BEARISH"
        else:
            implication = "WAIT"
            bias = "NEUTRAL"
        
        result = {
            'phase': phase.value,
            'confidence': round(confidence, 2),
            'price_change_pct': round(price_change * 100, 2),
            'volume_change_pct': round(volume_change * 100, 2),
            'volatility': round(volatility * 100, 2),
            'is_ranging': is_ranging,
            'implication': implication,
            'bias': bias
        }
        
        logger.info(f"📊 Wyckoff: {phase.value} ({confidence:.0%}) | {implication}")
        return result
    
    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """
        Smart Money Concept: Order Block Detection
        
        Bullish OB: Son düşüş öncesi son bullish mum
        Bearish OB: Son yükseliş öncesi son bearish mum
        """
        if len(df) < lookback:
            return {'bullish_obs': [], 'bearish_obs': []}
        
        recent = df.tail(lookback).copy()
        recent = recent.reset_index(drop=True)
        
        bullish_obs = []
        bearish_obs = []
        
        for i in range(3, len(recent) - 3):
            # Mum yönü
            is_bullish = recent.loc[i, 'close'] > recent.loc[i, 'open']
            
            # Sonraki harekete bak
            future_high = recent.loc[i+1:i+4, 'high'].max()
            future_low = recent.loc[i+1:i+4, 'low'].min()
            current_close = recent.loc[i, 'close']
            
            # Bullish Order Block: Bullish mum + sonra yukarı hareket
            if is_bullish:
                move_up = (future_high - current_close) / current_close
                if move_up > 0.02:  # %2+ yukarı hareket
                    bullish_obs.append({
                        'high': recent.loc[i, 'high'],
                        'low': recent.loc[i, 'low'],
                        'strength': min(1.0, move_up * 10),
                        'index': i
                    })
            
            # Bearish Order Block: Bearish mum + sonra aşağı hareket
            else:
                move_down = (current_close - future_low) / current_close
                if move_down > 0.02:
                    bearish_obs.append({
                        'high': recent.loc[i, 'high'],
                        'low': recent.loc[i, 'low'],
                        'strength': min(1.0, move_down * 10),
                        'index': i
                    })
        
        # En güçlü OB'ları seç
        bullish_obs.sort(key=lambda x: x['strength'], reverse=True)
        bearish_obs.sort(key=lambda x: x['strength'], reverse=True)
        
        current_price = df['close'].iloc[-1]
        
        # Güncel fiyata yakın OB'ları işaretle
        active_bullish = [ob for ob in bullish_obs[:3] if ob['high'] < current_price]
        active_bearish = [ob for ob in bearish_obs[:3] if ob['low'] > current_price]
        
        result = {
            'bullish_order_blocks': bullish_obs[:5],
            'bearish_order_blocks': bearish_obs[:5],
            'active_support_ob': active_bullish[0] if active_bullish else None,
            'active_resistance_ob': active_bearish[0] if active_bearish else None,
            'current_price': current_price
        }
        
        logger.info(f"🧱 Order Blocks: {len(bullish_obs)} Bullish, {len(bearish_obs)} Bearish")
        return result
    
    def detect_fair_value_gaps(self, df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
        """
        Fair Value Gap (FVG) / Imbalance Detection
        
        FVG: Üç ardışık mumda orta mumun gövdesinin 
        önceki ve sonraki mumların fitilleriyle örtüşmediği bölge.
        """
        if len(df) < lookback:
            return []
        
        recent = df.tail(lookback).copy()
        recent = recent.reset_index(drop=True)
        
        fvgs = []
        
        for i in range(1, len(recent) - 1):
            prev_high = recent.loc[i-1, 'high']
            prev_low = recent.loc[i-1, 'low']
            next_high = recent.loc[i+1, 'high']
            next_low = recent.loc[i+1, 'low']
            
            # Bullish FVG: Önceki mumun HIGH'ı, sonraki mumun LOW'undan düşük
            if prev_high < next_low:
                gap_size = (next_low - prev_high) / recent.loc[i, 'close']
                if gap_size > 0.002:  # Minimum %0.2 gap
                    fvgs.append({
                        'type': 'BULLISH',
                        'top': next_low,
                        'bottom': prev_high,
                        'size_pct': gap_size * 100,
                        'filled': False,
                        'index': i
                    })
            
            # Bearish FVG: Önceki mumun LOW'u, sonraki mumun HIGH'ından yüksek
            if prev_low > next_high:
                gap_size = (prev_low - next_high) / recent.loc[i, 'close']
                if gap_size > 0.002:
                    fvgs.append({
                        'type': 'BEARISH',
                        'top': prev_low,
                        'bottom': next_high,
                        'size_pct': gap_size * 100,
                        'filled': False,
                        'index': i
                    })
        
        # Doldurulmuş FVG'leri işaretle
        current_price = df['close'].iloc[-1]
        for fvg in fvgs:
            if fvg['type'] == 'BULLISH' and current_price <= fvg['top']:
                fvg['filled'] = True
            elif fvg['type'] == 'BEARISH' and current_price >= fvg['bottom']:
                fvg['filled'] = True
        
        # Doldurulmamış FVG'ler target olabilir
        unfilled = [f for f in fvgs if not f['filled']]
        
        logger.info(f"📐 FVG: {len(unfilled)} unfilled gaps found")
        return unfilled
    
    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Piyasa Yapısı Analizi
        
        Higher Highs + Higher Lows = Uptrend
        Lower Highs + Lower Lows = Downtrend
        Mixed = Range/Consolidation
        """
        if len(df) < 20:
            return {'structure': 'UNKNOWN', 'trend': 'NEUTRAL'}
        
        recent = df.tail(50).copy()
        
        # Swing High/Low tespiti
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent) - 2):
            # Swing High: Ortadaki en yüksek
            if (recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and 
                recent.iloc[i]['high'] > recent.iloc[i-2]['high'] and
                recent.iloc[i]['high'] > recent.iloc[i+1]['high'] and 
                recent.iloc[i]['high'] > recent.iloc[i+2]['high']):
                swing_highs.append({'price': recent.iloc[i]['high'], 'index': i})
            
            # Swing Low: Ortadaki en düşük
            if (recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and 
                recent.iloc[i]['low'] < recent.iloc[i-2]['low'] and
                recent.iloc[i]['low'] < recent.iloc[i+1]['low'] and 
                recent.iloc[i]['low'] < recent.iloc[i+2]['low']):
                swing_lows.append({'price': recent.iloc[i]['low'], 'index': i})
        
        # Trend structure
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price']  # Higher High
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price']    # Higher Low
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price']  # Lower High
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price']    # Lower Low
            
            if hh and hl:
                structure = "UPTREND"
                trend = "BULLISH"
                strength = 0.8
            elif lh and ll:
                structure = "DOWNTREND"
                trend = "BEARISH"
                strength = 0.8
            elif hh and ll:
                structure = "EXPANSION"
                trend = "VOLATILE"
                strength = 0.5
            elif lh and hl:
                structure = "CONTRACTION"
                trend = "RANGE"
                strength = 0.5
            else:
                structure = "TRANSITIONING"
                trend = "NEUTRAL"
                strength = 0.3
        else:
            structure = "INSUFFICIENT_DATA"
            trend = "NEUTRAL"
            strength = 0.2
        
        # Break of Structure (BoS) detection
        current_price = df['close'].iloc[-1]
        bos = None
        
        if swing_lows and current_price < swing_lows[-1]['price']:
            bos = "BEARISH_BOS"
        elif swing_highs and current_price > swing_highs[-1]['price']:
            bos = "BULLISH_BOS"
        
        result = {
            'structure': structure,
            'trend': trend,
            'strength': strength,
            'swing_highs': swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs,
            'swing_lows': swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows,
            'break_of_structure': bos,
            'current_price': current_price
        }
        
        logger.info(f"📈 Structure: {structure} | Trend: {trend} | BoS: {bos}")
        return result
    
    def get_full_pattern_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Tüm pattern analizlerini birleştir.
        """
        wyckoff = self.detect_wyckoff_phase(df)
        order_blocks = self.detect_order_blocks(df)
        fvgs = self.detect_fair_value_gaps(df)
        structure = self.detect_market_structure(df)
        
        # Composite bias
        bullish_signals = 0
        bearish_signals = 0
        
        if wyckoff['bias'] == 'BULLISH':
            bullish_signals += 2
        elif wyckoff['bias'] == 'BEARISH':
            bearish_signals += 2
        
        if structure['trend'] == 'BULLISH':
            bullish_signals += 2
        elif structure['trend'] == 'BEARISH':
            bearish_signals += 2
        
        if structure.get('break_of_structure') == 'BULLISH_BOS':
            bullish_signals += 1
        elif structure.get('break_of_structure') == 'BEARISH_BOS':
            bearish_signals += 1
        
        # Active OB
        if order_blocks.get('active_support_ob'):
            bullish_signals += 1
        if order_blocks.get('active_resistance_ob'):
            bearish_signals += 1
        
        # Final bias
        if bullish_signals > bearish_signals + 2:
            final_bias = "STRONG_BULLISH"
        elif bullish_signals > bearish_signals:
            final_bias = "BULLISH"
        elif bearish_signals > bullish_signals + 2:
            final_bias = "STRONG_BEARISH"
        elif bearish_signals > bullish_signals:
            final_bias = "BEARISH"
        else:
            final_bias = "NEUTRAL"
        
        return {
            'wyckoff': wyckoff,
            'order_blocks': order_blocks,
            'fair_value_gaps': fvgs,
            'market_structure': structure,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'final_bias': final_bias
        }


# Global Instance
_pattern_engine = None

def get_pattern_engine() -> PatternRecognition:
    """Get or create singleton PatternRecognition"""
    global _pattern_engine
    if _pattern_engine is None:
        _pattern_engine = PatternRecognition()
    return _pattern_engine
