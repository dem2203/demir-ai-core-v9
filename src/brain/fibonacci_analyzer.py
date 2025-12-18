# -*- coding: utf-8 -*-
"""
DEMIR AI - Fibonacci Analyzer
Fibonacci retracement ve extension seviyeleri.

PHASE 115: Fibonacci Analysis
- Fibonacci Retracement (0.236, 0.382, 0.5, 0.618, 0.786)
- Fibonacci Extension (1.272, 1.618, 2.0, 2.618)
- Otomatik swing high/low tespiti
- Destek/direnç olarak kullanım
"""
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("FIBONACCI")


@dataclass
class FibonacciLevel:
    """Fibonacci seviyesi."""
    ratio: float
    price: float
    level_type: str  # 'RETRACEMENT' or 'EXTENSION'
    is_support: bool
    is_resistance: bool
    distance_pct: float  # Mevcut fiyattan uzaklık


class FibonacciAnalyzer:
    """
    Fibonacci Analizörü
    
    Swing high/low noktalarından Fibonacci seviyelerini hesaplar.
    """
    
    # Fibonacci oranları
    RETRACEMENT_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    EXTENSION_RATIOS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618]
    
    def __init__(self):
        self.levels: List[FibonacciLevel] = []
        self.swing_high = 0
        self.swing_low = 0
        self.trend = 'UNKNOWN'
        logger.info("✅ Fibonacci Analyzer initialized")
    
    async def analyze(self, symbol: str = 'BTCUSDT', 
                     timeframe: str = '4h',
                     lookback: int = 100) -> Dict:
        """
        Fibonacci analizi yap.
        
        Args:
            symbol: Trading pair
            timeframe: Zaman dilimi (1h, 4h, 1d)
            lookback: Kaç mum geriye bak
            
        Returns:
            Dict with levels, trend, signals
        """
        try:
            # Veri çek
            df = await self._fetch_data(symbol, timeframe, lookback)
            if df.empty:
                return {'error': 'No data'}
            
            # Swing noktalarını bul
            self.swing_high, self.swing_low, self.trend = self._find_swing_points(df)
            
            if self.swing_high == 0 or self.swing_low == 0:
                return {'error': 'Could not find swing points'}
            
            # Mevcut fiyat
            current_price = df['close'].iloc[-1]
            
            # Fibonacci seviyelerini hesapla
            self.levels = self._calculate_levels(current_price)
            
            # En yakın seviyeleri bul
            nearest_support = self._find_nearest_level(current_price, 'support')
            nearest_resistance = self._find_nearest_level(current_price, 'resistance')
            
            # Sinyal oluştur
            signal = self._generate_signal(current_price, nearest_support, nearest_resistance)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'swing_high': self.swing_high,
                'swing_low': self.swing_low,
                'trend': self.trend,
                'levels': [self._level_to_dict(l) for l in self.levels],
                'nearest_support': self._level_to_dict(nearest_support) if nearest_support else None,
                'nearest_resistance': self._level_to_dict(nearest_resistance) if nearest_resistance else None,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fibonacci analysis failed: {e}")
            return {'error': str(e)}
    
    async def _fetch_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Veri çek."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': timeframe, 'limit': limit},
                timeout=10
            )
            
            if resp.status_code != 200:
                return pd.DataFrame()
            
            klines = resp.json()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close',
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return pd.DataFrame()
    
    def _find_swing_points(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """
        Swing high ve swing low noktalarını bul.
        
        Son 50 mumda en yüksek ve en düşük noktayı bulur.
        Trend yönünü belirler.
        """
        recent = df.tail(50)
        
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        # Hangi önce geldi?
        high_idx = recent['high'].idxmax()
        low_idx = recent['low'].idxmin()
        
        # Trend belirleme
        if high_idx > low_idx:
            trend = 'UPTREND'  # Düşük önce, yüksek sonra = yükseliş
        else:
            trend = 'DOWNTREND'  # Yüksek önce, düşük sonra = düşüş
        
        return swing_high, swing_low, trend
    
    def _calculate_levels(self, current_price: float) -> List[FibonacciLevel]:
        """Fibonacci seviyelerini hesapla."""
        levels = []
        
        diff = self.swing_high - self.swing_low
        
        # Retracement seviyeleri
        for ratio in self.RETRACEMENT_RATIOS:
            if self.trend == 'UPTREND':
                # Yukarı trend: yüksekten aşağı hesapla
                price = self.swing_high - (diff * ratio)
            else:
                # Aşağı trend: düşükten yukarı hesapla
                price = self.swing_low + (diff * ratio)
            
            distance_pct = ((price - current_price) / current_price) * 100
            
            levels.append(FibonacciLevel(
                ratio=ratio,
                price=price,
                level_type='RETRACEMENT',
                is_support=price < current_price,
                is_resistance=price > current_price,
                distance_pct=distance_pct
            ))
        
        # Extension seviyeleri (sadece trend yönünde)
        for ratio in self.EXTENSION_RATIOS[1:]:  # 1.0 zaten retracement'ta var
            if self.trend == 'UPTREND':
                price = self.swing_low + (diff * ratio)
            else:
                price = self.swing_high - (diff * ratio)
            
            # Sadece makul aralıktaki extension'ları ekle
            if abs((price - current_price) / current_price) > 0.5:  # %50'den uzağı alma
                continue
            
            distance_pct = ((price - current_price) / current_price) * 100
            
            levels.append(FibonacciLevel(
                ratio=ratio,
                price=price,
                level_type='EXTENSION',
                is_support=price < current_price,
                is_resistance=price > current_price,
                distance_pct=distance_pct
            ))
        
        return sorted(levels, key=lambda x: x.price)
    
    def _find_nearest_level(self, current_price: float, level_type: str) -> Optional[FibonacciLevel]:
        """En yakın destek veya direnci bul."""
        if level_type == 'support':
            candidates = [l for l in self.levels if l.is_support]
            if not candidates:
                return None
            return max(candidates, key=lambda x: x.price)  # En yakın destek
        else:
            candidates = [l for l in self.levels if l.is_resistance]
            if not candidates:
                return None
            return min(candidates, key=lambda x: x.price)  # En yakın direnç
    
    def _generate_signal(self, current_price: float, 
                        support: Optional[FibonacciLevel],
                        resistance: Optional[FibonacciLevel]) -> Dict:
        """Fibonacci sinyali oluştur."""
        signal = {
            'direction': 'NEUTRAL',
            'confidence': 40,
            'reason': ''
        }
        
        # Önemli Fibonacci seviyelerine yakınlık
        important_ratios = [0.382, 0.5, 0.618]
        
        for level in self.levels:
            if level.ratio in important_ratios:
                distance = abs(level.distance_pct)
                
                if distance < 1:  # %1'den yakın
                    if level.is_support and self.trend == 'UPTREND':
                        signal['direction'] = 'LONG'
                        signal['confidence'] = 70
                        signal['reason'] = f"Fib {level.ratio} desteğinde ({level.price:.0f})"
                    elif level.is_resistance and self.trend == 'DOWNTREND':
                        signal['direction'] = 'SHORT'
                        signal['confidence'] = 70
                        signal['reason'] = f"Fib {level.ratio} direncinde ({level.price:.0f})"
        
        return signal
    
    def _level_to_dict(self, level: FibonacciLevel) -> Dict:
        """FibonacciLevel'ı dict'e çevir."""
        return {
            'ratio': level.ratio,
            'price': round(level.price, 2),
            'type': level.level_type,
            'is_support': level.is_support,
            'is_resistance': level.is_resistance,
            'distance_pct': round(level.distance_pct, 2)
        }
    
    def format_for_telegram(self, result: Dict) -> str:
        """Telegram formatında sonuç."""
        if 'error' in result:
            return f"❌ Fibonacci hatası: {result['error']}"
        
        trend_emoji = "📈" if result['trend'] == 'UPTREND' else "📉"
        
        # Önemli seviyeler
        important = [l for l in result['levels'] if l['ratio'] in [0.382, 0.5, 0.618]]
        
        levels_text = ""
        for l in important:
            emoji = "🟢" if l['is_support'] else "🔴"
            levels_text += f"  {emoji} Fib {l['ratio']}: ${l['price']:,.0f} ({l['distance_pct']:+.1f}%)\n"
        
        support = result.get('nearest_support')
        resistance = result.get('nearest_resistance')
        
        msg = f"""
📐 FİBONACCİ ANALİZİ
━━━━━━━━━━━━━━━━━━━━━━
{result['symbol']} ({result['timeframe']})
{trend_emoji} Trend: {result['trend']}
━━━━━━━━━━━━━━━━━━━━━━
📊 Swing Noktaları:
  ↗️ Yüksek: ${result['swing_high']:,.0f}
  ↘️ Düşük: ${result['swing_low']:,.0f}
━━━━━━━━━━━━━━━━━━━━━━
🎯 Önemli Seviyeler:
{levels_text.strip()}
━━━━━━━━━━━━━━━━━━━━━━
💰 Şu An: ${result['current_price']:,.0f}
🟢 Destek: ${support['price']:,.0f} (Fib {support['ratio']}) 
🔴 Direnç: ${resistance['price']:,.0f} (Fib {resistance['ratio']})
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_fib = None

def get_fibonacci() -> FibonacciAnalyzer:
    """Get or create Fibonacci analyzer instance."""
    global _fib
    if _fib is None:
        _fib = FibonacciAnalyzer()
    return _fib
