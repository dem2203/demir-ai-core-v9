# -*- coding: utf-8 -*-
"""
DEMIR AI - Pivot Points Analyzer
Günlük, Haftalık Pivot seviyeleri ve Camarilla.

PHASE 115: Pivot Points Analysis
- Standard Pivot Points (Daily/Weekly)
- Camarilla Pivot Points
- Support/Resistance seviyeleri (S1-S3, R1-R3)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger("PIVOT_POINTS")


@dataclass
class PivotLevel:
    """Pivot seviyesi."""
    name: str  # 'PP', 'S1', 'S2', 'S3', 'R1', 'R2', 'R3'
    price: float
    level_type: str  # 'PIVOT', 'SUPPORT', 'RESISTANCE'
    distance_pct: float


class PivotPointsAnalyzer:
    """
    Pivot Points Analizörü
    
    Günlük ve haftalık pivot noktalarını hesaplar.
    """
    
    def __init__(self):
        self.daily_pivots: List[PivotLevel] = []
        self.weekly_pivots: List[PivotLevel] = []
        self.camarilla_pivots: List[PivotLevel] = []
        logger.info("✅ Pivot Points Analyzer initialized")
    
    async def analyze(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Pivot Points analizi yap.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with daily, weekly, and camarilla pivots
        """
        try:
            # Günlük veri çek
            daily_data = await self._fetch_daily_data(symbol)
            if not daily_data:
                return {'error': 'No daily data'}
            
            # Haftalık veri çek
            weekly_data = await self._fetch_weekly_data(symbol)
            
            # Mevcut fiyat
            current_price = await self._get_current_price(symbol)
            
            # Pivotları hesapla
            self.daily_pivots = self._calculate_standard_pivots(
                daily_data, current_price, 'DAILY'
            )
            
            if weekly_data:
                self.weekly_pivots = self._calculate_standard_pivots(
                    weekly_data, current_price, 'WEEKLY'
                )
            
            self.camarilla_pivots = self._calculate_camarilla_pivots(
                daily_data, current_price
            )
            
            # En yakın seviyeleri bul
            nearest_support = self._find_nearest_level(current_price, 'support')
            nearest_resistance = self._find_nearest_level(current_price, 'resistance')
            
            # Sinyal
            signal = self._generate_signal(current_price, nearest_support, nearest_resistance)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'daily_pivots': [self._level_to_dict(l) for l in self.daily_pivots],
                'weekly_pivots': [self._level_to_dict(l) for l in self.weekly_pivots],
                'camarilla_pivots': [self._level_to_dict(l) for l in self.camarilla_pivots],
                'nearest_support': self._level_to_dict(nearest_support) if nearest_support else None,
                'nearest_resistance': self._level_to_dict(nearest_resistance) if nearest_resistance else None,
                'signal': signal,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pivot analysis failed: {e}")
            return {'error': str(e)}
    
    async def _fetch_daily_data(self, symbol: str) -> Optional[Dict]:
        """Dünün OHLC verisi."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1d', 'limit': 2},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            if len(klines) < 2:
                return None
            
            # Dünün verisi
            yesterday = klines[-2]
            return {
                'high': float(yesterday[2]),
                'low': float(yesterday[3]),
                'close': float(yesterday[4])
            }
            
        except Exception as e:
            logger.error(f"Daily data fetch failed: {e}")
            return None
    
    async def _fetch_weekly_data(self, symbol: str) -> Optional[Dict]:
        """Geçen haftanın OHLC verisi."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1w', 'limit': 2},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            if len(klines) < 2:
                return None
            
            # Geçen haftanın verisi
            last_week = klines[-2]
            return {
                'high': float(last_week[2]),
                'low': float(last_week[3]),
                'close': float(last_week[4])
            }
            
        except Exception as e:
            logger.error(f"Weekly data fetch failed: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Mevcut fiyat."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def _calculate_standard_pivots(self, data: Dict, current_price: float, period: str) -> List[PivotLevel]:
        """
        Standard Pivot Points hesapla.
        
        PP = (High + Low + Close) / 3
        R1 = 2*PP - Low
        S1 = 2*PP - High
        R2 = PP + (High - Low)
        S2 = PP - (High - Low)
        R3 = High + 2*(PP - Low)
        S3 = Low - 2*(High - PP)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        pp = (high + low + close) / 3
        
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        def pct(price):
            return ((price - current_price) / current_price) * 100
        
        return [
            PivotLevel(f'{period}_S3', s3, 'SUPPORT', pct(s3)),
            PivotLevel(f'{period}_S2', s2, 'SUPPORT', pct(s2)),
            PivotLevel(f'{period}_S1', s1, 'SUPPORT', pct(s1)),
            PivotLevel(f'{period}_PP', pp, 'PIVOT', pct(pp)),
            PivotLevel(f'{period}_R1', r1, 'RESISTANCE', pct(r1)),
            PivotLevel(f'{period}_R2', r2, 'RESISTANCE', pct(r2)),
            PivotLevel(f'{period}_R3', r3, 'RESISTANCE', pct(r3)),
        ]
    
    def _calculate_camarilla_pivots(self, data: Dict, current_price: float) -> List[PivotLevel]:
        """
        Camarilla Pivot Points hesapla.
        
        R4 = Close + (High - Low) * 1.1/2
        R3 = Close + (High - Low) * 1.1/4
        R2 = Close + (High - Low) * 1.1/6
        R1 = Close + (High - Low) * 1.1/12
        S1 = Close - (High - Low) * 1.1/12
        S2 = Close - (High - Low) * 1.1/6
        S3 = Close - (High - Low) * 1.1/4
        S4 = Close - (High - Low) * 1.1/2
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        range_val = high - low
        
        r4 = close + range_val * 1.1 / 2
        r3 = close + range_val * 1.1 / 4
        r2 = close + range_val * 1.1 / 6
        r1 = close + range_val * 1.1 / 12
        s1 = close - range_val * 1.1 / 12
        s2 = close - range_val * 1.1 / 6
        s3 = close - range_val * 1.1 / 4
        s4 = close - range_val * 1.1 / 2
        
        def pct(price):
            return ((price - current_price) / current_price) * 100
        
        return [
            PivotLevel('CAM_S4', s4, 'SUPPORT', pct(s4)),
            PivotLevel('CAM_S3', s3, 'SUPPORT', pct(s3)),
            PivotLevel('CAM_S2', s2, 'SUPPORT', pct(s2)),
            PivotLevel('CAM_S1', s1, 'SUPPORT', pct(s1)),
            PivotLevel('CAM_R1', r1, 'RESISTANCE', pct(r1)),
            PivotLevel('CAM_R2', r2, 'RESISTANCE', pct(r2)),
            PivotLevel('CAM_R3', r3, 'RESISTANCE', pct(r3)),
            PivotLevel('CAM_R4', r4, 'RESISTANCE', pct(r4)),
        ]
    
    def _find_nearest_level(self, current_price: float, level_type: str) -> Optional[PivotLevel]:
        """En yakın destek veya direnci bul."""
        all_levels = self.daily_pivots + self.camarilla_pivots
        
        if level_type == 'support':
            candidates = [l for l in all_levels if l.price < current_price]
            if not candidates:
                return None
            return max(candidates, key=lambda x: x.price)
        else:
            candidates = [l for l in all_levels if l.price > current_price]
            if not candidates:
                return None
            return min(candidates, key=lambda x: x.price)
    
    def _generate_signal(self, current_price: float,
                        support: Optional[PivotLevel],
                        resistance: Optional[PivotLevel]) -> Dict:
        """Pivot sinyali oluştur."""
        signal = {
            'direction': 'NEUTRAL',
            'confidence': 40,
            'reason': ''
        }
        
        # Pivot seviyesine yakınlık
        for level in self.daily_pivots:
            if level.name.endswith('_PP'):
                distance = abs(level.distance_pct)
                if distance < 0.5:  # %0.5'ten yakın
                    signal['reason'] = f"Pivot noktasında (${level.price:,.0f})"
                    signal['confidence'] = 60
        
        # Camarilla S3/R3 breakout sinyali
        for level in self.camarilla_pivots:
            if 'S3' in level.name and level.distance_pct > -0.3 and level.distance_pct < 0:
                signal['direction'] = 'LONG'
                signal['confidence'] = 65
                signal['reason'] = f"Camarilla S3 desteğinde (${level.price:,.0f})"
            elif 'R3' in level.name and level.distance_pct < 0.3 and level.distance_pct > 0:
                signal['direction'] = 'SHORT'
                signal['confidence'] = 65
                signal['reason'] = f"Camarilla R3 direncinde (${level.price:,.0f})"
        
        return signal
    
    def _level_to_dict(self, level: PivotLevel) -> Dict:
        """PivotLevel'ı dict'e çevir."""
        return {
            'name': level.name,
            'price': round(level.price, 2),
            'type': level.level_type,
            'distance_pct': round(level.distance_pct, 2)
        }
    
    def format_for_telegram(self, result: Dict) -> str:
        """Telegram formatında sonuç."""
        if 'error' in result:
            return f"❌ Pivot hatası: {result['error']}"
        
        # Günlük pivotlar
        daily = result.get('daily_pivots', [])
        pp = next((l for l in daily if '_PP' in l['name']), None)
        s1 = next((l for l in daily if '_S1' in l['name']), None)
        r1 = next((l for l in daily if '_R1' in l['name']), None)
        
        support = result.get('nearest_support')
        resistance = result.get('nearest_resistance')
        
        msg = f"""
📍 PİVOT NOKTALARI
━━━━━━━━━━━━━━━━━━━━━━
{result['symbol']} | Günlük
━━━━━━━━━━━━━━━━━━━━━━
🔴 R1: ${r1['price']:,.0f} ({r1['distance_pct']:+.1f}%)
⚪ PP: ${pp['price']:,.0f} ({pp['distance_pct']:+.1f}%)
🟢 S1: ${s1['price']:,.0f} ({s1['distance_pct']:+.1f}%)
━━━━━━━━━━━━━━━━━━━━━━
💰 Şu An: ${result['current_price']:,.0f}
🟢 Destek: {support['name']} @ ${support['price']:,.0f}
🔴 Direnç: {resistance['name']} @ ${resistance['price']:,.0f}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_pivot = None

def get_pivot_points() -> PivotPointsAnalyzer:
    """Get or create Pivot Points analyzer instance."""
    global _pivot
    if _pivot is None:
        _pivot = PivotPointsAnalyzer()
    return _pivot
