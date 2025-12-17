# -*- coding: utf-8 -*-
"""
DEMIR AI - CME Gap Tracker
BTC CME Futures gap tespiti.

CME (Chicago Mercantile Exchange) hafta sonu kapalıdır.
Pazartesi açılışında fiyat farkı = GAP
%80+ ihtimalle gap kapanır.
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("CME_GAP_TRACKER")


@dataclass
class CMEGap:
    """CME Gap verisi"""
    gap_type: str  # BULLISH (yukarı gap) / BEARISH (aşağı gap)
    gap_start: float  # Gap başlangıç fiyatı (cuma kapanış)
    gap_end: float  # Gap bitiş fiyatı (pazartesi açılış)
    gap_size_usd: float  # Gap boyutu $
    gap_size_pct: float  # Gap boyutu %
    is_filled: bool  # Gap kapandı mı?
    fill_pct: float  # Ne kadar kapandı %
    created_at: datetime
    potential_target: float  # Gap kapanış hedefi


class CMEGapTracker:
    """
    CME Gap Takip Sistemi
    
    CME BTC Futures:
    - Pazar 23:00 UTC açılış
    - Cuma 22:00 UTC kapanış
    - Hafta sonu kapalı
    
    Gap Stratejisi:
    - Yukarı gap: SHORT sinyali (gap kapanması için)
    - Aşağı gap: LONG sinyali (gap kapanması için)
    - %80+ kapanma oranı
    """
    
    # CME trading saatleri (UTC)
    CME_OPEN_HOUR = 23  # Pazar
    CME_CLOSE_HOUR = 22  # Cuma
    
    # Minimum gap boyutu (%)
    MIN_GAP_SIZE_PCT = 0.5
    
    def __init__(self):
        self.current_gap: Optional[CMEGap] = None
        self.historical_gaps: List[CMEGap] = []
        self.friday_close: Optional[float] = None
        self.monday_open: Optional[float] = None
    
    def get_btc_price(self) -> float:
        """Güncel BTC fiyatı."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def get_friday_close_price(self) -> float:
        """Cuma kapanış fiyatını al (CME 22:00 UTC)."""
        try:
            # Son 1 haftalık saatlik veri al
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': 'BTCUSDT',
                    'interval': '1h',
                    'limit': 168  # 7 gün
                },
                timeout=10
            )
            
            if resp.status_code == 200:
                klines = resp.json()
                
                for kline in reversed(klines):
                    ts = datetime.fromtimestamp(kline[0] / 1000)
                    # Cuma 22:00 UTC
                    if ts.weekday() == 4 and ts.hour == 22:
                        return float(kline[4])  # Close price
            
        except Exception as e:
            logger.warning(f"Friday close fetch failed: {e}")
        
        return 0
    
    def get_monday_open_price(self) -> float:
        """Pazartesi (Pazar gece) açılış fiyatını al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    'symbol': 'BTCUSDT',
                    'interval': '1h',
                    'limit': 168
                },
                timeout=10
            )
            
            if resp.status_code == 200:
                klines = resp.json()
                
                for kline in klines:
                    ts = datetime.fromtimestamp(kline[0] / 1000)
                    # Pazar 23:00 UTC (CME açılış)
                    if ts.weekday() == 6 and ts.hour == 23:
                        return float(kline[1])  # Open price
            
        except Exception as e:
            logger.warning(f"Monday open fetch failed: {e}")
        
        return 0
    
    def detect_gap(self) -> Optional[CMEGap]:
        """Mevcut CME gap'i tespit et."""
        friday_close = self.get_friday_close_price()
        monday_open = self.get_monday_open_price()
        current_price = self.get_btc_price()
        
        if friday_close == 0 or monday_open == 0:
            logger.warning("Could not get CME reference prices")
            return None
        
        self.friday_close = friday_close
        self.monday_open = monday_open
        
        # Gap hesapla
        gap_size_usd = monday_open - friday_close
        gap_size_pct = (gap_size_usd / friday_close) * 100
        
        # Minimum gap kontrolü
        if abs(gap_size_pct) < self.MIN_GAP_SIZE_PCT:
            logger.info(f"No significant gap: {gap_size_pct:.2f}%")
            return None
        
        # Gap tipi
        if gap_size_usd > 0:
            gap_type = "BULLISH"  # Yukarı gap
            potential_target = friday_close  # Gap kapanma hedefi = Cuma kapanış
        else:
            gap_type = "BEARISH"  # Aşağı gap
            potential_target = friday_close
        
        # Gap dolum kontrolü
        if gap_type == "BULLISH":
            # Yukarı gap - fiyat Cuma kapanışına düştü mü?
            is_filled = current_price <= friday_close
            if is_filled:
                fill_pct = 100
            else:
                # Ne kadar kapandı?
                filled = monday_open - current_price
                fill_pct = (filled / gap_size_usd) * 100 if gap_size_usd != 0 else 0
                fill_pct = max(0, min(100, fill_pct))
        else:
            # Aşağı gap - fiyat Cuma kapanışına çıktı mı?
            is_filled = current_price >= friday_close
            if is_filled:
                fill_pct = 100
            else:
                filled = current_price - monday_open
                fill_pct = (filled / abs(gap_size_usd)) * 100 if gap_size_usd != 0 else 0
                fill_pct = max(0, min(100, fill_pct))
        
        gap = CMEGap(
            gap_type=gap_type,
            gap_start=friday_close,
            gap_end=monday_open,
            gap_size_usd=abs(gap_size_usd),
            gap_size_pct=abs(gap_size_pct),
            is_filled=is_filled,
            fill_pct=fill_pct,
            created_at=datetime.now(),
            potential_target=potential_target
        )
        
        self.current_gap = gap
        logger.info(f"CME Gap detected: {gap_type} ${abs(gap_size_usd):,.0f} ({abs(gap_size_pct):.2f}%)")
        
        return gap
    
    def get_signal_bias(self) -> Dict:
        """Gap'e dayalı sinyal önerisi."""
        gap = self.detect_gap()
        
        if gap is None:
            return {
                'has_gap': False,
                'signal': 'NEUTRAL',
                'confidence': 0,
                'reason': 'No significant CME gap'
            }
        
        if gap.is_filled:
            return {
                'has_gap': True,
                'signal': 'NEUTRAL',
                'confidence': 0,
                'reason': f'CME gap already filled',
                'gap_info': {
                    'type': gap.gap_type,
                    'size_pct': gap.gap_size_pct,
                    'filled': True
                }
            }
        
        # Gap kapanma stratejisi
        # Yukarı gap = SHORT (fiyat düşerek gap kapanır)
        # Aşağı gap = LONG (fiyat çıkarak gap kapanır)
        
        if gap.gap_type == "BULLISH":
            signal = "SHORT"
            reason = f"CME yukarı gap ${gap.gap_size_usd:,.0f} - kapanma bekleniyor"
        else:
            signal = "LONG"
            reason = f"CME aşağı gap ${gap.gap_size_usd:,.0f} - kapanma bekleniyor"
        
        # Güven = gap boyutuna ve dolum durumuna göre
        confidence = 50 + (gap.gap_size_pct * 5)  # Büyük gap = daha güvenli
        confidence = min(80, confidence)
        
        # Eğer gap kısmen dolmuşsa güven düşür
        confidence = confidence * (1 - gap.fill_pct / 200)
        
        return {
            'has_gap': True,
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'target_price': gap.potential_target,
            'gap_info': {
                'type': gap.gap_type,
                'size_usd': gap.gap_size_usd,
                'size_pct': gap.gap_size_pct,
                'fill_pct': gap.fill_pct,
                'friday_close': gap.gap_start,
                'monday_open': gap.gap_end
            }
        }
    
    def get_gap_status(self) -> Dict:
        """Dashboard için gap durumu."""
        gap = self.detect_gap()
        
        if gap is None:
            return {
                'status': 'NO_GAP',
                'message': 'Önemli CME gap yok',
                'signal': None
            }
        
        return {
            'status': 'GAP_ACTIVE' if not gap.is_filled else 'GAP_FILLED',
            'type': gap.gap_type,
            'size_usd': gap.gap_size_usd,
            'size_pct': gap.gap_size_pct,
            'fill_pct': gap.fill_pct,
            'target': gap.potential_target,
            'friday_close': gap.gap_start,
            'monday_open': gap.gap_end,
            'signal': 'SHORT' if gap.gap_type == 'BULLISH' else 'LONG',
            'message': f"{'🔴' if gap.gap_type == 'BULLISH' else '🟢'} {gap.gap_type} gap ${gap.gap_size_usd:,.0f} ({gap.gap_size_pct:.1f}%)"
        }


# Convenience functions
def get_cme_gap_signal() -> Dict:
    """Hızlı CME gap sinyali."""
    tracker = CMEGapTracker()
    return tracker.get_signal_bias()


def get_cme_status() -> Dict:
    """Dashboard için CME durumu."""
    tracker = CMEGapTracker()
    return tracker.get_gap_status()
