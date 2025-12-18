# -*- coding: utf-8 -*-
"""
DEMIR AI - AI Gözlem Sistemi
Erken piyasa gözlemleri - henüz sinyal değil, takip için.

PHASE 108: Turkish Notification System
- Öncü Hareket yerine AI Gözlem
- Henüz sinyal değil, izleme için
- Anlaşılır Türkçe
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

logger = logging.getLogger("AI_OBSERVATION")


class AIObservationSystem:
    """
    AI Gözlem Sistemi
    
    Piyasada ilginç hareketler tespit eder ama henüz sinyal göndermez.
    Kullanıcıya "izle" uyarısı gönderir.
    """
    
    STATE_FILE = "ai_observations.json"
    
    # Thresholds
    PRICE_CHANGE_THRESHOLD = 1.5  # %1.5 hareket
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x normal hacim
    
    # Cooldown
    COOLDOWN_MINUTES = 45
    
    def __init__(self):
        self.last_observations: Dict[str, datetime] = {}
        self._load_state()
        logger.info("✅ AI Observation System initialized")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    for key, ts in data.get('last_observations', {}).items():
                        self.last_observations[key] = datetime.fromisoformat(ts)
        except Exception as e:
            logger.debug(f"State load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            with open(self.STATE_FILE, 'w') as f:
                json.dump({
                    'last_observations': {k: v.isoformat() for k, v in self.last_observations.items()},
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"State save failed: {e}")
    
    def _can_observe(self, symbol: str) -> bool:
        """Cooldown kontrolü."""
        if symbol not in self.last_observations:
            return True
        
        minutes_since = (datetime.now() - self.last_observations[symbol]).total_seconds() / 60
        return minutes_since >= self.COOLDOWN_MINUTES
    
    async def observe(self, symbol: str = 'BTCUSDT') -> Optional[Dict]:
        """
        Piyasa gözlemi yap.
        
        Returns:
            Observation dict if interesting movement detected, None otherwise
        """
        if not self._can_observe(symbol):
            return None
        
        try:
            # Son 15 dakika fiyat hareketi
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={'symbol': symbol, 'interval': '1m', 'limit': 20},
                timeout=10
            )
            
            if resp.status_code != 200:
                return None
            
            klines = resp.json()
            if len(klines) < 15:
                return None
            
            # Fiyat değişimi
            price_15m_ago = float(klines[-15][4])
            current_price = float(klines[-1][4])
            change_pct = ((current_price - price_15m_ago) / price_15m_ago) * 100
            
            # Hacim analizi
            volumes = [float(k[5]) for k in klines[:-3]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            recent_volume = sum([float(k[5]) for k in klines[-3:]]) / 3
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Gözlem yapılacak mı?
            if abs(change_pct) < self.PRICE_CHANGE_THRESHOLD and volume_ratio < self.VOLUME_SPIKE_THRESHOLD:
                return None
            
            # Gözlem oluştur
            self.last_observations[symbol] = datetime.now()
            self._save_state()
            
            # Yön ve detaylar
            if change_pct < -self.PRICE_CHANGE_THRESHOLD:
                direction = 'DÜŞÜŞ'
                direction_emoji = '📉'
            elif change_pct > self.PRICE_CHANGE_THRESHOLD:
                direction = 'YÜKSELİŞ'
                direction_emoji = '📈'
            else:
                direction = 'HACİM'
                direction_emoji = '📊'
            
            # Tespit detayları
            detections = []
            
            if abs(change_pct) > 1:
                detections.append(f"Fiyat %{abs(change_pct):.1f} {direction.lower()}")
            
            if volume_ratio > 2:
                detections.append(f"Hacim normalin {volume_ratio:.1f} katı")
            
            # Olasılık hesapla (basit)
            probability = min(75, abs(change_pct) * 15 + (volume_ratio - 1) * 10)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'direction_emoji': direction_emoji,
                'change_pct': change_pct,
                'current_price': current_price,
                'volume_ratio': volume_ratio,
                'detections': detections,
                'probability': probability
            }
            
        except Exception as e:
            logger.debug(f"Observation failed: {e}")
        
        return None
    
    def format_observation(self, obs: Dict) -> str:
        """Telegram formatında gözlem."""
        symbol = obs['symbol']
        direction = obs['direction']
        direction_emoji = obs['direction_emoji']
        change_pct = obs['change_pct']
        current_price = obs['current_price']
        volume_ratio = obs['volume_ratio']
        detections = obs['detections']
        probability = obs['probability']
        
        # Detaylar
        detections_text = ""
        for d in detections:
            detections_text += f"• {d}\n"
        
        msg = f"""
👁️ AI GÖZLEM
━━━━━━━━━━━━━━━━━━━━━━
{direction_emoji} {symbol}: {direction} Belirtisi

🧠 Tespit:
{detections_text.strip()}

💰 Fiyat: ${current_price:,.2f}
📊 Olasılık: %{probability:.0f} {direction.lower()}
⏱️ Beklenen: 5-15 dakika içinde
━━━━━━━━━━━━━━━━━━━━━━
⚠️ Bu bir sinyal DEĞİL, takip edin.
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_observer = None

def get_observer() -> AIObservationSystem:
    """Get or create observer instance."""
    global _observer
    if _observer is None:
        _observer = AIObservationSystem()
    return _observer
