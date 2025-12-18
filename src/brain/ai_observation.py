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
        PHASE 121: Geliştirilmiş piyasa gözlemi.
        
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
            
            # Son mumlardaki hareket
            last_open = float(klines[-1][1])
            last_close = float(klines[-1][4])
            last_candle_direction = 'UP' if last_close > last_open else 'DOWN'
            
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
            
            # YÖN TAHMİNİ
            if change_pct > 0.5:
                direction = 'YUKARI'
                direction_emoji = '🟢'
                likely_action = 'LONG'
            elif change_pct < -0.5:
                direction = 'AŞAĞI'
                direction_emoji = '🔴'
                likely_action = 'SHORT'
            elif volume_ratio > 2:
                # Hacim spike - mum yönüne bak
                if last_candle_direction == 'UP':
                    direction = 'YUKARI'
                    direction_emoji = '🟢'
                    likely_action = 'LONG'
                else:
                    direction = 'AŞAĞI'
                    direction_emoji = '🔴'
                    likely_action = 'SHORT'
            else:
                direction = 'BELİRSİZ'
                direction_emoji = '⚪'
                likely_action = 'BEKLE'
            
            # TAHMİNİ HAREKET MİKTARI
            # Volume spike varsa daha büyük hareket bekle
            base_move = abs(change_pct) * 1.5  # Momentum devam edebilir
            volume_multiplier = min(2, volume_ratio / 2)  # Hacim faktörü
            estimated_move = base_move * volume_multiplier
            
            # Min/Max sınırla
            estimated_move = max(0.5, min(5, estimated_move))
            
            # Olasılık hesapla
            probability = min(80, abs(change_pct) * 20 + (volume_ratio - 1) * 15)
            
            # Hedef fiyat hesapla
            if direction == 'YUKARI':
                target_price = current_price * (1 + estimated_move/100)
            elif direction == 'AŞAĞI':
                target_price = current_price * (1 - estimated_move/100)
            else:
                target_price = current_price
            
            return {
                'symbol': symbol,
                'direction': direction,
                'direction_emoji': direction_emoji,
                'likely_action': likely_action,
                'change_pct': change_pct,
                'current_price': current_price,
                'target_price': target_price,
                'estimated_move': estimated_move,
                'volume_ratio': volume_ratio,
                'probability': probability
            }
            
        except Exception as e:
            logger.debug(f"Observation failed: {e}")
        
        return None
    
    def format_observation(self, obs: Dict) -> str:
        """PHASE 121: Geliştirilmiş Telegram formatı."""
        symbol = obs['symbol']
        direction = obs['direction']
        direction_emoji = obs['direction_emoji']
        likely_action = obs['likely_action']
        change_pct = obs['change_pct']
        current_price = obs['current_price']
        target_price = obs['target_price']
        estimated_move = obs['estimated_move']
        volume_ratio = obs['volume_ratio']
        probability = obs['probability']
        
        # Hareket emojisi
        move_emoji = '📈' if direction == 'YUKARI' else '📉' if direction == 'AŞAĞI' else '📊'
        
        msg = f"""
👁️ AI GÖZLEM - {symbol}
━━━━━━━━━━━━━━━━━━━━━━
{direction_emoji} Yön: {direction}
{move_emoji} Son 15dk: {change_pct:+.2f}%
🔥 Hacim: Normalin {volume_ratio:.1f}x

📊 TAHMİN:
• Olası hareket: %{estimated_move:.1f} {direction.lower()}
• Hedef: ${target_price:,.2f}
• Olasılık: %{probability:.0f}

💰 Şu an: ${current_price:,.2f}
🎯 Olası işlem: {likely_action}
━━━━━━━━━━━━━━━━━━━━━━
⚠️ SİNYAL DEĞİL - Sadece gözlem!
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
