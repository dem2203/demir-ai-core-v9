# -*- coding: utf-8 -*-
"""
DEMIR AI - Composite Alert Score
Birleşik alarm skoru - Tüm modüllerin güçlü uyumu.

PHASE 90: Composite Alert Score
- Tüm modüllerden birleşik skor
- Threshold bazlı ani alarm
- Acil sinyal sistemi
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("COMPOSITE_ALERT")


class CompositeAlertScore:
    """
    Birleşik Alarm Skoru Sistemi
    
    Tüm modüller aynı yönde güçlü sinyal verdiğinde
    ACİL ALARM gönderir.
    """
    
    # Alert thresholds
    URGENT_THRESHOLD = 0.75  # %75+ modül aynı yönde = ACİL
    HIGH_THRESHOLD = 0.60    # %60+ modül aynı yönde = YÜKSEK
    MEDIUM_THRESHOLD = 0.50  # %50+ modül aynı yönde = ORTA
    
    def __init__(self):
        self.last_alert = None
        self.alert_history = []
        logger.info("✅ Composite Alert Score initialized")
    
    def calculate_composite_score(self, module_signals: List) -> Dict:
        """
        Tüm modüllerden birleşik skor hesapla.
        
        Args:
            module_signals: SignalOrchestrator'dan gelen ModuleSignal listesi
            
        Returns:
            {
                'composite_score': 0.78,
                'direction': 'LONG',
                'alert_level': 'URGENT',
                'long_modules': 15,
                'short_modules': 3,
                'neutral_modules': 2,
                'total_modules': 20,
                'high_confidence_count': 8,
                'trigger_alert': True,
                'confidence': 80
            }
        """
        if not module_signals:
            return self._empty_result()
        
        # Yön sayımları
        long_count = sum(1 for s in module_signals if s.direction == 'LONG')
        short_count = sum(1 for s in module_signals if s.direction == 'SHORT')
        neutral_count = sum(1 for s in module_signals if s.direction in ['NEUTRAL', 'HOLD', 'WAIT'])
        total = len(module_signals)
        
        # Ağırlıklı skor
        long_weight = sum(s.weight * (s.confidence / 100) for s in module_signals if s.direction == 'LONG')
        short_weight = sum(s.weight * (s.confidence / 100) for s in module_signals if s.direction == 'SHORT')
        total_weight = long_weight + short_weight
        
        # Yön ve composite score
        if long_count > short_count:
            direction = 'LONG'
            composite_score = long_weight / total_weight if total_weight > 0 else 0.5
            dominant_count = long_count
        elif short_count > long_count:
            direction = 'SHORT'
            composite_score = short_weight / total_weight if total_weight > 0 else 0.5
            dominant_count = short_count
        else:
            direction = 'NEUTRAL'
            composite_score = 0.5
            dominant_count = 0
        
        # Yüksek güvenli modül sayısı
        high_conf_count = sum(1 for s in module_signals if s.confidence >= 65 and s.direction == direction)
        
        # Alert seviyesi
        dominant_ratio = dominant_count / total if total > 0 else 0
        
        if dominant_ratio >= self.URGENT_THRESHOLD and high_conf_count >= 5:
            alert_level = 'URGENT'
            trigger_alert = True
            confidence = 85
        elif dominant_ratio >= self.HIGH_THRESHOLD and high_conf_count >= 3:
            alert_level = 'HIGH'
            trigger_alert = True
            confidence = 70
        elif dominant_ratio >= self.MEDIUM_THRESHOLD:
            alert_level = 'MEDIUM'
            trigger_alert = False
            confidence = 55
        else:
            alert_level = 'LOW'
            trigger_alert = False
            confidence = 40
        
        result = {
            'composite_score': composite_score,
            'direction': direction,
            'alert_level': alert_level,
            'long_modules': long_count,
            'short_modules': short_count,
            'neutral_modules': neutral_count,
            'total_modules': total,
            'high_confidence_count': high_conf_count,
            'dominant_ratio': dominant_ratio,
            'trigger_alert': trigger_alert,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'available': True
        }
        
        if trigger_alert:
            self._log_alert(result)
        
        return result
    
    def _log_alert(self, result: Dict):
        """Alert'i logla."""
        self.last_alert = result
        self.alert_history.append(result)
        
        level = result['alert_level']
        direction = result['direction']
        score = result['composite_score']
        
        if level == 'URGENT':
            logger.critical(f"🚨 URGENT ALERT: {direction} - Score: {score:.2f}")
        else:
            logger.warning(f"⚠️ HIGH ALERT: {direction} - Score: {score:.2f}")
    
    def get_alert_message(self, result: Dict, current_price: float) -> str:
        """Telegram için alert mesajı oluştur."""
        if not result.get('trigger_alert'):
            return ""
        
        direction = result['direction']
        level = result['alert_level']
        score = result['composite_score']
        
        if direction == 'LONG':
            emoji = "🟢"
            dir_text = "LONG 📈"
            tp1 = current_price * 1.02
            tp2 = current_price * 1.04
            sl = current_price * 0.985
        else:
            emoji = "🔴"
            dir_text = "SHORT 📉"
            tp1 = current_price * 0.98
            tp2 = current_price * 0.96
            sl = current_price * 1.015
        
        level_emoji = "🚨🚨🚨" if level == 'URGENT' else "⚠️⚠️"
        
        msg = f"""
{level_emoji} **{level} ALERT** {level_emoji}
━━━━━━━━━━━━━━━━━━━━━━
{emoji} Yön: **{dir_text}**
📊 Composite Score: **{score:.0%}**
━━━━━━━━━━━━━━━━━━━━━━
💰 Giriş: **${current_price:,.2f}**
🎯 TP1: ${tp1:,.2f}
🎯 TP2: ${tp2:,.2f}
🛡️ SL: ${sl:,.2f}
━━━━━━━━━━━━━━━━━━━━━━
📈 LONG Modül: {result['long_modules']}
📉 SHORT Modül: {result['short_modules']}
⭐ Yüksek Güven: {result['high_confidence_count']}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
🧠 35 modülden analiz edildi
""".strip()
        
        return msg
    
    def _empty_result(self) -> Dict:
        return {
            'composite_score': 0.5,
            'direction': 'NEUTRAL',
            'alert_level': 'LOW',
            'long_modules': 0,
            'short_modules': 0,
            'neutral_modules': 0,
            'total_modules': 0,
            'high_confidence_count': 0,
            'dominant_ratio': 0,
            'trigger_alert': False,
            'confidence': 0,
            'available': False
        }


# Convenience function
def calculate_composite_score(signals: List) -> Dict:
    """Quick composite score calculation."""
    scorer = CompositeAlertScore()
    return scorer.calculate_composite_score(signals)
