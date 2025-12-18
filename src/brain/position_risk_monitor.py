# -*- coding: utf-8 -*-
"""
DEMIR AI - Position Risk Monitor
Aktif pozisyonlar için risk takibi ve uyarı sistemi.

PHASE 94: Position Risk Monitor
- Aktif pozisyonları izle
- Risk tetikleyicilerini kontrol et
- Uyarı bildirimi gönder
- SL güncelleme önerisi
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger("RISK_MONITOR")


class PositionRiskMonitor:
    """
    Aktif Pozisyon Risk Monitörü
    
    Açık pozisyonlar için risk durumlarını tespit eder ve uyarı gönderir.
    """
    
    # Risk thresholds
    PRICE_DEVIATION_THRESHOLD = 2.0  # %2 ters hareket
    LS_RATIO_SPIKE = 0.3  # L/S ratio ani değişim
    FUNDING_EXTREME = 0.05  # %0.05 funding
    VOLUME_SPIKE_MULTIPLIER = 3.0  # 3x normal hacim
    
    # Cooldowns
    WARNING_COOLDOWN_MINUTES = 30  # Aynı pozisyon için 30dk'da 1 uyarı
    
    def __init__(self):
        self.last_warnings: Dict[str, datetime] = {}
        self.previous_ls_ratio: Dict[str, float] = {}
        logger.info("✅ Position Risk Monitor initialized")
    
    def check_position_risks(self, symbol: str, position: Dict) -> List[Dict]:
        """
        Aktif pozisyon için risk kontrolleri yap.
        
        Args:
            symbol: BTCUSDT
            position: {direction, entry, tp1, tp2, sl, created_at}
            
        Returns:
            List of risk alerts: [{type, severity, message, recommendation}]
        """
        risks = []
        
        direction = position.get('direction', 'LONG')
        entry = position.get('entry', 0)
        sl = position.get('sl', 0)
        
        if entry == 0:
            return risks
        
        # Mevcut fiyatı al
        current_price = self._get_price(symbol)
        if current_price == 0:
            return risks
        
        # 1. Fiyat sapması kontrolü
        price_risk = self._check_price_deviation(symbol, direction, entry, current_price, sl)
        if price_risk:
            risks.append(price_risk)
        
        # 2. L/S ratio spike kontrolü
        ls_risk = self._check_ls_ratio_spike(symbol, direction)
        if ls_risk:
            risks.append(ls_risk)
        
        # 3. Funding rate kontrolü
        funding_risk = self._check_funding_extreme(symbol, direction)
        if funding_risk:
            risks.append(funding_risk)
        
        # 4. Hacim spike kontrolü (piyasa paniği)
        volume_risk = self._check_volume_spike(symbol, direction)
        if volume_risk:
            risks.append(volume_risk)
        
        return risks
    
    def _check_price_deviation(self, symbol: str, direction: str, entry: float, current: float, sl: float) -> Optional[Dict]:
        """Fiyat sapması kontrolü."""
        if direction == 'LONG':
            deviation = ((current - entry) / entry) * 100
            if deviation < -self.PRICE_DEVIATION_THRESHOLD:
                # SL'e yakınlık
                sl_distance = ((current - sl) / current) * 100 if sl > 0 else 100
                
                severity = 'HIGH' if sl_distance < 1 else 'MEDIUM'
                
                return {
                    'type': 'PRICE_DEVIATION',
                    'severity': severity,
                    'message': f"⚠️ LONG pozisyon {abs(deviation):.1f}% zarar",
                    'details': {
                        'entry': entry,
                        'current': current,
                        'deviation': deviation,
                        'sl_distance': sl_distance
                    },
                    'recommendation': f"SL'e {sl_distance:.1f}% kaldı. Kısmi kapatma düşünün." if sl_distance < 2 else "İzlemeye devam."
                }
        
        elif direction == 'SHORT':
            deviation = ((entry - current) / entry) * 100
            if deviation < -self.PRICE_DEVIATION_THRESHOLD:
                sl_distance = ((sl - current) / current) * 100 if sl > 0 else 100
                
                severity = 'HIGH' if sl_distance < 1 else 'MEDIUM'
                
                return {
                    'type': 'PRICE_DEVIATION',
                    'severity': severity,
                    'message': f"⚠️ SHORT pozisyon {abs(deviation):.1f}% zarar",
                    'details': {
                        'entry': entry,
                        'current': current,
                        'deviation': deviation,
                        'sl_distance': sl_distance
                    },
                    'recommendation': f"SL'e {sl_distance:.1f}% kaldı. Kısmi kapatma düşünün." if sl_distance < 2 else "İzlemeye devam."
                }
        
        return None
    
    def _check_ls_ratio_spike(self, symbol: str, direction: str) -> Optional[Dict]:
        """L/S ratio ani değişim kontrolü."""
        try:
            resp = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
                params={'symbol': symbol, 'period': '5m', 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    current_ls = float(data[0].get('longShortRatio', 1))
                    
                    # Önceki değerle karşılaştır
                    prev_ls = self.previous_ls_ratio.get(symbol, current_ls)
                    change = current_ls - prev_ls
                    
                    # Güncelle
                    self.previous_ls_ratio[symbol] = current_ls
                    
                    # LONG pozisyon + L/S arttı = risk (herkes long'da)
                    if direction == 'LONG' and change > self.LS_RATIO_SPIKE:
                        return {
                            'type': 'LS_RATIO_SPIKE',
                            'severity': 'MEDIUM',
                            'message': f"⚠️ L/S ratio {current_ls:.2f}'e yükseldi (+{change:.2f})",
                            'details': {'current_ls': current_ls, 'change': change},
                            'recommendation': "Kalabalık LONG trade. Dikkatli olun."
                        }
                    
                    # SHORT pozisyon + L/S düştü = risk
                    elif direction == 'SHORT' and change < -self.LS_RATIO_SPIKE:
                        return {
                            'type': 'LS_RATIO_SPIKE',
                            'severity': 'MEDIUM',
                            'message': f"⚠️ L/S ratio {current_ls:.2f}'e düştü ({change:.2f})",
                            'details': {'current_ls': current_ls, 'change': change},
                            'recommendation': "Short pozisyonlar artıyor. Dikkatli olun."
                        }
        except:
            pass
        
        return None
    
    def _check_funding_extreme(self, symbol: str, direction: str) -> Optional[Dict]:
        """Funding rate aşırı değer kontrolü."""
        try:
            resp = requests.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate",
                params={'symbol': symbol, 'limit': 1},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    funding = float(data[0].get('fundingRate', 0)) * 100  # Yüzdeye çevir
                    
                    # LONG + yüksek pozitif funding = risk
                    if direction == 'LONG' and funding > self.FUNDING_EXTREME:
                        return {
                            'type': 'FUNDING_EXTREME',
                            'severity': 'LOW',
                            'message': f"⚠️ Funding rate çok yüksek: {funding:.3f}%",
                            'details': {'funding': funding},
                            'recommendation': "Long maliyeti yüksek. Uzun tutmayın."
                        }
                    
                    # SHORT + düşük negatif funding = risk
                    elif direction == 'SHORT' and funding < -self.FUNDING_EXTREME:
                        return {
                            'type': 'FUNDING_EXTREME',
                            'severity': 'LOW',
                            'message': f"⚠️ Funding rate çok düşük: {funding:.3f}%",
                            'details': {'funding': funding},
                            'recommendation': "Short maliyeti yüksek. Uzun tutmayın."
                        }
        except:
            pass
        
        return None
    
    def _check_volume_spike(self, symbol: str, direction: str) -> Optional[Dict]:
        """Hacim spike kontrolü."""
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/24hr",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                volume = float(data.get('volume', 0))
                price_change = float(data.get('priceChangePercent', 0))
                
                # Ani düşüş + yüksek hacim = panik satışı
                if direction == 'LONG' and price_change < -3 and volume > 20000:  # 20k BTC hacim
                    return {
                        'type': 'VOLUME_SPIKE',
                        'severity': 'HIGH',
                        'message': f"🚨 Panik satışı tespit! {price_change:.1f}% düşüş, {volume/1000:.0f}K BTC hacim",
                        'details': {'volume': volume, 'price_change': price_change},
                        'recommendation': "Acil aksiyon düşünün!"
                    }
                
                # Ani yükseliş + yüksek hacim = short squeeze
                elif direction == 'SHORT' and price_change > 3 and volume > 20000:
                    return {
                        'type': 'VOLUME_SPIKE',
                        'severity': 'HIGH',
                        'message': f"🚨 Short squeeze riski! +{price_change:.1f}% yükseliş, {volume/1000:.0f}K BTC hacim",
                        'details': {'volume': volume, 'price_change': price_change},
                        'recommendation': "Acil aksiyon düşünün!"
                    }
        except:
            pass
        
        return None
    
    def _get_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=2
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def can_send_warning(self, symbol: str) -> bool:
        """Uyarı gönderilebilir mi? (Cooldown kontrolü)"""
        if symbol not in self.last_warnings:
            return True
        
        last = self.last_warnings[symbol]
        minutes_since = (datetime.now() - last).total_seconds() / 60
        
        return minutes_since >= self.WARNING_COOLDOWN_MINUTES
    
    def mark_warning_sent(self, symbol: str):
        """Uyarı gönderildi olarak işaretle."""
        self.last_warnings[symbol] = datetime.now()
    
    def format_risk_warning(self, symbol: str, position: Dict, risks: List[Dict]) -> str:
        """Telegram için risk uyarısı formatla - ANLAŞILIR TÜRKÇE."""
        if not risks:
            return ""
        
        direction = position.get('direction', 'LONG')
        entry = position.get('entry', 0)
        current_price = self._get_price(symbol)
        
        # Türkçe çeviriler
        dir_tr = 'AL' if direction == 'LONG' else 'SAT'
        dir_emoji = "🟢" if direction == 'LONG' else "🔴"
        
        # Kar/zarar hesapla
        if direction == 'LONG':
            pnl_pct = ((current_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - current_price) / entry) * 100
        
        pnl_emoji = "📈" if pnl_pct > 0 else "📉"
        pnl_text = f"+{pnl_pct:.2f}%" if pnl_pct > 0 else f"{pnl_pct:.2f}%"
        
        # En yüksek severity'yi bul
        severities = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        max_severity = max(risks, key=lambda x: severities.get(x['severity'], 0))['severity']
        
        if max_severity == 'HIGH':
            header = "🚨 RİSK UYARISI"
        elif max_severity == 'MEDIUM':
            header = "⚠️ RİSK UYARISI"
        else:
            header = "📊 RİSK UYARISI"
        
        # Risk mesajlarını düzenle
        risks_text = ""
        recommendations = []
        for risk in risks:
            risks_text += f"• {risk['message']}\n"
            recommendations.append(risk['recommendation'])
        
        msg = f"""
{header}
━━━━━━━━━━━━━━━━━━━━━━
{dir_emoji} {symbol}: {dir_tr} pozisyonunuz tehlikede!

📊 Durum:
• Giriş Fiyatı: ${entry:,.2f}
• Şu Anki Fiyat: ${current_price:,.2f}
• Kar/Zarar: {pnl_emoji} {pnl_text}
━━━━━━━━━━━━━━━━━━━━━━
Tespit Edilen Riskler:
{risks_text.strip()}
━━━━━━━━━━━━━━━━━━━━━━
💡 Öneriler:
• Zarar kesmeyi daraltın
• Pozisyonu küçültün
• Piyasayı yakından takip edin
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_monitor = None

def get_monitor() -> PositionRiskMonitor:
    """Get or create monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PositionRiskMonitor()
    return _monitor
