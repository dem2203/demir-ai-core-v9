# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SMART RISK MANAGER
=================================
Sinyal kalitesi ve piyasa oynaklığına göre akıllı kasa yönetimi yapar.
Profesyonel risk yönetimi kurallarını uygular.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger("RISK_MANAGER")

@dataclass
class RiskProfile:
    position_size_pct: float  # Kasanın yüzde kaçı ile girilmeli (0-100)
    leverage: int             # Kaldıraç önerisi (1x - 20x)
    risk_per_trade_usd: float # (Opsiyonel) Dolar bazlı risk
    can_trade: bool           # Ticaret yapılabilir mi?
    reason: str               # Risk kararının nedeni

class SmartRiskManager:
    def __init__(self, max_risk_per_trade=0.02, max_leverage=10):
        """
        :param max_risk_per_trade: Tek işlemde kasanın maksimum % kaçı riske atılabilir (örn: 0.02 = %2)
        :param max_leverage: Maksimum izin verilen kaldıraç
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_leverage = max_leverage
        logger.info("🛡️ Smart Risk Manager initialized")

    def calculate_risk(self, confidence: float, volatility_ratio: float, account_balance: float = 1000.0) -> RiskProfile:
        """
        Sinyal güveni ve volatiliteye göre pozisyon büyüklüğü hesaplar.
        
        :param confidence: AI Sinyal Güveni (0-100)
        :param volatility_ratio: Volatilite Oranı (1.0 = Normal, >1.5 = Yüksek, <0.7 = Düşük)
        :return: RiskProfile
        """
        
        # 1. Güven Bazlı Taban (Kelly Benzeri Mantık)
        # Güven < 40 ise işlem yok
        if confidence < 40:
            return RiskProfile(0, 0, 0, False, "Düşük Güven Skoru")

        # Güven arttıkça risk iştahı artar (Lineer)
        # 40-60 puan -> %1 risk
        # 60-80 puan -> %2 risk
        # 80-100 puan -> %3 risk (Agresif)
        
        base_risk_pct = 1.0
        if confidence > 80:
            base_risk_pct = 3.0
        elif confidence > 60:
            base_risk_pct = 2.0
            
        # 2. Volatilite Ayarı
        # Volatilite yüksekse riski düşür (Stop olma ihtimali yüksek)
        # Volatilite düşükse (Squeeze) riski biraz artırabilirsin ama dikkatli ol
        if volatility_ratio > 1.5:
            base_risk_pct *= 0.5  # Yarı yarıya düşür
            reason_suffix = "(Yüksek Volatilite - Risk Azaltıldı)"
        elif volatility_ratio < 0.7:
            # Squeeze durumunda patlama sert olabilir, stopu yakın koyacağız
            # Risk miktarını değiştirmeye gerek yok ama leverage düşürülebilir
            reason_suffix = "(Squeeze - Patlama Bekleniyor)"
        else:
            reason_suffix = "(Normal Piyasa)"
            
        # 3. Kaldıraç Hesabı
        # Güven yüksekse kaldıraç artabilir, ama max sınırı aşamaz
        if confidence > 85 and volatility_ratio < 1.2:
            leverage = min(self.max_leverage, 10) # 10x max
        elif confidence > 60:
            leverage = 5  # 5x standart
        else:
            leverage = 3  # 3x güvenli
            
        # Çok yüksek volatilitede kaldıracı kıs
        if volatility_ratio > 2.0:
            leverage = max(1, leverage // 2)
            
        # 4. Pozisyon Büyüklüğü (Margin)
        # Stoploss %'sine göre kasa yönetimi (Risk Amount / SL distance)
        # Basitleştirilmiş: Direkt kasanın %X'i ile işleme gir (Margin)
        # Örn: %2 risk alacaksan ve 5x kaldıraç kullanacaksan -> Kasadan %10 margin ayırabilirsin (tehlikeli olabilir)
        # Biz burada "Position Size" değil "Margin Size" öneriyoruz (Kasadan ayrılacak para)
        
        # Daha güvenli yöntem: Sabit Margin
        # Güven yüksekse kasanın %5-10'u
        # Güven düşükse kasanın %1-3'ü
        
        margin_pct = base_risk_pct * 2.5 # Örn: Score risk %2 -> Margin %5
        margin_pct = min(margin_pct, 15.0) # Max kasanın %15'i bir işleme bağlanabilir
        
        risk_amount_usd = account_balance * (base_risk_pct / 100.0)
        
        return RiskProfile(
            position_size_pct=round(margin_pct, 1),
            leverage=leverage,
            risk_per_trade_usd=round(risk_amount_usd, 2),
            can_trade=True,
            reason=f"Güven: %{confidence:.0f} | Vol: {volatility_ratio:.2f} {reason_suffix}"
        )

# Singleton
_risk_manager = None

def get_risk_manager():
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = SmartRiskManager()
    return _risk_manager
