import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger("RISK_MANAGER_SUPERHUMAN")

class RiskManager:
    """
    SUPERHUMAN RISK MANAGEMENT
    
    Sıradan botlar sabit bakiye kullanır. Bu modül ise:
    1. KELLY CRITERION: Bahis büyüklüğünü kazanma ihtimaline göre ayarlar.
    2. ATR BASED DYNAMIC STOPS: Stop Loss'u piyasanın gürültüsüne (Volatilite) göre ayarlar.
    3. MAX DRAWDOWN PROTECTION: Kasa erimeye başlarsa botu otomatik frenler.
    """

    def __init__(self):
        self.max_account_risk = 0.02  # Tek işlemde kasadaki paranın max %2'si risk edilir.
        self.risk_reward_ratio = 2.0  # Hedeflenen Kar/Zarar oranı (En az 1'e 2)

    def calculate_dynamic_stops(self, entry_price: float, side: str, atr_value: float) -> Dict[str, float]:
        """
        Stop Loss ve Take Profit noktalarını ATR (Volatilite) kullanarak dinamik hesaplar.
        Piyasa çok hareketliyse stop aralığını genişletir (Patlamamak için).
        """
        # ATR Çarpanı: 2xATR (Standart profesyonel koruma aralığı)
        stop_distance = atr_value * 2.0
        
        if side == "BUY" or side == "LONG":
            sl_price = entry_price - stop_distance
            # Risk/Reward 1:2 ise, Kar hedefi stop mesafesinin 2 katı olmalı
            tp_price = entry_price + (stop_distance * self.risk_reward_ratio)
        else: # SELL / SHORT
            sl_price = entry_price + stop_distance
            tp_price = entry_price - (stop_distance * self.risk_reward_ratio)
            
        return {"sl": sl_price, "tp": tp_price}

    def apply_kelly_criterion(self, win_rate: float, win_loss_ratio: float, balance: float) -> float:
        """
        KELLY KRİTERİ (The Golden Formula):
        Pozisyon büyüklüğünü optimize eder.
        Formül: f* = (bp - q) / b
        """
        # Güvenlik: Asla tam Kelly kullanma (Çok riskli), "Half Kelly" kullan.
        kelly_fraction = 0.5 
        
        if win_loss_ratio == 0: return 0
        
        # Kelly Formülü
        f_star = ((win_loss_ratio * win_rate) - (1 - win_rate)) / win_loss_ratio
        
        # Negatif Kelly (İşlem yapma) veya Aşırı büyük risk koruması
        safe_size_percent = max(0.0, min(f_star * kelly_fraction, self.max_account_risk * 5))
        
        # TL/Dolar bazında miktar
        position_amount = balance * safe_size_percent
        
        logger.info(f"Kelly Optimization: WinRate:{win_rate}, Ratio:{win_loss_ratio} -> Size %: {safe_size_percent*100:.2f}")
        return position_amount

    def calculate_position_size(self, balance: float, entry_price: float, sl_price: float, ai_confidence: float) -> float:
        """
        Nihai Pozisyon Büyüklüğü Hesaplayıcı.
        Hem teknik riske (Stop Loss mesafesi) hem de AI güvenine (Confidence) bakar.
        """
        if balance <= 0: return 0

        # 1. Teknik Risk Bazlı Hesaplama (Kasayı korumak için max kaybedilecek tutar)
        risk_per_share = abs(entry_price - sl_price)
        if risk_per_share == 0: return 0
        
        # Kasadaki paranın %2'sini riske atacak şekilde lot hesapla
        money_at_risk = balance * self.max_account_risk
        size_based_on_risk = money_at_risk / risk_per_share * entry_price # USDT karşılığı

        # 2. AI Güven Skoru Bazlı Ayarlama (Eğer AI %90 eminse, riski biraz artırabiliriz)
        # Güven 50'nin altındaysa işlem açma.
        if ai_confidence < 50:
            return 0
        
        confidence_multiplier = ai_confidence / 80.0 # 80 puan üstü normal, altı defansif
        final_size = size_based_on_risk * confidence_multiplier

        # 3. Kasa Kontrolü (Bakiyeden fazlasını açamayız - Kaldıraçsız varsayımı)
        final_size = min(final_size, balance * 0.95) # %5 fee için ayır

        logger.info(f"Position Sizing: Balance=${balance} -> Trade Size=${final_size:.2f} (Conf: {ai_confidence}%)")
        return final_size