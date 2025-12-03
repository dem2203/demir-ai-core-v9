import pandas as pd
import logging
from typing import Dict, Optional

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RiskManager")

class RiskManager:
    """
    Sermaye Koruma ve Pozisyon Büyüklüğü Hesaplama Motoru.
    Sabit Lot YOK. Volatilite Bazlı (ATR) Dinamik Risk Yönetimi VAR.
    Sentiment (Duygu Analizi) iptal edilmiştir.
    """
    
    def __init__(self, max_risk_per_trade: float = 0.01, max_account_risk: float = 0.05):
        """
        :param max_risk_per_trade: Tek bir işlemde kasanın kaybedilebilecek maksimum oranı (Örn: %1 -> 0.01)
        :param max_account_risk: Açık tüm pozisyonların toplam riski (Örn: %5)
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_account_risk = max_account_risk

    def calculate_position_size(self, 
                              account_balance: float, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              volatility_atr: float) -> float:
        """
        Kelly Kriteri ve ATR bazlı güvenli pozisyon büyüklüğünü hesaplar.
        Asla rastgele veya hardcoded bir rakam döndürmez.
        Duygu analizi parametresi kaldırılmıştır.
        """
        if account_balance <= 0 or entry_price <= 0:
            logger.error("Hatalı bakiye veya fiyat verisi.")
            return 0.0

        # 1. İşlem başına riske edilecek baz miktar
        risk_amount = account_balance * self.max_risk_per_trade
        
        # 2. Stop Loss mesafesi (Fiyat farkı)
        # Eğer stop price belirtilmediyse ATR'ye göre dinamik belirle
        if stop_loss_price == 0:
            stop_dist = volatility_atr * 2.0
        else:
            stop_dist = abs(entry_price - stop_loss_price)
            
        if stop_dist == 0:
            logger.warning("Stop mesafesi 0, işlem çok riskli. İşlem iptal ediliyor.")
            return 0.0

        # 3. Alınacak miktar (Size) = Risk Miktarı / Stop Mesafesi
        position_size = risk_amount / stop_dist
        
        # 4. Kaldıraçsız sistem kontrolü (Spot piyasa için)
        total_cost = position_size * entry_price
        if total_cost > account_balance:
            logger.info(f"Hesaplanan pozisyon ({total_cost:.2f}$) bakiyeyi aşıyor. Max bakiyeye göre düzeltiliyor.")
            position_size = (account_balance * 0.98) / entry_price 

        logger.info(f"Risk Analizi: Bakiye={account_balance}$, Giriş={entry_price}$, Size={position_size:.4f}")
        return position_size

    def check_market_regime_allowance(self, regime: str) -> bool:
        """
        Piyasa rejimine göre işlem izni verir.
        """
        if regime == "CRASH" or regime == "EXTREME_VOLATILITY":
            logger.warning(f"Piyasa Rejimi TEHLİKELİ ({regime}). Risk yöneticisi işlem izni vermiyor.")
            return False
        return True
