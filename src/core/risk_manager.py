import logging
import numpy as np

logger = logging.getLogger("RISK_MANAGER_PRO")

class RiskManager:
    """
    DEMIR AI V19.0 - DYNAMIC RISK MANAGER
    
    Yenilikler:
    1. Kelly Criterion: Güvene dayalı dinamik pozisyon büyüklüğü.
    2. Chandelier Exit: Volatilite tabanlı akıllı takip eden stop.
    """
    
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.win_rate = 0.55 # Varsayılan (Zamanla güncellenebilir)
        self.risk_reward_ratio = 2.0 # Hedef R:R

    def calculate_kelly_size(self, confidence: float) -> float:
        """
        Kelly Kriteri Formülü: f* = (bp - q) / b
        b = oran (odds) - biz burada Risk/Reward kullanıyoruz
        p = kazanma olasılığı (confidence)
        q = kaybetme olasılığı (1-p)
        """
        # Güven skorunu olasılığa çevir (0-100 -> 0.5-0.95)
        # 50'nin altı zaten işlem açmaz.
        p = max(0.51, min(0.95, confidence / 100.0))
        q = 1 - p
        b = self.risk_reward_ratio
        
        kelly_fraction = (b * p - q) / b
        
        # Tam Kelly çok risklidir, "Half Kelly" kullanıyoruz (Daha güvenli)
        safe_kelly = kelly_fraction * 0.5
        
        # Sınırlar: Asla kasanın %20'sinden fazlasını veya %1'inden azını riske atma
        final_size = max(0.01, min(0.20, safe_kelly))
        
        return round(final_size * 100, 2) # Yüzde olarak dön

    @staticmethod
    def calculate_chandelier_exit(high_prices: list, atr: float, multiplier: float = 3.0, side: str = "BUY") -> float:
        """
        Chandelier Exit: En yüksek tepeden ATR katı kadar aşağıda stop.
        """
        if not high_prices: return 0.0
        
        highest_high = max(high_prices)
        lowest_low = min(high_prices)
        
        if side == "BUY":
            # Long için: En yüksek tepeden aşağı sarkma
            stop_level = highest_high - (atr * multiplier)
        else:
            # Short için: En düşük dipten yukarı sarkma
            stop_level = lowest_low + (atr * multiplier)
            
        return stop_level
