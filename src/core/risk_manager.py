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

    @staticmethod
    def calculate_smart_levels(entry_price: float, side: str, 
                             swing_low: float, swing_high: float, 
                             whale_support: float, whale_resistance: float,
                             magnet_price: float, atr: float) -> dict:
        """
        AKILLI SL/TP HESAPLAMA (Smart Money Concepts)
        
        SL Stratejisi:
        - LONG: Swing Low veya Whale Support'un hemen altı (Hangisi daha yakınsa ama çok yakın değilse)
        - SHORT: Swing High veya Whale Resistance'ın hemen üstü
        
        TP Stratejisi:
        - LONG: Liquidation Magnet veya Swing High (Direnç)
        - SHORT: Liquidation Magnet veya Swing Low (Destek)
        """
        sl_price = 0.0
        tp_price = 0.0
        setup_type = "Standard"
        
        # Güvenlik marjı (%0.2)
        margin = entry_price * 0.002
        
        if side == "BUY":
            # --- STOP LOSS (LONG) ---
            # 1. Tercih: Whale Support (Balina Duvarı)
            if whale_support > 0 and whale_support < entry_price:
                sl_price = whale_support - margin
                setup_type = "Whale Wall Defense"
            # 2. Tercih: Swing Low (Teknik Dip)
            elif swing_low > 0 and swing_low < entry_price:
                sl_price = swing_low - margin
                setup_type = "Swing Structure"
            # 3. Fallback: ATR (Eğer yapı yoksa)
            else:
                sl_price = entry_price - (atr * 2)
                setup_type = "Volatility Based"
            
            # --- TAKE PROFIT (LONG) ---
            # 1. Tercih: Liquidation Magnet (Fiyatın çekileceği yer)
            if magnet_price > entry_price:
                tp_price = magnet_price
            # 2. Tercih: Swing High (Direnç)
            elif swing_high > entry_price:
                tp_price = swing_high
            # 3. Fallback: Risk Reward 2:1
            else:
                risk = entry_price - sl_price
                tp_price = entry_price + (risk * 2.5)

        else: # SELL
            # --- STOP LOSS (SHORT) ---
            # 1. Tercih: Whale Resistance
            if whale_resistance > entry_price:
                sl_price = whale_resistance + margin
                setup_type = "Whale Wall Defense"
            # 2. Tercih: Swing High
            elif swing_high > entry_price:
                sl_price = swing_high + margin
                setup_type = "Swing Structure"
            # 3. Fallback: ATR
            else:
                sl_price = entry_price + (atr * 2)
                setup_type = "Volatility Based"
            
            # --- TAKE PROFIT (SHORT) ---
            # 1. Tercih: Liquidation Magnet
            if magnet_price > 0 and magnet_price < entry_price:
                tp_price = magnet_price
            # 2. Tercih: Swing Low
            elif swing_low > 0 and swing_low < entry_price:
                tp_price = swing_low
            # 3. Fallback: Risk Reward 2:1
            else:
                risk = sl_price - entry_price
                tp_price = entry_price - (risk * 2.5)
        
        return {
            "sl": sl_price,
            "tp": tp_price,
            "setup_type": setup_type,
            "risk_reward": abs(tp_price - entry_price) / abs(entry_price - sl_price) if sl_price != entry_price else 0
        }
