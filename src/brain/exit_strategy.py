import logging
from typing import Dict, Tuple, Optional

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartExit")

class SmartExitStrategy:
    """
    Bu modül, pozisyona girdikten sonra piyasayı 7/24 izler.
    Amacı: Karı maksimize etmek ve trend dönüşlerini (reversal) önceden sezerek
    kar erimeden pozisyondan çıkmaktır.
    
    Özellikler:
    1. Volatility-Based Trailing Stop (ATR İz Süren Stop)
    2. Breakeven Trigger (Zararsız Konuma Geçiş)
    3. Trend Exhaustion Exit (Yorgunluk Tespiti - RSI/Momentum)
    4. Time-Based Exit (Zaman Aşımı)
    """

    def __init__(self):
        self.trailing_stop_price = 0.0
        self.highest_price_seen = 0.0
        self.is_breakeven_active = False

    def calculate_exit_signal(self, 
                            current_price: float, 
                            entry_price: float, 
                            current_atr: float, 
                            rsi_value: float, 
                            bars_held: int) -> Tuple[bool, str, float]:
        """
        Her mum kapanışında veya her tikte çağrılır.
        
        Dönüş: (Çıkış Yapılsın mı? [True/False], Sebep [Str], Güncel Stop Fiyatı [Float])
        """
        
        # 1. Veri Doğrulama (Zero-Mock Policy)
        if current_price <= 0 or entry_price <= 0:
            return False, "Invalid Data", 0.0

        # Başlangıç değerlerini set et
        if self.highest_price_seen == 0.0:
            self.highest_price_seen = entry_price
        
        # Yeni zirve görüldü mü? (Long pozisyon için)
        if current_price > self.highest_price_seen:
            self.highest_price_seen = current_price

        # --- STRATEJİ 1: KAR KİLİTLEME (BREAKEVEN) ---
        # Eğer kar %1.5'u geçtiyse, Stop Loss'u giriş seviyesine çek (Risk-Free Trade)
        profit_pct = (current_price - entry_price) / entry_price
        if not self.is_breakeven_active and profit_pct > 0.015:
            self.trailing_stop_price = max(self.trailing_stop_price, entry_price * 1.001) # Girişin hafif üstü (Komisyon için)
            self.is_breakeven_active = True
            logger.info(f"Kar %1.5'u geçti. Stop Loss Giriş Seviyesine (Breakeven) çekildi: {self.trailing_stop_price}")

        # --- STRATEJİ 2: ATR TRAILING STOP (İZ SÜREN STOP) ---
        # Fiyat yükseldikçe stop da yukarı gelir. Asla aşağı inmez.
        # Standart piyasada 3 ATR, çok karlıysak 2 ATR geriden gelir.
        atr_multiplier = 3.0
        if profit_pct > 0.05: # %5 kardan sonra stopu sıkılaştır
            atr_multiplier = 2.0
            
        dynamic_stop = self.highest_price_seen - (current_atr * atr_multiplier)
        
        # Stop fiyatı sadece yukarı gidebilir
        if dynamic_stop > self.trailing_stop_price:
            self.trailing_stop_price = dynamic_stop

        # Fiyat stop seviyesinin altına düştü mü?
        if current_price < self.trailing_stop_price:
            return True, "TRAILING_STOP_HIT", self.trailing_stop_price

        # --- STRATEJİ 3: TREND EXHAUSTION (YORGUNLUK TESPİTİ - SİNYALCİ) ---
        # Fiyat yükseliyor ama RSI aşırı şiştiyse (Örn: RSI > 80)
        # Bu "Düşüş gelmeden hemen önceki" sinyaldir.
        if rsi_value > 80:
            # RSI çok şişti, çok agresif bir stop kullan (Örn: Mevcut fiyatın %1 altı)
            tight_stop = current_price * 0.99
            if tight_stop > self.trailing_stop_price:
                self.trailing_stop_price = tight_stop
                logger.info("RSI Aşırı Şişti (>80). Olası dönüş sinyali. Stop çok sıkılaştırıldı.")
                
            # Veya direkt kar al çık (Risk iştahına göre)
            # return True, "RSI_OVERBOUGHT_PANIC", current_price

        # --- STRATEJİ 4: TIME-BASED EXIT (ZAMAN AŞIMI) ---
        # 48 mumdur (Örn: 4 saatlikte 8 gün) pozisyondayız ve kar sadece %1 ise parayı bağlama, çık.
        if bars_held > 48 and profit_pct < 0.01:
             return True, "TIME_LIMIT_EXCEEDED_STAGNANT", current_price

        # Çıkış yok, izlemeye devam
        return False, "HOLD", self.trailing_stop_price

    def reset(self):
        """Yeni pozisyon için hafızayı temizler."""
        self.trailing_stop_price = 0.0
        self.highest_price_seen = 0.0
        self.is_breakeven_active = False
