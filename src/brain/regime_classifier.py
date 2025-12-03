import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("REGIME_CLASSIFIER")

class RegimeClassifier:
    """
    PIYASA REJİMİ SINIFLANDIRICISI
    
    Botun "Nerede savaştığını" anlamasını sağlar.
    - TRENDING: Güçlü yön (ADX > 25)
    - RANGING: Yönsüz/Yatay (ADX < 25, Bollinger Bandı Yatay)
    - VOLATILE: Yüksek risk (ATR ve Bollinger Width çok yüksek)
    """
    
    @staticmethod
    def identify_regime(df: pd.DataFrame) -> str:
        """
        Verilen veri setinin son durumuna göre piyasa rejimini belirler.
        """
        if df is None or len(df) < 20: return "UNKNOWN"
        
        last = df.iloc[-1]
        
        # 1. VOLATİLİTE KONTROLÜ (FIRTINA VAR MI?)
        # Bollinger bant genişliği son 50 mumun ortalamasının 2 katıysa piyasa patlıyordur.
        avg_bb_width = df['bb_width'].rolling(50).mean().iloc[-1]
        if last['bb_width'] > avg_bb_width * 1.8:
            return "VOLATILE" # Çok tehlikeli, işlem yapma veya stopları genişlet
            
        # 2. TREND KONTROLÜ (YÖN VAR MI?)
        # ADX 25'in üzerindeyse güçlü bir trend vardır.
        if last['adx'] > 25:
            if last['close'] > last['vwap']:
                return "TRENDING_BULL" # Yükseliş Trendi
            else:
                return "TRENDING_BEAR" # Düşüş Trendi
        
        # 3. YATAY PİYASA (TESTERE)
        # Trend yok ve volatilite normalse piyasa yataydır.
        return "RANGING"

    @staticmethod
    def get_risk_adjustment(regime: str) -> dict:
        """
        Rejime göre botun ayarlarını (Risk, Güven Eşiği) değiştirir.
        Bu kısım "İnsan Üstü" adaptasyonun kalbidir.
        """
        if regime == "TRENDING_BULL" or regime == "TRENDING_BEAR":
            return {
                "confidence_threshold": 0.55, # Trend varsa %55 güven yeterli
                "stop_loss_multiplier": 2.0,  # Trendde stopu biraz geniş tut
                "trade_allowed": True
            }
        elif regime == "RANGING":
            return {
                "confidence_threshold": 0.75, # Yatay piyasada çok emin olmadan girme!
                "stop_loss_multiplier": 1.0,  # Dar stop
                "trade_allowed": False        # Veya tamamen yasakla (Güvenli Mod)
            }
        elif regime == "VOLATILE":
            return {
                "confidence_threshold": 0.85, # Kaosta sadece mükemmel fırsatları al
                "stop_loss_multiplier": 3.0,  # Çok geniş stop (İğne atıp dönmesin diye)
                "trade_allowed": False        # Genelde nakitte kalmak iyidir
            }
        
        return {"confidence_threshold": 0.60, "stop_loss_multiplier": 1.5, "trade_allowed": True}
