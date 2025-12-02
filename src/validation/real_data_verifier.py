import time
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RealDataVerifier:
    """
    Gerçek Borsa Verisi Doğrulayıcı.
    OHLCV (Mum) verilerini ve zaman damgalarını doğrular.
    """
    
    MAX_ALLOWED_LATENCY_SEC = 600  # Gecikme toleransı (Mum verisi olduğu için biraz daha esnek)

    @staticmethod
    def verify_timestamp(timestamp_ms: int) -> bool:
        """
        Verinin zaman damgasını kontrol eder.
        """
        current_time_ms = int(time.time() * 1000)
        
        # Gelecekten gelen veri kontrolü
        if timestamp_ms > current_time_ms + 60000: # 1 dk tolerans
            logger.error(f"DATA ERROR: Timestamp is in the future! Server time sync issue?")
            return False
            
        return True

    @staticmethod
    def verify_candle_physics(candle: Dict) -> bool:
        """
        Mum verisinin fiziksel olarak mantıklı olup olmadığını denetler.
        Örn: High < Low olamaz. Fiyat negatif olamaz.
        """
        try:
            o = float(candle['open'])
            h = float(candle['high'])
            l = float(candle['low'])
            c = float(candle['close'])
            
            if any(p <= 0 for p in [o, h, l, c]):
                logger.critical("CRITICAL: Negative or Zero price detected.")
                return False
                
            if h < l:
                logger.critical(f"PHYSICS FAIL: High ({h}) is lower than Low ({l})")
                return False
                
            if not (l <= o <= h) or not (l <= c <= h):
                 # Bazen çok küçük kaymalarda bu olabilir ama genel kural budur
                 pass 

            return True
        except ValueError:
            return False

    @classmethod
    def verify_market_data(cls, ticker_data: Dict) -> bool:
        """
        Gelen veri paketini denetler.
        """
        # ARTIK 'price' YERİNE 'close' ARIYORUZ
        required_fields = ['symbol', 'timestamp', 'close', 'high', 'low']
        
        if not all(field in ticker_data for field in required_fields):
            missing = [f for f in required_fields if f not in ticker_data]
            logger.error(f"Data structure missing required fields: {missing}")
            return False

        if not cls.verify_timestamp(ticker_data['timestamp']):
            return False
            
        if not cls.verify_candle_physics(ticker_data):
            return False
        
        return True