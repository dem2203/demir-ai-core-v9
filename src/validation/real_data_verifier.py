import time
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RealDataVerifier:
    """
    Gerçek Borsa Verisi Doğrulayıcı.
    Fiyat doğrulamasını ve zaman damgası (Timestamp) kontrollerini yapar.
    """
    
    MAX_ALLOWED_LATENCY_SEC = 60  # Maksimum kabul edilebilir gecikme (sn)

    @staticmethod
    def verify_timestamp(timestamp_ms: int) -> bool:
        """
        Verinin zaman damgasını kontrol eder.
        Veri çok eskiyse veya gelecekten geliyorsa (saat hatası) reddeder.
        """
        current_time_ms = int(time.time() * 1000)
        diff = abs(current_time_ms - timestamp_ms)
        
        # Gelecekten gelen veri kontrolü (Server saat farkı toleransı: 5sn)
        if timestamp_ms > current_time_ms + 5000:
            logger.error(f"DATA ERROR: Timestamp is in the future! Diff: {diff}ms")
            return False
            
        # Bayat veri kontrolü
        if diff > (RealDataVerifier.MAX_ALLOWED_LATENCY_SEC * 1000):
            logger.error(f"DATA ERROR: Data is stale/old. Latency: {diff/1000}s")
            return False
            
        return True

    @staticmethod
    def verify_price_physics(price: float) -> bool:
        """
        Fiyatın mantıksal fizik kurallarına uygunluğunu denetler.
        Negatif veya Sıfır fiyat kripto piyasasında olamaz.
        """
        if not isinstance(price, (int, float)):
            logger.error(f"TYPE ERROR: Price is not a number: {type(price)}")
            return False
            
        if price <= 0:
            logger.critical(f"CRITICAL: Invalid price detected: {price}. Price must be positive.")
            return False
            
        return True

    @classmethod
    def verify_market_data(cls, ticker_data: Dict) -> bool:
        """
        Tek bir veri paketini (Ticker) tam kontrolden geçirir.
        """
        required_fields = ['symbol', 'price', 'timestamp']
        if not all(field in ticker_data for field in required_fields):
            logger.error("Data structure missing required fields.")
            return False

        is_time_valid = cls.verify_timestamp(ticker_data['timestamp'])
        is_price_valid = cls.verify_price_physics(ticker_data['price'])
        
        return is_time_valid and is_price_valid