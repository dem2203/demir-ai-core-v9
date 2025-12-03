import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

class RealDataVerifier:
    """
    Gerçek Veri Doğrulayıcı.
    Verinin zamansal ve fiziksel olarak gerçek dünya ile uyumlu olup olmadığını denetler.
    """
    
    MAX_DATA_AGE = 3600  # 1 saat (Railway için çok esnek)
    MIN_VOLATILITY_THRESHOLD = 0.000001  # Fiyat hiç oynamıyorsa şüphelidir

    @staticmethod
    def verify_market_data(data: Union[Dict, List[Dict]]) -> bool:
        """
        Gelen piyasa verisinin tutarlılığını kontrol eder.
        """
        # Liste gelirse son elemana bak (en güncel veri)
        if isinstance(data, list):
            if not data: return False
            latest = data[-1]
        else:
            latest = data

        # 1. Zaman Damgası Kontrolü (Freshness Check)
        timestamp = latest.get('timestamp')
        if not timestamp:
            logger.error("VALIDATION FAIL: Missing timestamp.")
            return False
            
        try:
            # Timestamp formatını anla (ISO veya Unix)
            if isinstance(timestamp, (int, float)):
                ts_time = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e11 else timestamp)
            else:
                ts_time = pd.to_datetime(timestamp)
                
            now = datetime.now()
            delay = (now - ts_time).total_seconds()
            
            # Gelecekten gelen veri? (Saat hatası veya manipülasyon)
            if delay < -5: 
                logger.warning(f"VALIDATION WARNING: Data is from the future? ({delay}s). Clock sync issue possible.")
                
            # Çok eski veri?
            if delay > RealDataVerifier.MAX_DATA_AGE:
                logger.error(f"VALIDATION FAIL: Data is stale. Delay: {delay:.2f}s > Limit: {RealDataVerifier.MAX_DATA_AGE}s")
                return False
                
        except Exception as e:
            logger.error(f"VALIDATION ERROR: Timestamp parsing failed: {e}")
            return False

        # 2. Fiyat Tutarlılık Kontrolü (Zero Price Check)
        price = float(latest.get('close', 0))
        if price <= 0:
            logger.critical(f"VALIDATION FAIL: Invalid price detected: {price}")
            return False

        # 3. Hacim Kontrolü (Opsiyonel ama iyi)
        volume = float(latest.get('volume', -1))
        if volume == 0:
            logger.warning("VALIDATION WARNING: Volume is zero. Market might be closed or illiquid.")

        return True

    @staticmethod
    def verify_volatility(prices: List[float]) -> bool:
        """
        Fiyatların doğal bir oynaklığa sahip olup olmadığını kontrol eder.
        """
        if len(prices) < 10: return True # Yeterli veri yoksa geç
        
        pct_changes = pd.Series(prices).pct_change().dropna()
        volatility = pct_changes.std()
        
        if volatility < RealDataVerifier.MIN_VOLATILITY_THRESHOLD:
            logger.error(f"VALIDATION FAIL: Volatility too low ({volatility}). Looks like hardcoded data.")
            return False
            
        return True