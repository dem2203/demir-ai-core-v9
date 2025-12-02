import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

class MockDataDetector:
    """
    Kapsamlı Mock/Fake/Hardcoded Veri Tespit Sınıfı.
    Sisteme giren verinin 'yapay' olup olmadığını analiz eder.
    """

    FORBIDDEN_KEYWORDS = [
        "mock", "fake", "test", "demo", "sample", "prototype", 
        "fallback", "placeholder", "dummy", "simulation"
    ]

    @staticmethod
    def contains_forbidden_keywords(data: Any) -> bool:
        """Veri yapısı içinde yasaklı kelimeleri (recursive) arar."""
        if isinstance(data, str):
            if any(keyword in data.lower() for keyword in MockDataDetector.FORBIDDEN_KEYWORDS):
                logger.warning(f"SECURITY ALERT: Forbidden keyword detected in data: {data}")
                return True
        elif isinstance(data, dict):
            return any(MockDataDetector.contains_forbidden_keywords(v) for v in data.values())
        elif isinstance(data, list):
            return any(MockDataDetector.contains_forbidden_keywords(i) for i in data)
        return False

    @staticmethod
    def is_statistically_artificial(prices: List[float]) -> bool:
        """
        Fiyat verisinin 'elle yazılmış' gibi durup durmadığını kontrol eder.
        Gerçek piyasa verisi asla mükemmel standart sapmaya (0) sahip olmaz.
        """
        if not prices or len(prices) < 2:
            return False
            
        arr = np.array(prices, dtype=float)
        
        # 1. Kontrol: Varyans 0 ise veri sabittir (Hardcoded)
        if np.var(arr) == 0:
            logger.critical("DATA INTEGRITY FAIL: Variance is zero. Data is hardcoded/static.")
            return True

        # 2. Kontrol: Mükemmel lineer artış (Örn: 100, 101, 102...)
        diffs = np.diff(arr)
        if np.all(diffs == diffs[0]):
            logger.critical("DATA INTEGRITY FAIL: Perfect linear progression detected. Data is likely synthetic.")
            return True

        return False

    @classmethod
    def validate(cls, data: Any) -> bool:
        """
        Ana kontrol noktası.
        Eğer Mock veri tespit edilirse True, temizse False döner.
        """
        if cls.contains_forbidden_keywords(data):
            return True
            
        # Eğer veri bir DataFrame veya Fiyat Listesi ise istatistiksel kontrol yap
        if isinstance(data, pd.DataFrame) and 'close' in data.columns:
            if cls.is_statistically_artificial(data['close'].tolist()):
                return True
                
        return False