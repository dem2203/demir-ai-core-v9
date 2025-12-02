from typing import Dict
import logging

logger = logging.getLogger(__name__)

class SignalIntegrityChecker:
    """
    Üretilen Al/Sat sinyallerinin tutarlılığını ve mantığını denetler.
    Hatalı hesaplanmış bir sinyalin borsaya gitmesini engeller.
    """

    @staticmethod
    def check_signal_logic(signal: Dict) -> bool:
        """
        Sinyal mantığını doğrular.
        Örnek: LONG işleminde Take Profit > Entry Price > Stop Loss olmalıdır.
        """
        try:
            side = signal.get('side')
            entry = float(signal.get('entry_price', 0))
            tp = float(signal.get('tp_price', 0))
            sl = float(signal.get('sl_price', 0))
            confidence = float(signal.get('confidence', 0))

            # 1. Güven Skoru Kontrolü
            if not (0 <= confidence <= 100):
                logger.error(f"SIGNAL ERROR: Invalid confidence score: {confidence}")
                return False

            # 2. Fiyat Hiyerarşisi Kontrolü
            if side == "BUY" or side == "LONG":
                if not (tp > entry > sl):
                    logger.error(f"SIGNAL LOGIC FAIL (LONG): TP({tp}) > Entry({entry}) > SL({sl}) condition not met.")
                    return False
            
            elif side == "SELL" or side == "SHORT":
                if not (tp < entry < sl):
                    logger.error(f"SIGNAL LOGIC FAIL (SHORT): TP({tp}) < Entry({entry}) < SL({sl}) condition not met.")
                    return False
            
            return True

        except Exception as e:
            logger.error(f"Signal integrity check failed with exception: {str(e)}")
            return False