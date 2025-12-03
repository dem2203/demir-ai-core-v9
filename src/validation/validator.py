from typing import Any, Dict
from src.validation.mock_detector import MockDataDetector
from src.validation.real_data_verifier import RealDataVerifier
from src.validation.signal_integrity import SignalIntegrityChecker
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DEMIR_AI_VALIDATOR")

class SignalValidator:
    """
    MASTER VALIDATOR CLASS
    Tüm doğrulama sınıflarını (Mock, RealData, Integrity) tek çatı altında toplar.
    Dış dünyadan gelen her veri önce buradan geçer.
    """

    @staticmethod
    def validate_incoming_data(data: Any) -> bool:
        """
        Borsadan gelen ham veriyi denetler.
        """
        # 1. Mock Veri Kontrolü
        if MockDataDetector.validate(data):
            logger.critical("BLOCKED: Mock/Fake data detected in input stream.")
            return False

        # 2. Veri Yapısı ve Zaman Kontrolü
        if not RealDataVerifier.verify_market_data(data):
            logger.error("BLOCKED: Data failed real-world verification (Time/Price physics).")
            return False

        return True

    @staticmethod
    def validate_outgoing_signal(signal: Dict) -> bool:
        """
        Botun ürettiği emri borsaya göndermeden önce denetler.
        """
        # 1. Sinyal Mantık Kontrolü
        if not SignalIntegrityChecker.check_signal_logic(signal):
            logger.error("BLOCKED: Signal logic is flawed. Execution aborted.")
            return False
            
        # 2. Sinyal İçeriğinde Mock Veri Var mı?
        if MockDataDetector.validate(signal):
            logger.critical("BLOCKED: Generated signal contains mock data patterns.")
            return False
            
        logger.info(f"VALIDATION SUCCESS: Signal {signal.get('symbol')} verified.")
        return True