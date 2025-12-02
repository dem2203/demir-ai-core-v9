import logging
import sys
import os

def setup_logger():
    """
    Profesyonel Loglama Yapılandırması.
    Hem ekrana (Console) hem de dosyaya (File) yazar.
    Format: [ZAMAN] [SEVİYE] [MODÜL] Mesaj
    """
    
    # Log formatı
    log_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger ayarı
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 1. Konsol Çıktısı (Railway Logları için)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 2. Dosya Çıktısı (Kalıcı kayıt için - Opsiyonel)
    # Railway'de disk geçicidir ama debug için iyidir.
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    file_handler = logging.FileHandler("logs/system.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logging.info("LOGGER SYSTEM INITIALIZED.")