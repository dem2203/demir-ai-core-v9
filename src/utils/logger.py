import logging
import sys
import os

def setup_logging(name="DEMIR_AI"):
    """
    Setup standard logging configuration.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler (Optional, if in dev)
        # file_handler = logging.FileHandler("demir_ai.log")
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
        
    return logger
