import os
import logging

class Config:
    """
    Central Configuration for Demir AI v12 "Phoenix"
    """
    # System
    VERSION = "12.0.0"
    ENV = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG_MODE", "True").lower() == "true"
    
    # Trading Focus
    TARGET_PAIRS = ["BTC/USDT", "ETH/USDT"]
    TIMEFRAME = "1h"
    
    # Binance Futures
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    # AI Services
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Gemini
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Claude
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # GPT-4
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # DeepSeek
    
    # Market Data APIs (for fallback)
    TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
    
    # Notifications
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHARTS_DIR = os.path.join(DATA_DIR, "charts")

    @classmethod
    def validate(cls):
        missing = []
        if not cls.BINANCE_API_KEY: missing.append("BINANCE_API_KEY")
        if not cls.BINANCE_API_SECRET: missing.append("BINANCE_API_SECRET")
        if not cls.TELEGRAM_TOKEN: missing.append("TELEGRAM_TOKEN")
        if not cls.GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY (Gemini)")
        if not cls.ANTHROPIC_API_KEY: missing.append("ANTHROPIC_API_KEY (Claude)")
        
        if missing:
            logging.warning(f"⚠️ Missing Critical Config: {', '.join(missing)}")
            return False
        return True
