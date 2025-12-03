import os
import sys

class Config:
    """
    Railway Environment Variables Configuration.
    """
    
    # --- İZLENECEK COINLER (Merkezi Liste) ---
    # Buraya istediğin coini ekleyebilirsin.
    TARGET_COINS = ["BTC/USDT", "ETH/USDT", "LTC/USDT"] 
    
    # --- System Settings ---
    VERSION = os.getenv("VERSION", "11.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    PYTHON_VERSION = os.getenv("PYTHON_VERSION", "3.11.9")
    
    # --- Database (PostgreSQL) ---
    DATABASE_URL = os.getenv("DATABASE_URL")
    # Not: Database URL yoksa sistem durmasın, loglasın (Database deploy edilmemiş olabilir)
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL not found. Database features will be disabled.")

    # --- Exchange API Keys (CRYPTO) ---
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
    
    BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
    BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
    
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
    
    # --- Macro & Financial Data APIs ---
    ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    YAHOO_FINANCE_KEY = os.getenv("Yahoo_Finance_API_KEY")
    FINNHUB_KEY = os.getenv("Finnhub_API_KEY")
    TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_API_KEY")
    
    # --- Social & Sentiment APIs ---
    TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
    CRYPTOALERT_KEY = os.getenv("CRYPTOALERT_API_KEY")
    
    # --- On-Chain & NFT ---
    ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY")
    OPENSEA_KEY = os.getenv("OPENSEA_API_KEY")
    DEXCHECK_KEY = os.getenv("DEXCHECK_API_KEY")
    
    # --- Notification ---
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
    # Discord (Phase 4B)
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    
    # Email (Phase 4B)
    EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false")
    EMAIL_FROM = os.getenv("EMAIL_FROM")
    EMAIL_TO = os.getenv("EMAIL_TO")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    
    # --- Streamlit Settings ---
    STREAMLIT_SERVER_HEADLESS = os.getenv("STREAMLIT_SERVER_HEADLESS", "true")

    @staticmethod
    def validate_keys():
        """
        Kritik anahtarların varlığını kontrol eder.
        """
        print(">> Validating Configuration...")
        if not Config.BINANCE_API_KEY:
            print(">> WARNING: BINANCE_API_KEY is missing!")
        if not Config.TELEGRAM_TOKEN:
            print(">> WARNING: TELEGRAM_TOKEN is missing!")
        print(">> Configuration loaded.")

# Import edildiğinde otomatik kontrol yap
Config.validate_keys()
