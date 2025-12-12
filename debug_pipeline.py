import asyncio
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure strict path
sys.path.append(os.getcwd())
load_dotenv()

from src.data_ingestion.macro_connector import MacroConnector
from src.brain.vision_analyst import VisionAnalyst
from src.data_ingestion.connectors.binance_connector import BinanceConnector
from src.config.settings import Config

# Mock Config for Safety (in case env vars missing locally)
Config.TARGET_COINS = ["BTC/USDT", "ETH/USDT"]

async def test_macro():
    print("\n--- TESTING MACRO CONNECTOR ---")
    macro = MacroConnector()
    data = macro.fetch_data()
    print("Macro Data:", data)
    
    df = await macro.fetch_macro_data()
    print("Macro DataFrame Head:")
    print(df.head())
    return data

async def test_vision():
    print("\n--- TESTING VISION ANALYST ---")
    vision = VisionAnalyst()
    print(f"Gemini Active: {vision.gemini_active}")
    print(f"OpenAI Active: {vision.openai_active}")
    
    # Create dummy dataframe
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': [50000 + i*10 for i in range(100)],
        'high': [50100 + i*10 for i in range(100)],
        'low': [49900 + i*10 for i in range(100)],
        'close': [50050 + i*10 for i in range(100)],
        'volume': [1000 for _ in range(100)]
    }).set_index('timestamp')
    
    result = vision.analyze_chart("BTC/USDT", df)
    print("Vision Analysis Result:", result)
    return result

async def test_binance_sync():
    print("\n--- TESTING BINANCE SYNC (Dashboard usage) ---")
    connector = BinanceConnector()
    print(f"Target Coins: {Config.TARGET_COINS}")
    
    for symbol in Config.TARGET_COINS:
        print(f"Fetching {symbol}...")
        df = connector.fetch_ohlcv(symbol, limit=10)
        if not df.empty:
            print(f"✅ {symbol}: {len(df)} rows")
            print(df.iloc[-1])
        else:
            print(f"❌ {symbol}: Empty DataFrame")

async def main():
    print("=== DEBUG PIPELINE START ===")
    
    try:
        await test_macro()
    except Exception as e:
        print(f"Macro Error: {e}")

    try:
        await test_vision()
    except Exception as e:
        print(f"Vision Error: {e}")
        
    try:
        await test_binance_sync()
    except Exception as e:
        print(f"BinanceSync Error: {e}")
        
    print("=== DEBUG PIPELINE END ===")

if __name__ == "__main__":
    asyncio.run(main())
