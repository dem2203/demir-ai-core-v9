import asyncio
import logging
from src.brain.premium_signals import get_premium_generator

logging.basicConfig(level=logging.INFO)

async def test():
    generator = get_premium_generator()
    signal = await generator.generate("BTCUSDT")
    if signal:
        print("\n" + "="*50)
        print(generator.format_telegram_message(signal))
        print("="*50)
    else:
        print("No signal generated")

asyncio.run(test())
