
import asyncio
import logging
from src.brain.liquidation_hunter import LiquidationHunter
from src.brain.onchain_intel import OnChainIntelligence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VERIFICATION")

async def test_zero_mock():
    logger.info("🚀 STARTING ZERO-MOCK VERIFICATION")
    
    # 1. Test OnChainIntelligence (Whale Tracker)
    logger.info("--- Testing OnChainIntelligence (Whale Tracker) ---")
    onchain = OnChainIntelligence()
    # This should trigger lazy load of WhaleTracker and start WebSocket
    whale_data = await onchain.detect_whale_trades("BTCUSDT")
    logger.info(f"Whale Data Result: {whale_data}")
    
    if whale_data.get('direction'):
        logger.info("✅ OnChainIntelligence returned structured data")
    else:
        logger.error("❌ OnChainIntelligence failed to return valid data")

    # 2. Test LiquidationHunter (CoinGlass Scraper)
    logger.info("\n--- Testing LiquidationHunter (CoinGlass Scraper) ---")
    hunter = LiquidationHunter()
    # This should trigger lazy load of CoinGlassScraper and start Browser
    liq_data = await hunter.calculate_liquidation_levels("BTCUSDT")
    logger.info(f"Liquidation Data Result keys: {liq_data.keys()}")
    
    if 'real_data' in liq_data:
        real = liq_data['real_data']
        logger.info(f"✅ Real Data Found: {real}")
    else:
        logger.error("❌ Real Data NOT found in LiquidationHunter result")

    # Cleanup
    from src.brain.whale_tracker import get_whale_tracker
    from src.brain.coinglass_scraper import get_cg_scraper
    
    logger.info("Cleaning up...")
    await get_whale_tracker().stop()
    await get_cg_scraper().close()
    await onchain.close()
    await hunter.close() # if it has close method

if __name__ == "__main__":
    asyncio.run(test_zero_mock())
