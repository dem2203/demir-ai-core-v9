import logging
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from src.config import Config

logger = logging.getLogger("CHART_CAPTURE")

class TradingViewCapture:
    """
    Captures TradingView chart screenshots for visual analysis
    """
    
    def __init__(self):
        self.charts_dir = Path(Config.CHARTS_DIR)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.browser = None
        
    async def capture_chart(self, symbol: str, timeframe: str = "15") -> str:
        """
        Capture TradingView chart screenshot
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            timeframe: Chart timeframe (e.g., "15" for 15m)
            
        Returns:
            Path to saved screenshot
        """
        try:
            # Format symbol for TradingView
            tv_symbol = f"BINANCE:{symbol.replace('/', '')}"
            url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
            
            logger.info(f"📸 Capturing chart: {tv_symbol} ({timeframe}m)")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={'width': 1280, 'height': 720})
                
                # Navigate to chart
                await page.goto(url, wait_until="networkidle", timeout=15000)
                
                # Wait for chart to load
                await asyncio.sleep(3)
                
                # Change timeframe (click timeframe button)
                try:
                    await page.click(f'button:has-text("{timeframe}")', timeout=3000)
                    await asyncio.sleep(1)
                except:
                    logger.warning(f"Could not set timeframe to {timeframe}m")
                
                # Take screenshot
                screenshot_path = self.charts_dir / f"{symbol}_{timeframe}m.png"
                await page.screenshot(path=str(screenshot_path), full_page=False)
                
                await browser.close()
                
                logger.info(f"✅ Screenshot saved: {screenshot_path.name}")
                return str(screenshot_path)
                
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None
    
    async def capture_with_retry(self, symbol: str, max_retries: int = 2) -> str:
        """Capture with retry logic"""
        for attempt in range(max_retries):
            result = await self.capture_chart(symbol)
            if result:
                return result
            if attempt < max_retries - 1:
                logger.warning(f"Retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(2)
        return None
