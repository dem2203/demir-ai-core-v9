# -*- coding: utf-8 -*-
"""
DEMIR AI - WEB SCRAPER COLLECTION
=================================
API Key gerektiren verileri web scraping ile toplayan modül.
Playwright kullanarak gerçek tarayıcı ile veri kazır.

⚠️ RAILWAY WARNING: Playwright is DISABLED on Railway due to thread limits.
   All scraper methods return fallback data on Railway.

Hedef Siteler:
1. CoinGlass - Exchange Flow, Options
2. DefiLlama - Stablecoin Supply
3. TradingView - CME Gap
4. SoSoValue - ETF Flow
5. Blockchain.com - Network Metrics
"""
import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, Optional

# Check if running on Railway (limited resources)
IS_RAILWAY = os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY_SERVICE_NAME')

# Only import Playwright if NOT on Railway
PLAYWRIGHT_AVAILABLE = False
if not IS_RAILWAY:
    try:
        from playwright.async_api import async_playwright, Browser, Page
        PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger("WEB_SCRAPERS")

if IS_RAILWAY:
    logger.warning("🚫 Running on Railway - Playwright scrapers DISABLED to prevent thread exhaustion")


class WebScraperManager:
    """
    Tüm web scraperları yöneten ana sınıf.
    Tek bir browser instance paylaşılır.
    
    ⚠️ On Railway: All methods return fallback data (no browser launching)
    """
    
    def __init__(self):
        self._browser = None
        self._playwright = None
        self._initialized = False
        self._disabled = IS_RAILWAY or not PLAYWRIGHT_AVAILABLE
        
        if self._disabled:
            logger.info("📴 WebScraperManager: Browser scraping disabled (Railway mode or no Playwright)")
    
    async def _ensure_browser(self):
        """Browser'ı başlat veya var olanı kullan."""
        if self._disabled:
            return  # Skip browser on Railway
            
        if not self._initialized and PLAYWRIGHT_AVAILABLE:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            self._initialized = True
            logger.info("🌐 Web Scraper Browser started")
    
    async def close(self):
        """Browser'ı kapat."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False
    
    async def _new_page(self):
        """Yeni sayfa oluştur. Returns None on Railway."""
        if self._disabled:
            return None  # Railway mode - no browser
            
        await self._ensure_browser()
        if not self._browser:
            return None
        context = await self._browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        return await context.new_page()

    # =========================================
    # 1. EXCHANGE FLOW (CoinGlass)
    # =========================================
    
    async def scrape_exchange_flow(self, symbol: str = "BTC") -> Dict:
        """
        CoinGlass'tan exchange inflow/outflow verisi kazı.
        URL: https://www.coinglass.com/exchange-flow
        """
        # Railway fallback - return neutral data
        if self._disabled:
            return {'inflow': 0, 'outflow': 0, 'netflow': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'inflow': 0, 'outflow': 0, 'netflow': 0, 'source': 'fallback'}
                
            await page.goto(f'https://www.coinglass.com/exchange-flow/{symbol}', wait_until='networkidle', timeout=30000)
            
            # Sayfanın yüklenmesini bekle
            await asyncio.sleep(2)
            
            # Veriyi kazı
            data = await page.evaluate('''() => {
                const result = {inflow: 0, outflow: 0, netflow: 0};
                
                // Tablo verisini bul
                const tables = document.querySelectorAll('table');
                if (tables.length > 0) {
                    const rows = tables[0].querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const cells = row.querySelectorAll('td');
                        if (cells.length >= 3) {
                            const inflow = parseFloat(cells[1]?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                            const outflow = parseFloat(cells[2]?.textContent?.replace(/[^0-9.-]/g, '') || 0);
                            result.inflow += inflow;
                            result.outflow += outflow;
                        }
                    });
                }
                
                // Alternatif: Özet kısmını bul
                const summaryElements = document.querySelectorAll('[class*="summary"], [class*="total"]');
                summaryElements.forEach(el => {
                    const text = el.textContent || '';
                    if (text.includes('Inflow')) {
                        const match = text.match(/[\d,.]+/);
                        if (match) result.inflow = parseFloat(match[0].replace(/,/g, ''));
                    }
                    if (text.includes('Outflow')) {
                        const match = text.match(/[\d,.]+/);
                        if (match) result.outflow = parseFloat(match[0].replace(/,/g, ''));
                    }
                });
                
                result.netflow = result.inflow - result.outflow;
                return result;
            }''')
            
            await page.close()
            data['source'] = 'live'
            logger.info(f"📊 Exchange Flow scraped: In={data['inflow']}, Out={data['outflow']}")
            return data
            
        except Exception as e:
            logger.warning(f"Exchange flow scrape failed: {e}")
            return {'inflow': 0, 'outflow': 0, 'netflow': 0}

    # =========================================
    # 2. STABLECOIN SUPPLY (DefiLlama)
    # =========================================
    
    async def scrape_stablecoin_supply(self) -> Dict:
        """
        DefiLlama'dan stablecoin supply değişikliklerini kazı.
        URL: https://defillama.com/stablecoins
        """
        if self._disabled:
            return {'usdt_change': 0, 'usdc_change': 0, 'total_supply': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'usdt_change': 0, 'usdc_change': 0, 'total_supply': 0, 'source': 'fallback'}
            await page.goto('https://defillama.com/stablecoins', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(3)
            
            data = await page.evaluate('''() => {
                const result = {usdt_change: 0, usdc_change: 0, total_supply: 0};
                
                // Tablo satırlarını bul
                const rows = document.querySelectorAll('tr');
                rows.forEach(row => {
                    const text = row.textContent || '';
                    
                    // USDT
                    if (text.includes('Tether') || text.includes('USDT')) {
                        const changeMatch = text.match(/([+-]?[\d.]+%)/);
                        if (changeMatch) {
                            result.usdt_change = parseFloat(changeMatch[1].replace('%', ''));
                        }
                    }
                    
                    // USDC
                    if (text.includes('USD Coin') || text.includes('USDC')) {
                        const changeMatch = text.match(/([+-]?[\d.]+%)/);
                        if (changeMatch) {
                            result.usdc_change = parseFloat(changeMatch[1].replace('%', ''));
                        }
                    }
                });
                
                // Toplam supply
                const totalElements = document.querySelectorAll('[class*="total"], h1, h2');
                totalElements.forEach(el => {
                    const text = el.textContent || '';
                    const match = text.match(/\$?([\d,.]+)\s*[BM]/i);
                    if (match) {
                        let value = parseFloat(match[1].replace(/,/g, ''));
                        if (text.includes('B')) value *= 1e9;
                        if (text.includes('M')) value *= 1e6;
                        result.total_supply = value;
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            logger.info(f"💵 Stablecoin Supply scraped: USDT={data['usdt_change']}%, USDC={data['usdc_change']}%")
            return data
            
        except Exception as e:
            logger.warning(f"Stablecoin scrape failed: {e}")
            return {'usdt_change': 0, 'usdc_change': 0, 'total_supply': 0}

    # =========================================
    # 3. OPTIONS DATA (CoinGlass)
    # =========================================
    
    async def scrape_options_data(self, symbol: str = "BTC") -> Dict:
        """
        CoinGlass'tan options put/call ratio kazı.
        URL: https://www.coinglass.com/options
        """
        if self._disabled:
            return {'put_call': 1.0, 'max_pain': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'put_call': 1.0, 'max_pain': 0, 'source': 'fallback'}
            await page.goto(f'https://www.coinglass.com/options/{symbol}', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(2)
            
            data = await page.evaluate('''() => {
                const result = {put_call: 1.0, max_pain: 0, call_oi: 0, put_oi: 0};
                
                // Put/Call ratio bul
                const elements = document.querySelectorAll('*');
                elements.forEach(el => {
                    const text = el.textContent || '';
                    
                    // Put/Call ratio
                    if (text.includes('Put/Call') || text.includes('P/C Ratio')) {
                        const match = text.match(/([\d.]+)/);
                        if (match) result.put_call = parseFloat(match[1]);
                    }
                    
                    // Max Pain
                    if (text.includes('Max Pain')) {
                        const match = text.match(/\$?([\d,]+)/);
                        if (match) result.max_pain = parseFloat(match[1].replace(/,/g, ''));
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            logger.info(f"📈 Options scraped: P/C={data['put_call']}, MaxPain=${data['max_pain']}")
            return data
            
        except Exception as e:
            logger.warning(f"Options scrape failed: {e}")
            return {'put_call': 1.0, 'max_pain': 0}

    # =========================================
    # 4. CME GAP (TradingView Widget)
    # =========================================
    
    async def scrape_cme_gap(self) -> Dict:
        """
        CME BTC Futures gap analizi.
        Cuma kapanış vs Pazartesi açılış karşılaştırması.
        """
        if self._disabled:
            return {'gap_price': 0, 'filled': True, 'direction': '', 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'gap_price': 0, 'filled': True, 'direction': '', 'source': 'fallback'}
            # CME BTC verisi için Binance Futures kullan (CME direkt erişilemez)
            await page.goto('https://www.binance.com/en/futures/BTCUSDT', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(2)
            
            # Bu basitleştirilmiş bir yaklaşım
            # Gerçek CME gap için hafta sonu kapanış/açılış karşılaştırması gerekir
            data = await page.evaluate('''() => {
                const result = {gap_price: 0, filled: true, direction: ''};
                
                // Fiyat bilgisini al
                const priceElements = document.querySelectorAll('[class*="price"], [class*="ticker"]');
                priceElements.forEach(el => {
                    const text = el.textContent || '';
                    const match = text.match(/[\d,]+\.?\d*/);
                    if (match && parseFloat(match[0].replace(/,/g, '')) > 10000) {
                        result.gap_price = parseFloat(match[0].replace(/,/g, ''));
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            return data
            
        except Exception as e:
            logger.warning(f"CME gap scrape failed: {e}")
            return {'gap_price': 0, 'filled': True, 'direction': ''}

    # =========================================
    # 5. ETF FLOW (SoSoValue)
    # =========================================
    
    async def scrape_etf_flow(self) -> Dict:
        """
        SoSoValue'dan Bitcoin ETF akış verisi kazı.
        URL: https://sosovalue.xyz/assets/etf/us-btc-spot
        """
        if self._disabled:
            return {'daily_flow': 0, 'total_aum': 0, 'gbtc_premium': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'daily_flow': 0, 'total_aum': 0, 'gbtc_premium': 0, 'source': 'fallback'}
            await page.goto('https://sosovalue.xyz/assets/etf/us-btc-spot', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(3)
            
            data = await page.evaluate('''() => {
                const result = {daily_flow: 0, total_aum: 0, gbtc_premium: 0};
                
                // Günlük akış bul
                const elements = document.querySelectorAll('*');
                elements.forEach(el => {
                    const text = el.textContent || '';
                    
                    // Daily net flow
                    if (text.includes('Daily') && text.includes('Flow')) {
                        const match = text.match(/([+-]?\$?[\d,.]+)\s*M/i);
                        if (match) {
                            result.daily_flow = parseFloat(match[1].replace(/[$,]/g, ''));
                        }
                    }
                    
                    // Total AUM
                    if (text.includes('AUM') || text.includes('Total')) {
                        const match = text.match(/\$?([\d,.]+)\s*B/i);
                        if (match) {
                            result.total_aum = parseFloat(match[1].replace(/,/g, '')) * 1000; // Convert to M
                        }
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            logger.info(f"📊 ETF Flow scraped: Daily=${data['daily_flow']}M, AUM=${data['total_aum']}M")
            return data
            
        except Exception as e:
            logger.warning(f"ETF flow scrape failed: {e}")
            return {'daily_flow': 0, 'total_aum': 0, 'gbtc_premium': 0}

    # =========================================
    # 6. NETWORK METRICS (Blockchain.com)
    # =========================================
    
    async def scrape_network_metrics(self) -> Dict:
        """
        Blockchain.com'dan ağ metrikleri kazı.
        URL: https://www.blockchain.com/explorer/charts
        """
        if self._disabled:
            return {'hash_rate': 0, 'active_addresses': 0, 'hash_change': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'hash_rate': 0, 'active_addresses': 0, 'hash_change': 0, 'source': 'fallback'}
            await page.goto('https://www.blockchain.com/explorer/charts/hash-rate', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(2)
            
            data = await page.evaluate('''() => {
                const result = {hash_rate: 0, active_addresses: 0, hash_change: 0};
                
                // Hash rate bul
                const elements = document.querySelectorAll('*');
                elements.forEach(el => {
                    const text = el.textContent || '';
                    
                    // Hash rate
                    if (text.includes('EH/s') || text.includes('Hash Rate')) {
                        const match = text.match(/([\d,.]+)\s*EH/i);
                        if (match) {
                            result.hash_rate = parseFloat(match[1].replace(/,/g, ''));
                        }
                    }
                    
                    // Change
                    if (text.includes('%')) {
                        const match = text.match(/([+-]?[\d.]+)%/);
                        if (match) {
                            result.hash_change = parseFloat(match[1]);
                        }
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            logger.info(f"⛏️ Network Metrics scraped: HashRate={data['hash_rate']}EH/s")
            return data
            
        except Exception as e:
            logger.warning(f"Network metrics scrape failed: {e}")
            return {'hash_rate': 0, 'active_addresses': 0, 'hash_change': 0}

    # =========================================
    # 7. GRAYSCALE PREMIUM
    # =========================================
    
    async def scrape_grayscale_premium(self) -> Dict:
        """
        Grayscale GBTC premium/discount kazı.
        """
        if self._disabled:
            return {'premium': 0, 'source': 'fallback'}
            
        try:
            page = await self._new_page()
            if not page:
                return {'premium': 0, 'source': 'fallback'}
            await page.goto('https://ycharts.com/companies/GBTC/discount_or_premium_to_nav', wait_until='networkidle', timeout=30000)
            
            await asyncio.sleep(2)
            
            data = await page.evaluate('''() => {
                const result = {premium: 0};
                
                const elements = document.querySelectorAll('*');
                elements.forEach(el => {
                    const text = el.textContent || '';
                    
                    if (text.includes('%') && (text.includes('Premium') || text.includes('Discount'))) {
                        const match = text.match(/([+-]?[\d.]+)%/);
                        if (match) {
                            result.premium = parseFloat(match[1]);
                        }
                    }
                });
                
                return result;
            }''')
            
            await page.close()
            logger.info(f"📊 Grayscale Premium: {data['premium']}%")
            return data
            
        except Exception as e:
            logger.warning(f"Grayscale scrape failed: {e}")
            return {'premium': 0}


# =========================================
# GLOBAL INSTANCE
# =========================================

_scraper_manager: Optional[WebScraperManager] = None

def get_web_scraper() -> WebScraperManager:
    """Get or create scraper manager instance."""
    global _scraper_manager
    if _scraper_manager is None:
        _scraper_manager = WebScraperManager()
    return _scraper_manager
