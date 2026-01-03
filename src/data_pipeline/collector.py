# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - DATA COLLECTOR
==============================
Gerçek profesyonel quant sistemi için veri altyapısı.

ÖZELLİKLER:
- Binance Futures'dan 2 yıllık dakikalık veri indir
- Parquet formatında kaydet (hızlı okuma)
- Artımlı güncelleme (sadece eksik veriyi indir)
- Rate limit koruması

KURAL: ASLA MOCK/FALLBACK VERİ YOK - SADECE GERÇEK VERİ!

Author: DEMIR AI Team
Date: 2026-01-03
"""
import asyncio
import aiohttp
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import time

logger = logging.getLogger("DATA_COLLECTOR")

# Konfigürasyon
DATA_DIR = Path("data/raw")
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
RATE_LIMIT_PER_MIN = 1200  # Binance limiti


class DataCollector:
    """
    Profesyonel veri toplama sistemi.
    
    2 yıllık dakikalık veri = ~1 milyon mum/coin
    Her mum: open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote
    """
    
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._request_count = 0
        self._last_reset = time.time()
    
    async def download_symbol(
        self, 
        symbol: str, 
        interval: str = "1m",
        days: int = 730  # 2 yıl
    ) -> pd.DataFrame:
        """
        Sembol için tarihsel veri indir.
        
        Args:
            symbol: BTCUSDT, ETHUSDT, vs.
            interval: 1m, 5m, 15m, 1h, 4h, 1d
            days: Kaç günlük veri
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"📊 Downloading {symbol} - {days} days of {interval} data...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            while current_start < end_ms:
                # Rate limit kontrolü
                await self._check_rate_limit()
                
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current_start,
                    "limit": 1500  # Max per request
                }
                
                try:
                    async with session.get(BINANCE_FUTURES_URL, params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if not data:
                                break
                            
                            all_data.extend(data)
                            
                            # Sonraki batch için start time güncelle
                            last_time = data[-1][0]
                            current_start = last_time + 60000  # 1 dakika sonra
                            
                            # Progress log
                            progress = (current_start - int(start_time.timestamp() * 1000)) / (end_ms - int(start_time.timestamp() * 1000))
                            if len(all_data) % 10000 == 0:
                                logger.info(f"  Progress: {progress*100:.1f}% ({len(all_data)} candles)")
                            
                        elif resp.status == 429:
                            logger.warning("Rate limited! Waiting 60s...")
                            await asyncio.sleep(60)
                        else:
                            logger.error(f"API error: {resp.status}")
                            break
                            
                except Exception as e:
                    logger.error(f"Request error: {e}")
                    await asyncio.sleep(5)
        
        if not all_data:
            logger.error(f"No data received for {symbol}")
            return pd.DataFrame()
        
        # DataFrame oluştur
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 
            'taker_buy_volume', 'taker_buy_quote', 'ignore'
        ])
        
        # Type conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_volume', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        df['trades'] = df['trades'].astype(int)
        
        # Gereksiz kolonları at
        df = df.drop(['close_time', 'ignore'], axis=1)
        
        # Kaydet
        filepath = DATA_DIR / f"{symbol}_{interval}.parquet"
        df.to_parquet(filepath, index=False)
        
        logger.info(f"✅ {symbol}: {len(df)} candles saved to {filepath}")
        return df
    
    async def download_all_symbols(
        self, 
        symbols: List[str], 
        interval: str = "1m",
        days: int = 730
    ):
        """Tüm semboller için veri indir."""
        logger.info(f"📦 Downloading {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            await self.download_symbol(symbol, interval, days)
            await asyncio.sleep(1)  # Semboller arası bekleme
        
        logger.info("✅ All symbols downloaded!")
    
    def load_symbol(self, symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Kayıtlı veriyi yükle."""
        filepath = DATA_DIR / f"{symbol}_{interval}.parquet"
        
        if not filepath.exists():
            logger.warning(f"No data file for {symbol}")
            return None
        
        df = pd.read_parquet(filepath)
        logger.info(f"📂 Loaded {symbol}: {len(df)} candles")
        return df
    
    async def update_symbol(self, symbol: str, interval: str = "1m") -> pd.DataFrame:
        """Mevcut veriyi güncelle (sadece eksik kısmı indir)."""
        existing = self.load_symbol(symbol, interval)
        
        if existing is None or len(existing) == 0:
            # Veri yok, sıfırdan indir
            return await self.download_symbol(symbol, interval)
        
        # Son zaman damgasını bul
        last_time = existing['timestamp'].max()
        days_missing = (datetime.now() - last_time).days + 1
        
        if days_missing <= 0:
            logger.info(f"{symbol} is up to date")
            return existing
        
        # Sadece eksik kısmı indir
        logger.info(f"Updating {symbol}: {days_missing} days missing")
        new_data = await self.download_symbol(symbol, interval, days=days_missing)
        
        if len(new_data) > 0:
            # Birleştir ve duplicate kaldır
            combined = pd.concat([existing, new_data])
            combined = combined.drop_duplicates(subset=['timestamp'])
            combined = combined.sort_values('timestamp')
            
            # Kaydet
            filepath = DATA_DIR / f"{symbol}_{interval}.parquet"
            combined.to_parquet(filepath, index=False)
            
            logger.info(f"✅ {symbol} updated: {len(combined)} total candles")
            return combined
        
        return existing
    
    async def _check_rate_limit(self):
        """Rate limit kontrolü."""
        now = time.time()
        
        # Reset counter every minute
        if now - self._last_reset > 60:
            self._request_count = 0
            self._last_reset = now
        
        self._request_count += 1
        
        if self._request_count >= RATE_LIMIT_PER_MIN:
            wait_time = 60 - (now - self._last_reset)
            if wait_time > 0:
                logger.info(f"Rate limit approaching, waiting {wait_time:.0f}s")
                await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = time.time()
    
    def get_data_summary(self) -> dict:
        """Mevcut veri durumu özeti."""
        summary = {}
        
        for filepath in DATA_DIR.glob("*.parquet"):
            symbol = filepath.stem.split("_")[0]
            df = pd.read_parquet(filepath)
            summary[symbol] = {
                "candles": len(df),
                "start": df['timestamp'].min().isoformat() if len(df) > 0 else None,
                "end": df['timestamp'].max().isoformat() if len(df) > 0 else None,
                "file_size_mb": filepath.stat().st_size / (1024 * 1024)
            }
        
        return summary


# Singleton
_collector: Optional[DataCollector] = None


def get_data_collector() -> DataCollector:
    """Get or create DataCollector singleton."""
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector


# CLI
async def main():
    """Ana çalıştırma fonksiyonu."""
    collector = get_data_collector()
    
    # Ana coinler
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    
    # 2 yıllık veri indir
    await collector.download_all_symbols(symbols, interval="1m", days=730)
    
    # Özet
    print("\n" + "="*50)
    print("📊 DATA SUMMARY")
    print("="*50)
    summary = collector.get_data_summary()
    for symbol, info in summary.items():
        print(f"{symbol}: {info['candles']:,} candles | {info['file_size_mb']:.1f} MB")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
