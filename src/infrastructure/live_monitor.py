import asyncio
import logging
import json
from typing import Callable
from binance import AsyncClient, BinanceSocketManager
from src.config import Config

logger = logging.getLogger("LIVE_MONITOR")

class LiveMarketMonitor:
    """
    7/24 WebSocket stream - ANINDA fiyat değişikliklerini yakalar
    """
    def __init__(self, on_signal_trigger: Callable):
        self.client = None
        self.bsm = None
        self.on_signal_trigger = on_signal_trigger  # AI'yı tetikleyen callback
        
        # Monitoring state
        self.last_prices = {}
        self.candle_data = {}
        self.last_ai_check = {}
        
        # Trigger thresholds
        self.PRICE_CHANGE_THRESHOLD = 0.005  # 0.5% hareket = AI trigger
        self.VOLUME_SPIKE_MULTIPLIER = 2.0    # 2x normal volume = AI trigger
        self.MIN_TIME_BETWEEN_AI = 300        # AI'yı en fazla 5 dakikada 1 çağır
        
    async def start(self):
        """Start WebSocket streams for BTC and ETH"""
        logger.info("🔴 LIVE MODE: Starting WebSocket streams...")
        
        self.client = await AsyncClient.create(
            Config.BINANCE_API_KEY, 
            Config.BINANCE_API_SECRET
        )
        self.bsm = BinanceSocketManager(self.client)
        
        # Start streams for both symbols
        tasks = [
            self._monitor_symbol("BTCUSDT"),
            self._monitor_symbol("ETHUSDT")
        ]
        
        await asyncio.gather(*tasks)
        
    async def _monitor_symbol(self, symbol: str):
        """Monitor single symbol via WebSocket"""
        logger.info(f"📡 Monitoring {symbol} live...")
        
        # Subscribe to kline (candlestick) stream - 1 minute candles
        async with self.bsm.kline_socket(symbol, interval='1m') as stream:
            while True:
                try:
                    msg = await stream.recv()
                    await self._process_kline(symbol, msg)
                except Exception as e:
                    logger.error(f"WebSocket error {symbol}: {e}")
                    await asyncio.sleep(5)
                    
    async def _process_kline(self, symbol: str, data: dict):
        """Process incoming kline data"""
        kline = data['k']
        is_closed = kline['x']  # Candle closed?
        
        close_price = float(kline['c'])
        volume = float(kline['v'])
        
        # Initialize tracking
        if symbol not in self.last_prices:
            self.last_prices[symbol] = close_price
            self.candle_data[symbol] = {'volumes': [volume]}
            self.last_ai_check[symbol] = 0
            return
        
        # Calculate changes
        price_change_pct = abs(close_price - self.last_prices[symbol]) / self.last_prices[symbol]
        
        # Average volume (last 20 candles)
        volumes = self.candle_data[symbol].get('volumes', [volume])
        avg_volume = sum(volumes[-20:]) / min(len(volumes), 20)
        volume_spike = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Check if we should trigger AI
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_ai_check[symbol]
        
        should_trigger = False
        reason = ""
        
        # Trigger conditions
        if price_change_pct >= self.PRICE_CHANGE_THRESHOLD:
            should_trigger = True
            reason = f"Price move: {price_change_pct*100:.2f}%"
            
        elif volume_spike >= self.VOLUME_SPIKE_MULTIPLIER:
            should_trigger = True
            reason = f"Volume spike: {volume_spike:.1f}x normal"
            
        elif time_since_last >= 3600:  # Fallback: at least every hour
            should_trigger = True
            reason = "Scheduled check (1h)"
        
        # Only trigger if enough time passed
        if should_trigger and time_since_last >= self.MIN_TIME_BETWEEN_AI:
            logger.info(f"⚡ TRIGGER for {symbol}: {reason}")
            logger.info(f"   Price: ${close_price:,.2f} | Volume: {volume:,.0f}")
            
            # Update state
            self.last_ai_check[symbol] = current_time
            self.last_prices[symbol] = close_price
            
            # Call AI analysis (async callback)
            asyncio.create_task(self.on_signal_trigger(symbol, reason))
        
        # Update data
        if is_closed:
            self.candle_data[symbol]['volumes'].append(volume)
            if len(self.candle_data[symbol]['volumes']) > 100:
                self.candle_data[symbol]['volumes'].pop(0)
                
    async def stop(self):
        """Stop WebSocket connections"""
        if self.client:
            await self.client.close_connection()
