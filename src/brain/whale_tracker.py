# -*- coding: utf-8 -*-
"""
DEMIR AI - Real-Time Whale Tracker (Zero-Mock)
==============================================
Uses Binance WebSockets to track "Exchange Whales" in real-time.
Replaces slow On-Chain data with millisecond-level Order Book & Trade analysis.

Capabilities:
- Order Book Imbalance (Depth20)
- Large Trade Detection (AggTrade > $100k)
- Real-time Buying/Selling Pressure
"""
import logging
import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable
from collections import deque

logger = logging.getLogger("WHALE_TRACKER")

class WhaleTracker:
    """
    Real-Time Whale Tracker via Binance WebSockets.
    Monitors BTC/USDT by default.
    """
    
    WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth20@100ms/btcusdt@aggTrade"
    
    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth20@100ms/{self.symbol}@aggTrade"
        self.running = False
        self.ws_task = None
        
        # Real-time State
        self.order_book_imbalance = 0.0 # > 1 means Bullish (More Bids), < 1 means Bearish (More Asks)
        self.large_trades = deque(maxlen=50) # Keep last 50 big trades
        self.net_volume_flow = 0.0 # Net Buy - Sell Volume (USD)
        
        # Thresholds
        self.WHALE_THRESHOLD_USD = 100_000 # Track trades > $100k
        
    async def start(self):
        """Start the WebSocket Monitor."""
        if self.running: return
        self.running = True
        self.ws_task = asyncio.create_task(self._listen())
        logger.info(f"🐋 Whale Tracker Started for {self.symbol}")

    async def stop(self):
        """Stop the Monitor."""
        self.running = False
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        logger.info("❌ Whale Tracker Stopped")

    async def _listen(self):
        """Main WebSocket Loop."""
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("🔌 Connected to Binance Whale Stream")
                    
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        event_type = data.get('e')
                        
                        if event_type == 'depthUpdate':
                            self._process_depth(data)
                        elif event_type == 'aggTrade':
                            self._process_trade(data)
                            
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")
                await asyncio.sleep(5) # Reconnect delay

    def _process_depth(self, data: Dict):
        """
        Process Order Book Depth.
        Calculates Imbalance Ratio: (Bid Vol / Ask Vol)
        """
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        if not bids or not asks: return
        
        # Sum top 20 levels (Price * Quantity)
        total_bid_usd = sum(float(b[0]) * float(b[1]) for b in bids)
        total_ask_usd = sum(float(a[0]) * float(a[1]) for a in asks)
        
        # Avoid division by zero
        if total_ask_usd > 0:
            self.order_book_imbalance = total_bid_usd / total_ask_usd
        else:
            self.order_book_imbalance = 1.0

    def _process_trade(self, data: Dict):
        """
        Process Trade Stream.
        Detects Whale Moves.
        """
        # q = Quantity, p = Price, m = isBuyerMaker (True=Sell, False=Buy)
        qty = float(data.get('q', 0))
        price = float(data.get('p', 0))
        is_sell = data.get('m', False) # In derivatives, isBuyerMaker=True means Sell (Maker was Buyer)
        
        usd_value = qty * price
        
        if usd_value >= self.WHALE_THRESHOLD_USD:
            direction = "SELL" if is_sell else "BUY"
            
            # Update Net Flow
            if is_sell:
                self.net_volume_flow -= usd_value
            else:
                self.net_volume_flow += usd_value
                
            # Log Whale Trade
            trade_info = {
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': price,
                'value': usd_value,
                'direction': direction
            }
            self.large_trades.appendleft(trade_info)
            # logger.info(f"🐋 WHALE {direction}: ${usd_value/1000:.1f}k @ {price}")

    def get_whale_summary(self) -> Dict:
        """
        Returns snapshot of Whale Activity.
        NO FAKE DATA. Returns 'Unavailable' if not running.
        """
        if not self.running and not self.large_trades:
             return {'available': False, 'reason': 'Tracker Not Running'}
             
        # Determine Status
        status = "NEUTRAL"
        confidence = 0
        
        # Logic: If Order Book supports it AND Net Flow is positive -> BULLISH
        if self.order_book_imbalance > 1.5 and self.net_volume_flow > 1_000_000:
            status = "BULLISH"
            confidence = 80
        elif self.order_book_imbalance < 0.6 and self.net_volume_flow < -1_000_000:
            status = "BEARISH"
            confidence = 80
            
        return {
            'available': True,
            'imbalance_ratio': round(self.order_book_imbalance, 2),
            'net_flow_usd': self.net_volume_flow,
            'whale_trade_count': len(self.large_trades),
            'last_whale_trades': list(self.large_trades)[:5],
            'status': status,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }

# Global Singleton
_tracker = None

def get_whale_tracker() -> WhaleTracker:
    global _tracker
    if _tracker is None:
        _tracker = WhaleTracker()
    return _tracker
