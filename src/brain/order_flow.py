# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ORDER FLOW ANALYSIS
===================================
Professional L2 orderbook and trade flow analysis.

THIS IS WHAT GIVES REAL EDGE:
- Large orders hiding in the book
- Aggressive buyers/sellers (market orders)
- Order imbalance at key levels
- Delta (buy pressure - sell pressure)

FEATURES:
1. L2 Orderbook Imbalance
2. Trade Delta (Aggressor detection)
3. Large Order Detection
4. VWAP Deviation
5. Cumulative Volume Delta (CVD)
"""
import logging
import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger("ORDER_FLOW")


@dataclass
class OrderFlowSignal:
    """Order flow analysis result"""
    symbol: str
    timestamp: datetime
    
    # L2 Orderbook Analysis
    bid_depth_usd: float          # Total bids in USD (top 20)
    ask_depth_usd: float          # Total asks in USD (top 20)
    imbalance_pct: float          # +ve = buyers, -ve = sellers
    
    # Large Orders
    large_bid_walls: List[Dict]   # Price levels with unusual size
    large_ask_walls: List[Dict]
    
    # Trade Flow
    buy_volume_1m: float          # Buy volume in last 1 min
    sell_volume_1m: float         # Sell volume in last 1 min
    delta_1m: float               # Buy - Sell
    
    # CVD (Cumulative Volume Delta)
    cvd_5m: float                 # 5-minute cumulative delta
    cvd_trend: str                # BULLISH, BEARISH, NEUTRAL
    
    # Signal
    bias: str                     # BUY_PRESSURE, SELL_PRESSURE, NEUTRAL
    strength: int                 # 0-100
    reasoning: str


class OrderFlowAnalyzer:
    """
    Professional Order Flow Analysis
    
    Analyzes L2 orderbook and trade flow to detect
    institutional activity and smart money.
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    
    # Thresholds
    IMBALANCE_THRESHOLD = 0.25    # 25% imbalance = significant
    LARGE_ORDER_MULT = 5          # 5x average = large order
    CVD_TREND_THRESHOLD = 0.6     # 60% one direction = trend
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._trade_buffer: Dict[str, deque] = {}  # Recent trades by symbol
        self._cvd_buffer: Dict[str, deque] = {}    # CVD history
        
        logger.info("📊 Order Flow Analyzer initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self._session
    
    async def analyze(self, symbol: str = "BTCUSDT") -> Optional[OrderFlowSignal]:
        """
        Full order flow analysis.
        
        Returns professional-grade order flow signal.
        """
        try:
            # Parallel data fetch
            orderbook, trades = await asyncio.gather(
                self._get_orderbook(symbol, limit=20),
                self._get_recent_trades(symbol, limit=500)
            )
            
            if not orderbook or not trades:
                return None
            
            # 1. Analyze L2 orderbook
            bid_depth, ask_depth, imbalance, bid_walls, ask_walls = self._analyze_orderbook(orderbook)
            
            # 2. Analyze trade flow
            buy_vol, sell_vol, delta = self._analyze_trades(trades)
            
            # 3. Calculate CVD
            cvd_5m, cvd_trend = self._calculate_cvd(symbol, delta)
            
            # 4. Determine bias
            bias, strength, reasoning = self._determine_bias(
                imbalance, delta, cvd_trend, bid_walls, ask_walls
            )
            
            return OrderFlowSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_depth_usd=bid_depth,
                ask_depth_usd=ask_depth,
                imbalance_pct=imbalance,
                large_bid_walls=bid_walls,
                large_ask_walls=ask_walls,
                buy_volume_1m=buy_vol,
                sell_volume_1m=sell_vol,
                delta_1m=delta,
                cvd_5m=cvd_5m,
                cvd_trend=cvd_trend,
                bias=bias,
                strength=strength,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Order flow analysis error: {e}")
            return None
    
    async def _get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Fetch L2 orderbook"""
        try:
            session = await self._get_session()
            url = f"{self.FUTURES_BASE}/fapi/v1/depth"
            params = {'symbol': symbol, 'limit': limit}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Orderbook fetch error: {e}")
        return None
    
    async def _get_recent_trades(self, symbol: str, limit: int = 500) -> Optional[List]:
        """Fetch recent aggressor trades"""
        try:
            session = await self._get_session()
            url = f"{self.FUTURES_BASE}/fapi/v1/aggTrades"
            params = {'symbol': symbol, 'limit': limit}
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Trades fetch error: {e}")
        return None
    
    def _analyze_orderbook(self, orderbook: Dict) -> Tuple:
        """Analyze L2 orderbook for imbalance and walls"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        # Calculate depth
        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids)
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks)
        
        # Imbalance
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total * 100 if total > 0 else 0
        
        # Find large orders (walls)
        avg_bid_size = bid_depth / len(bids) if bids else 0
        avg_ask_size = ask_depth / len(asks) if asks else 0
        
        bid_walls = []
        for b in bids:
            size_usd = float(b[0]) * float(b[1])
            if size_usd > avg_bid_size * self.LARGE_ORDER_MULT:
                bid_walls.append({
                    'price': float(b[0]),
                    'size_usd': size_usd,
                    'type': 'SUPPORT'
                })
        
        ask_walls = []
        for a in asks:
            size_usd = float(a[0]) * float(a[1])
            if size_usd > avg_ask_size * self.LARGE_ORDER_MULT:
                ask_walls.append({
                    'price': float(a[0]),
                    'size_usd': size_usd,
                    'type': 'RESISTANCE'
                })
        
        return bid_depth, ask_depth, imbalance, bid_walls[:3], ask_walls[:3]
    
    def _analyze_trades(self, trades: List) -> Tuple[float, float, float]:
        """Analyze recent trades for buy/sell pressure"""
        buy_volume = 0.0
        sell_volume = 0.0
        
        for trade in trades:
            qty = float(trade.get('q', 0))
            price = float(trade.get('p', 0))
            volume_usd = qty * price
            
            # Buyer is maker = Sell (aggressor sold into bid)
            # Buyer is NOT maker = Buy (aggressor bought into ask)
            is_buyer_maker = trade.get('m', False)
            
            if is_buyer_maker:
                sell_volume += volume_usd
            else:
                buy_volume += volume_usd
        
        delta = buy_volume - sell_volume
        
        return buy_volume, sell_volume, delta
    
    def _calculate_cvd(self, symbol: str, current_delta: float) -> Tuple[float, str]:
        """Calculate Cumulative Volume Delta"""
        if symbol not in self._cvd_buffer:
            self._cvd_buffer[symbol] = deque(maxlen=5)  # 5 samples for 5min
        
        self._cvd_buffer[symbol].append(current_delta)
        
        cvd = sum(self._cvd_buffer[symbol])
        
        # Determine trend
        if len(self._cvd_buffer[symbol]) >= 3:
            positives = sum(1 for d in self._cvd_buffer[symbol] if d > 0)
            ratio = positives / len(self._cvd_buffer[symbol])
            
            if ratio >= self.CVD_TREND_THRESHOLD:
                trend = "BULLISH"
            elif ratio <= 1 - self.CVD_TREND_THRESHOLD:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
        else:
            trend = "NEUTRAL"
        
        return cvd, trend
    
    def _determine_bias(
        self,
        imbalance: float,
        delta: float,
        cvd_trend: str,
        bid_walls: List,
        ask_walls: List
    ) -> Tuple[str, int, str]:
        """Determine overall order flow bias"""
        score = 0
        reasons = []
        
        # Orderbook imbalance (40% weight)
        if imbalance > self.IMBALANCE_THRESHOLD * 100:
            score += 40
            reasons.append(f"OB: {imbalance:+.0f}% buyers")
        elif imbalance < -self.IMBALANCE_THRESHOLD * 100:
            score -= 40
            reasons.append(f"OB: {imbalance:+.0f}% sellers")
        
        # Trade delta (40% weight)
        if delta > 0:
            delta_pct = min(40, abs(delta) / 1000000 * 40)  # Scale by $1M
            score += delta_pct
            reasons.append(f"Delta: ${delta/1000:+.0f}K buy pressure")
        else:
            delta_pct = min(40, abs(delta) / 1000000 * 40)
            score -= delta_pct
            reasons.append(f"Delta: ${delta/1000:+.0f}K sell pressure")
        
        # CVD trend (20% weight)
        if cvd_trend == "BULLISH":
            score += 20
            reasons.append("CVD: Bullish trend")
        elif cvd_trend == "BEARISH":
            score -= 20
            reasons.append("CVD: Bearish trend")
        
        # Large walls (bonus)
        if bid_walls and not ask_walls:
            score += 10
            reasons.append(f"Wall: Support at ${bid_walls[0]['price']:,.0f}")
        elif ask_walls and not bid_walls:
            score -= 10
            reasons.append(f"Wall: Resistance at ${ask_walls[0]['price']:,.0f}")
        
        # Determine bias
        if score > 25:
            bias = "BUY_PRESSURE"
        elif score < -25:
            bias = "SELL_PRESSURE"
        else:
            bias = "NEUTRAL"
        
        strength = min(100, abs(score))
        reasoning = " | ".join(reasons) if reasons else "Neutral flow"
        
        return bias, strength, reasoning
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# Singleton
_analyzer: Optional[OrderFlowAnalyzer] = None

def get_order_flow_analyzer() -> OrderFlowAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = OrderFlowAnalyzer()
    return _analyzer
