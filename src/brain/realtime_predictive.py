# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - REAL-TIME PREDICTIVE ENGINE
===========================================
Professional-grade early detection system.

HOW PROFESSIONAL FIRMS DETECT MOVES EARLY:
1. Order Flow Imbalance - See aggressive buyers/sellers BEFORE price moves
2. Large Order Detection - Spot institutional orders
3. Volume Spike Detection - Unusual activity = something about to happen
4. Liquidation Cascade Detection - Predict cascade moves
5. CVD Divergence - Volume delta leads price

THIS MODULE:
- WebSocket connection to Binance for real-time data
- Sub-second order flow analysis  
- Instant alerts when institutional activity detected
- Predictive signals BEFORE the move

Author: DEMIR AI Team
Date: 2026-01-02
"""
import logging
import asyncio
import json
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from collections import deque
import statistics

logger = logging.getLogger("PREDICTIVE_ENGINE")


@dataclass
class PredictiveAlert:
    """Early warning alert"""
    timestamp: datetime
    symbol: str
    alert_type: str          # VOLUME_SPIKE, ORDER_IMBALANCE, LARGE_ORDER, LIQ_CASCADE, CVD_DIVERGENCE
    direction: str           # BULLISH, BEARISH
    strength: int            # 0-100
    message: str
    price_at_detection: float
    expected_move_pct: float  # Predicted move size
    confidence: int          # 0-100
    urgency: str             # IMMEDIATE, SOON, WATCH


@dataclass
class OrderFlowState:
    """Real-time order flow state"""
    symbol: str
    
    # Trade flow (last N seconds)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta: float = 0.0           # buy - sell
    
    # Order imbalance
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    imbalance: float = 0.0       # +ve = buyers, -ve = sellers
    
    # Large orders detected
    large_buys: int = 0
    large_sells: int = 0
    
    # Volume spike
    current_volume_1m: float = 0.0
    avg_volume_1m: float = 0.0
    volume_ratio: float = 1.0
    
    # CVD (Cumulative Volume Delta)
    cvd: float = 0.0
    cvd_trend: str = "NEUTRAL"
    
    # Price
    current_price: float = 0.0
    price_1m_ago: float = 0.0
    
    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)


class RealTimePredictiveEngine:
    """
    Real-Time Predictive Trading Engine
    
    Uses WebSocket streams for sub-second analysis:
    - Aggressor trade flow
    - Order book depth changes
    - Large order detection
    - Volume anomalies
    """
    
    WS_BASE = "wss://fstream.binance.com/ws"
    
    # Detection thresholds
    VOLUME_SPIKE_THRESHOLD = 3.0      # 3x normal volume
    IMBALANCE_THRESHOLD = 0.40        # 40% imbalance
    LARGE_ORDER_USD = 500_000         # $500k+ = large order
    CVD_DIVERGENCE_THRESHOLD = 0.7    # 70% one direction
    
    # History for averaging
    HISTORY_SECONDS = 300  # 5 minutes
    
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.states: Dict[str, OrderFlowState] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Historical data for averaging
        self._trade_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._cvd_history: Dict[str, deque] = {}
        
        # WebSocket
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._running = False
        
        # Initialize states
        for sym in self.symbols:
            self.states[sym] = OrderFlowState(symbol=sym)
            self._trade_history[sym] = deque(maxlen=1000)
            self._volume_history[sym] = deque(maxlen=60)  # 1 min of 1s samples
            self._cvd_history[sym] = deque(maxlen=300)    # 5 min
        
        logger.info(f"🎯 Real-Time Predictive Engine initialized for {self.symbols}")
    
    def add_alert_callback(self, callback: Callable[[PredictiveAlert], None]):
        """Add callback for when alerts are generated"""
        self.alert_callbacks.append(callback)
    
    async def start(self):
        """Start real-time monitoring"""
        self._running = True
        logger.info("🚀 Starting real-time predictive monitoring...")
        
        # Start WebSocket streams
        await asyncio.gather(
            self._run_trade_stream(),
            self._run_depth_stream(),
            self._run_analysis_loop()
        )
    
    async def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._ws_session:
            await self._ws_session.close()
    
    async def _run_trade_stream(self):
        """Connect to aggTrade stream for real-time trades"""
        streams = "/".join([f"{s.lower()}@aggTrade" for s in self.symbols])
        url = f"{self.WS_BASE}/{streams}"
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info("📡 Trade stream connected")
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._process_trade(json.loads(msg.data))
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                break
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break
                                
            except Exception as e:
                logger.error(f"Trade stream error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
    
    async def _run_depth_stream(self):
        """Connect to depth stream for orderbook updates"""
        streams = "/".join([f"{s.lower()}@depth5@100ms" for s in self.symbols])
        url = f"{self.WS_BASE}/{streams}"
        
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url) as ws:
                        logger.info("📊 Depth stream connected")
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._process_depth(json.loads(msg.data))
                            elif msg.type in [aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR]:
                                break
                                
            except Exception as e:
                logger.error(f"Depth stream error: {e}")
                await asyncio.sleep(5)
    
    async def _run_analysis_loop(self):
        """Analyze state every second for predictive alerts"""
        while self._running:
            try:
                for symbol in self.symbols:
                    alerts = self._analyze_state(symbol)
                    for alert in alerts:
                        await self._emit_alert(alert)
                
                await asyncio.sleep(1)  # 1 second analysis interval
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_trade(self, data: dict):
        """Process real-time trade"""
        try:
            symbol = data.get('s', '')
            if symbol not in self.states:
                return
            
            state = self.states[symbol]
            
            price = float(data.get('p', 0))
            qty = float(data.get('q', 0))
            volume_usd = price * qty
            is_buyer_maker = data.get('m', False)  # True = sell aggressor
            
            # Update price
            state.current_price = price
            
            # Track trade
            trade = {
                'time': datetime.now(),
                'price': price,
                'volume': volume_usd,
                'is_buy': not is_buyer_maker
            }
            self._trade_history[symbol].append(trade)
            
            # Update flow metrics
            if is_buyer_maker:
                state.sell_volume += volume_usd
            else:
                state.buy_volume += volume_usd
            
            state.delta = state.buy_volume - state.sell_volume
            
            # CVD update
            cvd_change = volume_usd if not is_buyer_maker else -volume_usd
            state.cvd += cvd_change
            self._cvd_history[symbol].append(cvd_change)
            
            # Large order detection
            if volume_usd >= self.LARGE_ORDER_USD:
                if is_buyer_maker:
                    state.large_sells += 1
                    await self._emit_alert(PredictiveAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="LARGE_ORDER",
                        direction="BEARISH",
                        strength=min(100, int(volume_usd / self.LARGE_ORDER_USD * 50)),
                        message=f"🐋 Large SELL: ${volume_usd/1000:.0f}K @ ${price:,.0f}",
                        price_at_detection=price,
                        expected_move_pct=-0.5,
                        confidence=70,
                        urgency="IMMEDIATE"
                    ))
                else:
                    state.large_buys += 1
                    await self._emit_alert(PredictiveAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="LARGE_ORDER",
                        direction="BULLISH",
                        strength=min(100, int(volume_usd / self.LARGE_ORDER_USD * 50)),
                        message=f"🐋 Large BUY: ${volume_usd/1000:.0f}K @ ${price:,.0f}",
                        price_at_detection=price,
                        expected_move_pct=0.5,
                        confidence=70,
                        urgency="IMMEDIATE"
                    ))
            
            state.last_update = datetime.now()
            
        except Exception as e:
            logger.debug(f"Trade process error: {e}")
    
    async def _process_depth(self, data: dict):
        """Process orderbook depth update"""
        try:
            symbol = data.get('s', '')
            if symbol not in self.states:
                return
            
            state = self.states[symbol]
            
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            # Calculate depth
            bid_depth = sum(float(b[0]) * float(b[1]) for b in bids)
            ask_depth = sum(float(a[0]) * float(a[1]) for a in asks)
            
            state.bid_depth = bid_depth
            state.ask_depth = ask_depth
            
            # Imbalance
            total = bid_depth + ask_depth
            if total > 0:
                state.imbalance = (bid_depth - ask_depth) / total
            
        except Exception as e:
            logger.debug(f"Depth process error: {e}")
    
    def _analyze_state(self, symbol: str) -> List[PredictiveAlert]:
        """Analyze current state for predictive alerts"""
        alerts = []
        state = self.states.get(symbol)
        if not state or state.current_price == 0:
            return alerts
        
        # === 1. VOLUME SPIKE DETECTION ===
        recent_trades = [t for t in self._trade_history[symbol] 
                        if (datetime.now() - t['time']).seconds < 60]
        current_vol = sum(t['volume'] for t in recent_trades)
        
        # Get average from history
        if len(self._volume_history[symbol]) > 10:
            avg_vol = statistics.mean(self._volume_history[symbol])
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                state.volume_ratio = vol_ratio
                
                if vol_ratio >= self.VOLUME_SPIKE_THRESHOLD:
                    # Determine direction from delta
                    buy_vol = sum(t['volume'] for t in recent_trades if t['is_buy'])
                    sell_vol = current_vol - buy_vol
                    direction = "BULLISH" if buy_vol > sell_vol else "BEARISH"
                    
                    alerts.append(PredictiveAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="VOLUME_SPIKE",
                        direction=direction,
                        strength=min(100, int(vol_ratio * 20)),
                        message=f"📈 Volume {vol_ratio:.1f}x normal - {direction} bias",
                        price_at_detection=state.current_price,
                        expected_move_pct=1.0 if direction == "BULLISH" else -1.0,
                        confidence=75,
                        urgency="IMMEDIATE"
                    ))
        
        self._volume_history[symbol].append(current_vol)
        
        # === 2. ORDER IMBALANCE DETECTION ===
        if abs(state.imbalance) >= self.IMBALANCE_THRESHOLD:
            direction = "BULLISH" if state.imbalance > 0 else "BEARISH"
            strength = int(abs(state.imbalance) * 100)
            
            alerts.append(PredictiveAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type="ORDER_IMBALANCE",
                direction=direction,
                strength=strength,
                message=f"⚖️ {abs(state.imbalance)*100:.0f}% orderbook {direction.lower()} imbalance",
                price_at_detection=state.current_price,
                expected_move_pct=0.5 if direction == "BULLISH" else -0.5,
                confidence=65,
                urgency="SOON"
            ))
        
        # === 3. CVD DIVERGENCE (Price vs Volume) ===
        if len(self._cvd_history[symbol]) >= 60:  # 1 min of data
            recent_cvd = list(self._cvd_history[symbol])[-60:]
            positive_count = sum(1 for c in recent_cvd if c > 0)
            cvd_ratio = positive_count / len(recent_cvd)
            
            if cvd_ratio >= self.CVD_DIVERGENCE_THRESHOLD:
                state.cvd_trend = "BULLISH"
            elif cvd_ratio <= 1 - self.CVD_DIVERGENCE_THRESHOLD:
                state.cvd_trend = "BEARISH"
            else:
                state.cvd_trend = "NEUTRAL"
            
            # Divergence: CVD trending but price not moving
            if state.price_1m_ago > 0:
                price_change = (state.current_price - state.price_1m_ago) / state.price_1m_ago
                
                # CVD bullish but price flat/down = BULLISH DIVERGENCE (buy signal)
                if state.cvd_trend == "BULLISH" and price_change < 0.001:
                    alerts.append(PredictiveAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="CVD_DIVERGENCE",
                        direction="BULLISH",
                        strength=70,
                        message="🔄 CVD Bullish Divergence - Buyers accumulating",
                        price_at_detection=state.current_price,
                        expected_move_pct=1.5,
                        confidence=80,
                        urgency="SOON"
                    ))
                
                # CVD bearish but price flat/up = BEARISH DIVERGENCE (sell signal)
                elif state.cvd_trend == "BEARISH" and price_change > -0.001:
                    alerts.append(PredictiveAlert(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        alert_type="CVD_DIVERGENCE",
                        direction="BEARISH",
                        strength=70,
                        message="🔄 CVD Bearish Divergence - Sellers distributing",
                        price_at_detection=state.current_price,
                        expected_move_pct=-1.5,
                        confidence=80,
                        urgency="SOON"
                    ))
        
        # Update price history
        if len(self._trade_history[symbol]) > 0:
            old_trades = [t for t in self._trade_history[symbol]
                         if 55 < (datetime.now() - t['time']).seconds < 65]
            if old_trades:
                state.price_1m_ago = old_trades[0]['price']
        
        return alerts
    
    async def _emit_alert(self, alert: PredictiveAlert):
        """Emit alert to all callbacks"""
        logger.info(f"🚨 ALERT: {alert.symbol} {alert.alert_type} - {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_state(self, symbol: str) -> Optional[OrderFlowState]:
        """Get current state for symbol"""
        return self.states.get(symbol)
    
    def get_summary(self) -> Dict:
        """Get summary of all states"""
        return {
            sym: {
                'price': state.current_price,
                'delta': state.delta,
                'imbalance': state.imbalance,
                'volume_ratio': state.volume_ratio,
                'cvd_trend': state.cvd_trend,
                'large_buys': state.large_buys,
                'large_sells': state.large_sells
            }
            for sym, state in self.states.items()
        }


# Singleton
_engine: Optional[RealTimePredictiveEngine] = None

def get_realtime_engine(symbols: List[str] = None) -> RealTimePredictiveEngine:
    global _engine
    if _engine is None:
        _engine = RealTimePredictiveEngine(symbols)
    return _engine
