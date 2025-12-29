# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - MOMENTUM DETECTOR
================================
Ani fiyat hareketlerini ÖNCEDEN algılayan modül.

Algılanan Sinyaller:
1. Volume Spike - Ani hacim artışı
2. Price Momentum - 5m/15m momentum
3. Liquidation Cluster - Büyük liq seviyeleri
4. OI Surge - Open Interest ani artışı
5. CVD Divergence - Gerçek alım/satım baskısı
"""
import logging
import aiohttp
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("MOMENTUM_DETECTOR")


@dataclass
class MomentumAlert:
    """Ani hareket uyarısı"""
    symbol: str
    alert_type: str  # VOLUME_SPIKE, MOMENTUM_SURGE, LIQ_CLUSTER, OI_SURGE, CVD_DIVERGENCE
    severity: str    # LOW, MEDIUM, HIGH, CRITICAL
    direction: str   # BULLISH, BEARISH, NEUTRAL
    value: float
    threshold: float
    message: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'alert_type': self.alert_type,
            'severity': self.severity,
            'direction': self.direction,
            'value': self.value,
            'threshold': self.threshold,
            'message': self.message,
            'timestamp': self.timestamp
        }


@dataclass 
class MomentumContext:
    """Tüm momentum verileri"""
    symbol: str
    
    # Volume Analysis
    volume_spike: bool = False
    volume_ratio: float = 1.0  # Current vol / Avg vol
    
    # Price Momentum
    momentum_5m: float = 0.0   # % change
    momentum_15m: float = 0.0  # % change
    momentum_1h: float = 0.0   # % change
    momentum_direction: str = "NEUTRAL"  # STRONG_BULL, BULL, NEUTRAL, BEAR, STRONG_BEAR
    
    # Liquidation Analysis
    nearest_liq_long: float = 0.0   # En yakın long liq
    nearest_liq_short: float = 0.0  # En yakın short liq
    liq_magnet: str = "NONE"        # LONG_LIQ, SHORT_LIQ, NONE
    liq_distance_pct: float = 0.0   # Magnet'e mesafe %
    
    # Open Interest
    oi_change_5m: float = 0.0
    oi_change_1h: float = 0.0
    oi_surge: bool = False
    
    # CVD (Cumulative Volume Delta)
    cvd_5m: float = 0.0  # + = net buying, - = net selling
    cvd_15m: float = 0.0
    cvd_divergence: bool = False  # Price up but CVD down = bearish divergence
    
    # Alerts
    alerts: List[MomentumAlert] = None
    
    # Overall
    breakout_probability: float = 0.0  # 0-100
    expected_direction: str = "UNKNOWN"
    
    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'volume_spike': self.volume_spike,
            'volume_ratio': round(self.volume_ratio, 2),
            'momentum_5m': round(self.momentum_5m, 2),
            'momentum_15m': round(self.momentum_15m, 2),
            'momentum_1h': round(self.momentum_1h, 2),
            'momentum_direction': self.momentum_direction,
            'nearest_liq_long': self.nearest_liq_long,
            'nearest_liq_short': self.nearest_liq_short,
            'liq_magnet': self.liq_magnet,
            'oi_change_5m': round(self.oi_change_5m, 2),
            'oi_change_1h': round(self.oi_change_1h, 2),
            'oi_surge': self.oi_surge,
            'cvd_5m': round(self.cvd_5m, 2),
            'cvd_divergence': self.cvd_divergence,
            'breakout_probability': round(self.breakout_probability, 1),
            'expected_direction': self.expected_direction,
            'alert_count': len(self.alerts)
        }
    
    def get_summary(self) -> str:
        """Telegram için özet"""
        lines = []
        
        if self.volume_spike:
            lines.append(f"🔥 Volume Spike! ({self.volume_ratio:.1f}x)")
        
        if abs(self.momentum_5m) > 0.5:
            emoji = "📈" if self.momentum_5m > 0 else "📉"
            lines.append(f"{emoji} 5m Momentum: {self.momentum_5m:+.2f}%")
        
        if self.oi_surge:
            lines.append(f"📊 OI Surge: {self.oi_change_5m:+.1f}%")
        
        if self.cvd_divergence:
            lines.append("⚠️ CVD Divergence!")
        
        if self.liq_magnet != "NONE":
            lines.append(f"🧲 Liq Magnet: {self.liq_magnet} ({self.liq_distance_pct:.1f}% away)")
        
        if self.breakout_probability > 60:
            lines.append(f"🚀 Breakout Prob: {self.breakout_probability:.0f}%")
        
        return "\n".join(lines) if lines else "Normal conditions"


class MomentumDetector:
    """Ani hareket algılayıcı"""
    
    BINANCE_FUTURES = "https://fapi.binance.com"
    
    # Thresholds
    VOLUME_SPIKE_THRESHOLD = 2.5  # 2.5x average = spike
    MOMENTUM_THRESHOLD = 0.5      # 0.5% in 5m = significant
    OI_SURGE_THRESHOLD = 3.0      # 3% OI change in 5m = surge
    LIQ_DISTANCE_ALERT = 2.0      # 2% distance to liq cluster = alert
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, MomentumContext] = {}
        self._cache_time: Dict[str, datetime] = {}
        
        logger.info("⚡ Momentum Detector initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self.HEADERS)
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def analyze(self, symbol: str) -> MomentumContext:
        """
        Tam momentum analizi yap.
        
        Returns:
            MomentumContext: Tüm momentum verileri
        """
        context = MomentumContext(symbol=symbol)
        
        try:
            session = await self._get_session()
            
            # Parallel data fetching
            import asyncio
            
            klines_5m, klines_15m, klines_1h, ticker, oi_data = await asyncio.gather(
                self._get_klines(session, symbol, "5m", 20),
                self._get_klines(session, symbol, "15m", 10),
                self._get_klines(session, symbol, "1h", 5),
                self._get_ticker(session, symbol),
                self._get_open_interest(session, symbol),
                return_exceptions=True
            )
            
            current_price = 0
            if ticker and not isinstance(ticker, Exception):
                current_price = float(ticker.get('lastPrice', 0))
            
            # 1. VOLUME SPIKE DETECTION
            if klines_5m and not isinstance(klines_5m, Exception):
                context = self._analyze_volume(context, klines_5m)
            
            # 2. MOMENTUM CALCULATION
            if klines_5m and not isinstance(klines_5m, Exception):
                context = self._analyze_momentum(context, klines_5m, klines_15m, klines_1h)
            
            # 3. CVD (Cumulative Volume Delta)
            if klines_5m and not isinstance(klines_5m, Exception):
                context = self._analyze_cvd(context, klines_5m, klines_15m)
            
            # 4. OPEN INTEREST SURGE
            if oi_data and not isinstance(oi_data, Exception):
                context = self._analyze_oi(context, oi_data)
            
            # 5. LIQUIDATION CLUSTERS (from existing liq hunter data)
            context = await self._analyze_liquidations(context, current_price)
            
            # 6. CALCULATE BREAKOUT PROBABILITY
            context = self._calculate_breakout_probability(context)
            
            # 7. GENERATE ALERTS
            context = self._generate_alerts(context)
            
            # Cache
            self._cache[symbol] = context
            self._cache_time[symbol] = datetime.now()
            
            if context.alerts:
                logger.info(f"⚡ {symbol}: {len(context.alerts)} momentum alerts!")
            
        except Exception as e:
            logger.error(f"Momentum analysis error for {symbol}: {e}")
        
        return context
    
    async def _get_klines(self, session: aiohttp.ClientSession, 
                          symbol: str, interval: str, limit: int) -> List:
        """Kline verisi al"""
        try:
            url = f"{self.BINANCE_FUTURES}/fapi/v1/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Klines error: {e}")
        return []
    
    async def _get_ticker(self, session: aiohttp.ClientSession, symbol: str) -> Dict:
        """Ticker verisi al"""
        try:
            url = f"{self.BINANCE_FUTURES}/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"Ticker error: {e}")
        return {}
    
    async def _get_open_interest(self, session: aiohttp.ClientSession, symbol: str) -> List:
        """Open Interest history al"""
        try:
            url = f"{self.BINANCE_FUTURES}/futures/data/openInterestHist"
            params = {"symbol": symbol, "period": "5m", "limit": 12}  # Son 1 saat
            
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.debug(f"OI error: {e}")
        return []
    
    def _analyze_volume(self, context: MomentumContext, klines: List) -> MomentumContext:
        """Volume spike analizi"""
        try:
            if len(klines) < 10:
                return context
            
            volumes = [float(k[5]) for k in klines]  # Volume at index 5
            
            current_vol = volumes[-1]
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1])  # Exclude current
            
            if avg_vol > 0:
                context.volume_ratio = current_vol / avg_vol
                context.volume_spike = context.volume_ratio >= self.VOLUME_SPIKE_THRESHOLD
                
        except Exception as e:
            logger.debug(f"Volume analysis error: {e}")
        
        return context
    
    def _analyze_momentum(self, context: MomentumContext, 
                          klines_5m: List, klines_15m: List, klines_1h: List) -> MomentumContext:
        """Momentum hesapla"""
        try:
            # 5m momentum
            if len(klines_5m) >= 2:
                prev_close = float(klines_5m[-2][4])
                curr_close = float(klines_5m[-1][4])
                if prev_close > 0:
                    context.momentum_5m = ((curr_close - prev_close) / prev_close) * 100
            
            # 15m momentum
            if klines_15m and len(klines_15m) >= 2:
                prev_close = float(klines_15m[-2][4])
                curr_close = float(klines_15m[-1][4])
                if prev_close > 0:
                    context.momentum_15m = ((curr_close - prev_close) / prev_close) * 100
            
            # 1h momentum
            if klines_1h and len(klines_1h) >= 2:
                prev_close = float(klines_1h[-2][4])
                curr_close = float(klines_1h[-1][4])
                if prev_close > 0:
                    context.momentum_1h = ((curr_close - prev_close) / prev_close) * 100
            
            # Direction
            avg_momentum = (context.momentum_5m * 3 + context.momentum_15m * 2 + context.momentum_1h) / 6
            
            if avg_momentum > 1.0:
                context.momentum_direction = "STRONG_BULL"
            elif avg_momentum > 0.3:
                context.momentum_direction = "BULL"
            elif avg_momentum < -1.0:
                context.momentum_direction = "STRONG_BEAR"
            elif avg_momentum < -0.3:
                context.momentum_direction = "BEAR"
            else:
                context.momentum_direction = "NEUTRAL"
                
        except Exception as e:
            logger.debug(f"Momentum analysis error: {e}")
        
        return context
    
    def _analyze_cvd(self, context: MomentumContext, 
                     klines_5m: List, klines_15m: List) -> MomentumContext:
        """CVD (Cumulative Volume Delta) hesapla"""
        try:
            # CVD = sum of (close > open ? volume : -volume)
            if len(klines_5m) >= 3:
                cvd = 0
                for k in klines_5m[-3:]:  # Son 3 bar (15 dk)
                    open_p = float(k[1])
                    close_p = float(k[4])
                    volume = float(k[5])
                    
                    if close_p > open_p:
                        cvd += volume  # Buying
                    else:
                        cvd -= volume  # Selling
                
                # Normalize to percentage-like value
                total_vol = sum(float(k[5]) for k in klines_5m[-3:])
                if total_vol > 0:
                    context.cvd_5m = (cvd / total_vol) * 100
            
            # CVD Divergence Check
            # Price going up but CVD negative = bearish divergence
            if context.momentum_5m > 0.3 and context.cvd_5m < -20:
                context.cvd_divergence = True
            elif context.momentum_5m < -0.3 and context.cvd_5m > 20:
                context.cvd_divergence = True
                
        except Exception as e:
            logger.debug(f"CVD analysis error: {e}")
        
        return context
    
    def _analyze_oi(self, context: MomentumContext, oi_data: List) -> MomentumContext:
        """Open Interest analizi"""
        try:
            if len(oi_data) >= 2:
                # 5m change
                latest_oi = float(oi_data[-1].get('sumOpenInterest', 0))
                prev_oi = float(oi_data[-2].get('sumOpenInterest', 0))
                
                if prev_oi > 0:
                    context.oi_change_5m = ((latest_oi - prev_oi) / prev_oi) * 100
                
                # 1h change (if we have enough data)
                if len(oi_data) >= 12:
                    old_oi = float(oi_data[0].get('sumOpenInterest', 0))
                    if old_oi > 0:
                        context.oi_change_1h = ((latest_oi - old_oi) / old_oi) * 100
                
                # Surge detection
                context.oi_surge = abs(context.oi_change_5m) >= self.OI_SURGE_THRESHOLD
                
        except Exception as e:
            logger.debug(f"OI analysis error: {e}")
        
        return context
    
    async def _analyze_liquidations(self, context: MomentumContext, 
                                    current_price: float) -> MomentumContext:
        """Liquidation cluster analizi"""
        try:
            # Get data from existing liquidation hunter
            from src.brain.liquidation_hunter import get_liquidation_hunter
            liq_hunter = get_liquidation_hunter()
            
            liq_data = await liq_hunter.analyze(context.symbol)
            
            if liq_data and liq_data.get('data_available'):
                clusters = liq_data.get('heatmap_clusters', [])
                
                if clusters and current_price > 0:
                    # Find nearest clusters
                    longs = [c for c in clusters if c.get('price', 0) < current_price]
                    shorts = [c for c in clusters if c.get('price', 0) > current_price]
                    
                    if longs:
                        context.nearest_liq_long = max(c['price'] for c in longs)
                    if shorts:
                        context.nearest_liq_short = min(c['price'] for c in shorts)
                    
                    # Determine magnet
                    dist_to_long = abs(current_price - context.nearest_liq_long) / current_price * 100 if context.nearest_liq_long else 100
                    dist_to_short = abs(context.nearest_liq_short - current_price) / current_price * 100 if context.nearest_liq_short else 100
                    
                    if dist_to_long < dist_to_short and dist_to_long < self.LIQ_DISTANCE_ALERT:
                        context.liq_magnet = "LONG_LIQ"
                        context.liq_distance_pct = dist_to_long
                    elif dist_to_short < self.LIQ_DISTANCE_ALERT:
                        context.liq_magnet = "SHORT_LIQ"
                        context.liq_distance_pct = dist_to_short
                        
        except Exception as e:
            logger.debug(f"Liquidation analysis error: {e}")
        
        return context
    
    def _calculate_breakout_probability(self, context: MomentumContext) -> MomentumContext:
        """Breakout olasılığı hesapla"""
        try:
            prob = 0
            
            # Volume spike = high probability
            if context.volume_spike:
                prob += 30
            elif context.volume_ratio > 1.5:
                prob += 15
            
            # Strong momentum = high probability
            if context.momentum_direction in ["STRONG_BULL", "STRONG_BEAR"]:
                prob += 25
            elif context.momentum_direction in ["BULL", "BEAR"]:
                prob += 10
            
            # OI surge = position building
            if context.oi_surge:
                prob += 20
            
            # Close to liquidation cluster = magnet effect
            if context.liq_magnet != "NONE":
                prob += 15
            
            # CVD divergence = reversal coming
            if context.cvd_divergence:
                prob += 10
            
            context.breakout_probability = min(prob, 100)
            
            # Expected direction
            if context.momentum_direction in ["STRONG_BULL", "BULL"]:
                context.expected_direction = "UP"
            elif context.momentum_direction in ["STRONG_BEAR", "BEAR"]:
                context.expected_direction = "DOWN"
            else:
                if context.liq_magnet == "LONG_LIQ":
                    context.expected_direction = "DOWN"  # Price moves towards liq
                elif context.liq_magnet == "SHORT_LIQ":
                    context.expected_direction = "UP"
                else:
                    context.expected_direction = "UNKNOWN"
                    
        except Exception as e:
            logger.debug(f"Breakout probability error: {e}")
        
        return context
    
    def _generate_alerts(self, context: MomentumContext) -> MomentumContext:
        """Alert'ler oluştur"""
        now = datetime.now().isoformat()
        
        # Volume Spike Alert
        if context.volume_spike:
            context.alerts.append(MomentumAlert(
                symbol=context.symbol,
                alert_type="VOLUME_SPIKE",
                severity="HIGH" if context.volume_ratio > 4 else "MEDIUM",
                direction="NEUTRAL",
                value=context.volume_ratio,
                threshold=self.VOLUME_SPIKE_THRESHOLD,
                message=f"Volume {context.volume_ratio:.1f}x ortalamadan fazla!",
                timestamp=now
            ))
        
        # Momentum Surge Alert
        if abs(context.momentum_5m) > 1.0:  # 1% in 5 minutes
            context.alerts.append(MomentumAlert(
                symbol=context.symbol,
                alert_type="MOMENTUM_SURGE",
                severity="HIGH",
                direction="BULLISH" if context.momentum_5m > 0 else "BEARISH",
                value=context.momentum_5m,
                threshold=1.0,
                message=f"5 dakikada {context.momentum_5m:+.2f}% hareket!",
                timestamp=now
            ))
        
        # OI Surge Alert
        if context.oi_surge:
            context.alerts.append(MomentumAlert(
                symbol=context.symbol,
                alert_type="OI_SURGE",
                severity="MEDIUM",
                direction="NEUTRAL",
                value=context.oi_change_5m,
                threshold=self.OI_SURGE_THRESHOLD,
                message=f"Open Interest {context.oi_change_5m:+.1f}% değişti!",
                timestamp=now
            ))
        
        # CVD Divergence Alert
        if context.cvd_divergence:
            context.alerts.append(MomentumAlert(
                symbol=context.symbol,
                alert_type="CVD_DIVERGENCE",
                severity="HIGH",
                direction="BEARISH" if context.momentum_5m > 0 else "BULLISH",
                value=context.cvd_5m,
                threshold=0,
                message="Fiyat ve hacim uyuşmuyor - Dönüş yakın!",
                timestamp=now
            ))
        
        # Liquidation Cluster Alert
        if context.liq_magnet != "NONE" and context.liq_distance_pct < 1.5:
            context.alerts.append(MomentumAlert(
                symbol=context.symbol,
                alert_type="LIQ_CLUSTER",
                severity="CRITICAL",
                direction="BEARISH" if context.liq_magnet == "LONG_LIQ" else "BULLISH",
                value=context.liq_distance_pct,
                threshold=self.LIQ_DISTANCE_ALERT,
                message=f"Liq cluster sadece %{context.liq_distance_pct:.1f} uzakta!",
                timestamp=now
            ))
        
        return context


# Singleton
_momentum_detector: Optional[MomentumDetector] = None


def get_momentum_detector() -> MomentumDetector:
    global _momentum_detector
    if _momentum_detector is None:
        _momentum_detector = MomentumDetector()
    return _momentum_detector


async def get_momentum_context(symbol: str) -> MomentumContext:
    """Quick access to momentum analysis"""
    detector = get_momentum_detector()
    return await detector.analyze(symbol)
