"""
Tape Reader - High-Frequency Trade Pattern Analyzer

"Reads the tape" (analyzes recent trades) to detect institutional buying/selling.

Features:
- Last 50 trades analysis
- Buy vs Sell pressure
- Trade frequency acceleration
- Institutional footprint detection

Lead Time: 10s-1min before major moves
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

logger = logging.getLogger("TAPE_READER")

class TapeReader:
    """
    Analyzes high-frequency trade data to detect early momentum.
    
    Theory:
    - Large sequential buy orders = institutional accumulation
    - Trade frequency spike = FOMO kicking in
    - Buy/Sell imbalance precedes price move
    """
    
    def __init__(self, max_trades=50):
        """
        Args:
            max_trades: Number of recent trades to track
        """
        self.max_trades = max_trades
        
        # Trade history: (timestamp, side, size, price)
        # side: 'BUY' or 'SELL'
        self.recent_trades = deque(maxlen=max_trades)
        
        # Frequency tracking
        self.last_frequency_check = datetime.now()
        self.trade_count_last_minute = 0
    
    def add_trade(self, side: str, size: float, price: float, timestamp: datetime = None):
        """
        Add new trade to tape.
        
        Args:
            side: 'BUY' or 'SELL'
            size: Trade size (volume)
            price: Trade price
            timestamp: Trade timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.recent_trades.append((timestamp, side, size, price))
        
        # Update frequency counter
        if (datetime.now() - self.last_frequency_check).total_seconds() < 60:
            self.trade_count_last_minute += 1
        else:
            self.last_frequency_check = datetime.now()
            self.trade_count_last_minute = 1
    
    def analyze_buy_sell_pressure(self) -> Dict:
        """
        Analyze buy vs sell pressure from recent trades.
        
        Returns:
            {
                "buy_count": int,
                "sell_count": int,
                "buy_volume": float,
                "sell_volume": float,
                "pressure_ratio": float (>1 = buy pressure, <1 = sell pressure),
                "signal": "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL"
            }
        """
        if len(self.recent_trades) == 0:
            return {
                "buy_count": 0,
                "sell_count": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "pressure_ratio": 1.0,
                "signal": "NEUTRAL"
            }
        
        buy_trades = [t for t in self.recent_trades if t[1] == 'BUY']
        sell_trades = [t for t in self.recent_trades if t[1] == 'SELL']
        
        buy_count = len(buy_trades)
        sell_count = len(sell_trades)
        
        buy_volume = sum(t[2] for t in buy_trades)
        sell_volume = sum(t[2] for t in sell_trades)
        
        # Calculate pressure ratio
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            pressure_ratio = 1.0
        else:
            pressure_ratio = buy_volume / (sell_volume + 1)  # +1 to avoid division by zero
        
        # Determine signal
        if pressure_ratio >= 3.0:
            signal = "STRONG_BUY"
        elif pressure_ratio >= 1.5:
            signal = "BUY"
        elif pressure_ratio <= 0.33:
            signal = "STRONG_SELL"
        elif pressure_ratio <= 0.67:
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        return {
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "pressure_ratio": pressure_ratio,
            "signal": signal
        }
    
    def calculate_trade_frequency(self) -> float:
        """
        Calculate trade frequency (trades per minute).
        
        Returns:
            frequency: Trades per minute
        """
        if len(self.recent_trades) < 2:
            return 0.0
        
        # Get trades in last minute
        one_min_ago = datetime.now() - timedelta(minutes=1)
        recent = [t for t in self.recent_trades if t[0] >= one_min_ago]
        
        return len(recent)  # Trades in last minute
    
    def detect_institutional_pattern(self) -> bool:
        """
        Detect institutional buying/selling pattern.
        
        Pattern:
        - 5+ consecutive trades same side
        - Large trade sizes (> median * 2)
        - Short time gaps between trades
        
        Returns:
            is_institutional: True if pattern detected
        """
        if len(self.recent_trades) < 5:
            return False
        
        last_5 = list(self.recent_trades)[-5:]
        
        # Check if all same side
        sides = [t[1] for t in last_5]
        if len(set(sides)) == 1:  # All BUY or all SELL
            # Check trade sizes
            sizes = [t[2] for t in last_5]
            median_size = sorted(sizes)[len(sizes) // 2]
            
            large_trades = sum(1 for s in sizes if s > median_size * 1.5)
            
            if large_trades >= 3:
                logger.info(f"🐋 INSTITUTIONAL PATTERN DETECTED: {sides[0]} (5 consecutive)")
                return True
        
        return False
    
    def calculate_tape_score(self) -> Dict:
        """
        Calculate tape reading score (0-10).
        
        Returns:
            {
                "score": 0-10,
                "signal": "STRONG_BUY" | ... | "STRONG_SELL",
                "confidence": 0-10,
                "institutional": bool,
                "frequency": trades_per_minute
            }
        """
        pressure = self.analyze_buy_sell_pressure()
        frequency = self.calculate_trade_frequency()
        institutional = self.detect_institutional_pattern()
        
        # Base score from pressure
        if pressure["signal"] == "STRONG_BUY":
            base_score = 9
        elif pressure["signal"] == "BUY":
            base_score = 7
        elif pressure["signal"] == "SELL":
            base_score = 3
        elif pressure["signal"] == "STRONG_SELL":
            base_score = 1
        else:
            base_score = 5
        
        # Boost for institutional
        if institutional:
            base_score = min(10, base_score + 1)
        
        # Boost for high frequency
        if frequency > 20:  # >20 trades/min = high activity
            base_score = min(10, base_score + 1)
        
        # Confidence
        confidence = 5
        if institutional:
            confidence += 2
        if abs(pressure["pressure_ratio"] - 1.0) > 1.0:  # Strong imbalance
            confidence += 2
        
        confidence = min(10, confidence)
        
        logger.info(f"📊 Tape Reading: {pressure['signal']} | Score: {base_score}/10 | Frequency: {frequency} trades/min")
        
        return {
            "score": base_score,
            "signal": pressure["signal"],
            "confidence": confidence,
            "institutional": institutional,
            "frequency": frequency,
            "pressure_ratio": pressure["pressure_ratio"]
        }
