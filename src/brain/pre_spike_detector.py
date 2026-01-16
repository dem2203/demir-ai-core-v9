"""
Pre-Spike Detector - Combined Early Warning System

Integrates all pre-pump indicators into single actionable signal.

Layers:
1. Volume Acceleration (volume_momentum)
2. Order Book Imbalance (market_microstructure)
3. Tape Reading (tape_reader)
4. Funding Rate Changes (market_microstructure)

Output:
- Combined score (0-10)
- PRE-PUMP alert when score ≥ 7.5
- Lead time estimate (30s-2min)
- Confidence rating
"""

import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger("PRE_SPIKE_DETECTOR")

class PreSpikeDetector:
    """
    Professional pre-pump detection combining multiple signals.
    
    Weighted Algorithm:
    - Volume Acceleration: 40% weight (most important)
    - Order Book Imbalance: 30% weight
    - Tape Reading: 20% weight
    - Funding Rate: 10% weight
    
    Threshold: Combined score ≥ 7.5 → PRE-PUMP ALERT
    """
    
    def __init__(self):
        # Weighting factors
        self.weights = {
            "volume": 0.40,
            "orderbook": 0.30,
            "tape": 0.20,
            "funding": 0.10
        }
        
        # Alert threshold
        self.ALERT_THRESHOLD = 7.5
        
        # Last alert tracking (prevent spam)
        self.last_alert_time = {}
        self.MIN_ALERT_INTERVAL = 60  # Seconds between alerts per symbol
    
    def _should_alert(self, symbol: str) -> bool:
        """Check if enough time passed since last alert"""
        if symbol not in self.last_alert_time:
            return True
        
        elapsed = (datetime.now() - self.last_alert_time[symbol]).total_seconds()
        return elapsed >= self.MIN_ALERT_INTERVAL
    
    def analyze(self, 
                volume_score: float,
                orderbook_score: float,
                tape_score: float,
                funding_score: float,
                symbol: str) -> Dict:
        """
        Combine all signals into pre-spike analysis.
        
        Args:
            volume_score: 0-10 from VolumeAccelerationDetector
            orderbook_score: 0-10 from OrderBookImbalance
            tape_score: 0-10 from TapeReader
            funding_score: 0-10 from FundingRate analysis
            symbol: Trading symbol
        
        Returns:
            {
                "combined_score": 0-10,
                "is_pre_pump": bool,
                "confidence": 0-10,
                "lead_time_estimate": "30s-2min" or None,
                "signal": "PRE_PUMP" | "BUILDING" | "NORMAL",
                "breakdown": {...}
            }
        """
        # Weighted combination
        combined_score = (
            (volume_score * self.weights["volume"]) +
            (orderbook_score * self.weights["orderbook"]) +
            (tape_score * self.weights["tape"]) +
            (funding_score * self.weights["funding"])
        )
        
        # Determine signal
        is_pre_pump = combined_score >= self.ALERT_THRESHOLD
        
        if is_pre_pump:
            signal = "PRE_PUMP"
            lead_time = "30s-2min"
            confidence = min(10, int((combined_score / self.ALERT_THRESHOLD) * 9))
        elif combined_score >= 6.5:
            signal = "BUILDING"
            lead_time = "2-5min"
            confidence = 7
        else:
            signal = "NORMAL"
            lead_time = None
            confidence = max(1, int(combined_score))
        
        # Should we alert?
        should_alert = is_pre_pump and self._should_alert(symbol)
        
        if should_alert:
            self.last_alert_time[symbol] = datetime.now()
            logger.warning(f"🚨 PRE-PUMP ALERT: {symbol} | Score: {combined_score:.1f}/10 | Lead: {lead_time}")
        else:
            logger.info(f"📊 Pre-Spike: {symbol} | Score: {combined_score:.1f}/10 | Signal: {signal}")
        
        return {
            "combined_score": round(combined_score, 2),
            "is_pre_pump": is_pre_pump,
            "should_alert": should_alert,
            "confidence": confidence,
            "lead_time_estimate": lead_time,
            "signal": signal,
            "breakdown": {
                "volume": volume_score,
                "orderbook": orderbook_score,
                "tape": tape_score,
                "funding": funding_score
            },
            "weights": self.weights
        }
    
    def get_alert_message(self, symbol: str, analysis: Dict, current_price: float) -> str:
        """
        Generate Telegram alert message for pre-pump.
        
        Args:
            symbol: Trading symbol
            analysis: Result from analyze()
            current_price: Current price
        
        Returns:
            formatted_message: Telegram-ready alert
        """
        lead_time = analysis["lead_time_estimate"]
        score = analysis["combined_score"]
        confidence = analysis["confidence"]
        breakdown = analysis["breakdown"]
        
        message = f"""⚡ **PRE-PUMP ALERT!**

🎯 **{symbol}**
💰 Current: ${current_price:,.2f}

🔮 **Lead Time:** {lead_time}
📊 **Score:** {score}/10
✅ **Confidence:** {confidence}/10

**Signal Breakdown:**
📈 Volume: {breakdown['volume']}/10 (40%)
📊 Order Book: {breakdown['orderbook']}/10 (30%)
📝 Tape: {breakdown['tape']}/10 (20%)
💸 Funding: {breakdown['funding']}/10 (10%)

⏰ **Action:** Watch closely! Price movement expected in {lead_time}.
"""
        return message.strip()
    
    def get_summary(self, symbol: str, analysis: Dict) -> str:
        """Get concise summary for logging"""
        return f"""
Pre-Spike Analysis ({symbol}):
- Combined Score: {analysis['combined_score']}/10
- Signal: {analysis['signal']}
- Confidence: {analysis['confidence']}/10
- Lead Time: {analysis['lead_time_estimate'] or 'N/A'}
- Alert: {'YES ⚡' if analysis['should_alert'] else 'No'}
"""
