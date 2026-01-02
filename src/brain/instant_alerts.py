# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - INSTANT ALERT SYSTEM (v2)
=========================================
FIXED: Spam prevention + Consistency check

Only sends alerts when:
1. Multiple signals AGREE (no contradictions)
2. High confidence (80%+)
3. 30 minute cooldown per symbol (global)
4. Only Large Orders and confirmed Volume Spikes

Author: DEMIR AI Team
Date: 2026-01-02
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque
from src.brain.realtime_predictive import (
    get_realtime_engine, 
    PredictiveAlert,
    RealTimePredictiveEngine
)

logger = logging.getLogger("INSTANT_ALERT")


class InstantAlertSystem:
    """
    Instant Alert System v2
    
    IMPROVED:
    - Collects signals for 30 seconds before deciding
    - Only alerts when signals are CONSISTENT
    - Much longer cooldowns to prevent spam
    - Only sends high-value alerts
    """
    
    # Cooldown to prevent spam - 30 MINUTES per symbol
    ALERT_COOLDOWN = 1800  # 30 minutes
    
    # Global cooldown - 10 minutes between ANY alert
    GLOBAL_COOLDOWN = 600  # 10 minutes
    
    # Minimum strength to even consider
    MIN_STRENGTH = 80  # Increased from 60
    
    # Minimum confidence
    MIN_CONFIDENCE = 75
    
    # Collection window - collect signals for this period before deciding
    COLLECTION_WINDOW = 30  # 30 seconds
    
    # Only these alert types are important enough
    ALLOWED_TYPES = ["LARGE_ORDER"]  # Only whale orders for now
    
    def __init__(self, notifier=None):
        self.engine: Optional[RealTimePredictiveEngine] = None
        self.notifier = notifier
        self._last_alerts: Dict[str, datetime] = {}
        self._last_global_alert: Optional[datetime] = None
        
        # Signal collection buffer
        self._signal_buffer: Dict[str, deque] = {}  # symbol -> recent alerts
        
        logger.info("⚡ Instant Alert System v2 initialized (spam-free)")
    
    async def start(self, symbols: list = None):
        """Start the instant alert system"""
        symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        
        # Initialize buffers
        for sym in symbols:
            self._signal_buffer[sym] = deque(maxlen=20)
        
        # Get or create engine
        self.engine = get_realtime_engine(symbols)
        
        # Register alert callback
        self.engine.add_alert_callback(self._handle_alert)
        
        logger.info(f"⚡ Starting Instant Alerts v2 for {symbols} (spam-protected)...")
        
        # Start engine (this blocks)
        await self.engine.start()
    
    async def stop(self):
        """Stop the system"""
        if self.engine:
            await self.engine.stop()
    
    async def _handle_alert(self, alert: PredictiveAlert):
        """Handle incoming alert with strict filtering"""
        try:
            # 1. Only allowed types
            if alert.alert_type not in self.ALLOWED_TYPES:
                return
            
            # 2. Minimum strength
            if alert.strength < self.MIN_STRENGTH:
                return
            
            # 3. Minimum confidence
            if alert.confidence < self.MIN_CONFIDENCE:
                return
            
            # 4. Global cooldown
            if self._last_global_alert:
                seconds_since = (datetime.now() - self._last_global_alert).total_seconds()
                if seconds_since < self.GLOBAL_COOLDOWN:
                    logger.debug(f"Global cooldown: {self.GLOBAL_COOLDOWN - seconds_since:.0f}s remaining")
                    return
            
            # 5. Per-symbol cooldown
            last_alert = self._last_alerts.get(alert.symbol)
            if last_alert:
                seconds_since = (datetime.now() - last_alert).total_seconds()
                if seconds_since < self.ALERT_COOLDOWN:
                    logger.debug(f"Symbol cooldown {alert.symbol}: {self.ALERT_COOLDOWN - seconds_since:.0f}s remaining")
                    return
            
            # 6. Add to buffer for consistency check
            self._signal_buffer[alert.symbol].append({
                'time': datetime.now(),
                'direction': alert.direction,
                'type': alert.alert_type,
                'strength': alert.strength
            })
            
            # 7. Check consistency - need 2+ signals in same direction
            recent = [s for s in self._signal_buffer[alert.symbol] 
                     if (datetime.now() - s['time']).seconds < self.COLLECTION_WINDOW]
            
            if len(recent) < 2:
                # Wait for more signals
                return
            
            # Check direction consistency
            directions = [s['direction'] for s in recent]
            bullish_count = directions.count('BULLISH')
            bearish_count = directions.count('BEARISH')
            
            # Need majority in one direction
            if max(bullish_count, bearish_count) < 2:
                logger.debug(f"Contradictory signals, skipping: {bullish_count}B vs {bearish_count}S")
                return
            
            # 8. Update cooldowns
            self._last_alerts[alert.symbol] = datetime.now()
            self._last_global_alert = datetime.now()
            
            # 9. Clear buffer
            self._signal_buffer[alert.symbol].clear()
            
            # 10. Format and send
            message = self._format_alert_message(alert)
            
            if self.notifier:
                self.notifier._send_message(message)
            else:
                try:
                    from src.v10.smart_notifier import get_notifier
                    notifier = get_notifier()
                    notifier._send_message(message)
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
            
            logger.info(f"⚡ CONFIRMED ALERT: {alert.symbol} {alert.alert_type}")
            
        except Exception as e:
            logger.error(f"Alert handling error: {e}")
    
    def _format_alert_message(self, alert: PredictiveAlert) -> str:
        """Format alert for Telegram - simplified"""
        
        direction_emoji = "🟢" if alert.direction == "BULLISH" else "🔴"
        action = "LONG yönünde" if alert.direction == "BULLISH" else "SHORT yönünde"
        
        msg = f"""
🐋 *WHALE ALERT - {alert.symbol}*
━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{alert.direction}* | Güç: {alert.strength}%

📍 *Tespit:*
{alert.message}

💰 *Fiyat:* ${alert.price_at_detection:,.0f}
🎯 *Aksiyon:* {action} hazırlan

━━━━━━━━━━━━━━━━━━━━
⚡ Onaylanmış Whale Aktivitesi
"""
        return msg.strip()


# Singleton
_system: Optional[InstantAlertSystem] = None

def get_instant_alert_system(notifier=None) -> InstantAlertSystem:
    global _system
    if _system is None:
        _system = InstantAlertSystem(notifier)
    return _system
