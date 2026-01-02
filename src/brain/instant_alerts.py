# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - INSTANT ALERT SYSTEM
====================================
Connects real-time predictive engine to Telegram.

Sends IMMEDIATE alerts when institutional activity detected:
- Large orders ($500K+)
- Volume spikes (3x+)
- Order imbalance (40%+)
- CVD divergence

Author: DEMIR AI Team
Date: 2026-01-02
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict
from src.brain.realtime_predictive import (
    get_realtime_engine, 
    PredictiveAlert,
    RealTimePredictiveEngine
)

logger = logging.getLogger("INSTANT_ALERT")


class InstantAlertSystem:
    """
    Instant Alert System
    
    Bridges real-time predictive engine to Telegram for
    immediate notification of institutional activity.
    """
    
    # Cooldown to prevent spam (seconds)
    ALERT_COOLDOWN = 300  # 5 minutes per symbol per type
    
    # Minimum strength to alert
    MIN_STRENGTH = 60
    
    # Priority levels
    PRIORITY_IMMEDIATE = ["LARGE_ORDER", "VOLUME_SPIKE"]
    PRIORITY_SOON = ["ORDER_IMBALANCE", "CVD_DIVERGENCE"]
    
    def __init__(self, notifier=None):
        self.engine: Optional[RealTimePredictiveEngine] = None
        self.notifier = notifier
        self._last_alerts: Dict[str, datetime] = {}
        
        logger.info("⚡ Instant Alert System initialized")
    
    async def start(self, symbols: list = None):
        """Start the instant alert system"""
        symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        
        # Get or create engine
        self.engine = get_realtime_engine(symbols)
        
        # Register alert callback
        self.engine.add_alert_callback(self._handle_alert)
        
        logger.info(f"⚡ Starting Instant Alerts for {symbols}...")
        
        # Start engine (this blocks)
        await self.engine.start()
    
    async def stop(self):
        """Stop the system"""
        if self.engine:
            await self.engine.stop()
    
    async def _handle_alert(self, alert: PredictiveAlert):
        """Handle incoming alert from predictive engine"""
        try:
            # Check strength threshold
            if alert.strength < self.MIN_STRENGTH:
                return
            
            # Check cooldown
            cooldown_key = f"{alert.symbol}_{alert.alert_type}"
            last_alert = self._last_alerts.get(cooldown_key)
            
            if last_alert:
                seconds_since = (datetime.now() - last_alert).total_seconds()
                if seconds_since < self.ALERT_COOLDOWN:
                    logger.debug(f"Alert cooldown: {cooldown_key}")
                    return
            
            # Update last alert time
            self._last_alerts[cooldown_key] = datetime.now()
            
            # Format message
            message = self._format_alert_message(alert)
            
            # Send to Telegram
            if self.notifier:
                self.notifier._send_message(message)
            else:
                # Try to get notifier
                try:
                    from src.v10.smart_notifier import get_notifier
                    notifier = get_notifier()
                    notifier._send_message(message)
                except Exception as e:
                    logger.error(f"Failed to send alert: {e}")
                    print(f"🚨 ALERT: {message}")
            
            logger.info(f"⚡ Alert sent: {alert.symbol} {alert.alert_type}")
            
        except Exception as e:
            logger.error(f"Alert handling error: {e}")
    
    def _format_alert_message(self, alert: PredictiveAlert) -> str:
        """Format alert for Telegram"""
        
        # Emoji based on alert type
        type_emoji = {
            "LARGE_ORDER": "🐋",
            "VOLUME_SPIKE": "📊",
            "ORDER_IMBALANCE": "⚖️",
            "CVD_DIVERGENCE": "🔄",
            "LIQ_CASCADE": "💥"
        }
        
        emoji = type_emoji.get(alert.alert_type, "🚨")
        direction_emoji = "🟢" if alert.direction == "BULLISH" else "🔴"
        
        # Urgency indicator
        urgency_text = {
            "IMMEDIATE": "⚡ ACT NOW",
            "SOON": "⏰ Prepare",
            "WATCH": "👀 Monitor"
        }
        
        msg = f"""
{emoji} *ERKEN UYARI - {alert.symbol}*
━━━━━━━━━━━━━━━━━━━━

{direction_emoji} *{alert.direction}* | Güç: {alert.strength}%

📍 *Tespit:*
{alert.message}

💰 *Fiyat:* ${alert.price_at_detection:,.2f}
📈 *Beklenen Hareket:* {alert.expected_move_pct:+.1f}%
🎯 *Güven:* {alert.confidence}%

⏱️ *Aciliyet:* {urgency_text.get(alert.urgency, alert.urgency)}

━━━━━━━━━━━━━━━━━━━━
⚡ Real-Time Order Flow Alert
"""
        return msg.strip()


# Integration function for main engine
async def run_instant_alerts(symbols: list = None, notifier=None):
    """Run instant alert system as background task"""
    system = InstantAlertSystem(notifier)
    await system.start(symbols)


# Singleton
_system: Optional[InstantAlertSystem] = None

def get_instant_alert_system(notifier=None) -> InstantAlertSystem:
    global _system
    if _system is None:
        _system = InstantAlertSystem(notifier)
    return _system
