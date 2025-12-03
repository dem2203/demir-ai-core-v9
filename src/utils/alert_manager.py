import logging
from typing import Dict
from src.utils.notifications import NotificationManager

logger = logging.getLogger("ALERT_MANAGER")

class AlertManager:
    """
    DEMIR AI V22.0 - SMART ALERT MANAGER
    
    Sends critical real-time alerts to Telegram.
    Prevents spam with cooldown periods.
    """
    
    COOLDOWN_SECONDS = 300  # 5 minutes between similar alerts
    
    def __init__(self):
        self.notifier = NotificationManager()
        self.last_alerts = {}  # Track last alert time per type
    
    async def send_anomaly_alert(self, symbol: str, anomaly_data: Dict):
        """Sends anomaly detection alert."""
        alert_key = f"anomaly_{symbol}"
        
        if self._is_on_cooldown(alert_key):
            return
        
        message = (
            f"🚨 **ANOMALY DETECTED**\n"
            f"━━━━━━━━━━━━━━\n"
            f"**Asset:** {symbol}\n"
            f"**Volume Surge:** {anomaly_data['volume_surge']:.1f}x normal\n"
            f"**Price Change:** {anomaly_data['price_change_pct']:+.2f}%\n"
            f"**Anomaly Score:** {abs(anomaly_data['anomaly_score'])*100:.0f}/100\n"
            f"━━━━━━━━━━━━━━\n"
            f"⚠️ Unusual market activity detected!"
        )
        
        await self.notifier.send_message_raw(message)
        self._mark_sent(alert_key)
        logger.info(f"✅ Anomaly alert sent for {symbol}")
    
    async def send_orderflow_alert(self, symbol: str, flow_data: Dict):
        """Sends order flow imbalance alert."""
        alert_key = f"orderflow_{symbol}"
        
        if self._is_on_cooldown(alert_key):
            return
        
        direction_icon = "📈" if flow_data['direction'] == 'BUY' else "📉"
        
        message = (
            f"{direction_icon} **ORDER FLOW IMBALANCE**\n"
            f"━━━━━━━━━━━━━━\n"
            f"**Asset:** {symbol}\n"
            f"**Net Pressure:** ${flow_data['net_flow_usd']/1e6:.2f}M ({flow_data['direction']})\n"
            f"**Bid/Ask Ratio:** {flow_data['imbalance_ratio']:.2f}\n"
            f"━━━━━━━━━━━━━━\n"
            f"🐋 Large players are {'accumulating' if flow_data['direction'] == 'BUY' else 'distributing'}!"
        )
        
        await self.notifier.send_message_raw(message)
        self._mark_sent(alert_key)
        logger.info(f"✅ Order flow alert sent for {symbol}")
    
    async def send_funding_alert(self, symbol: str, binance_rate: float, bybit_rate: float):
        """Sends funding rate divergence alert."""
        alert_key = f"funding_{symbol}"
        
        if self._is_on_cooldown(alert_key):
            return
        
        divergence = abs(binance_rate - bybit_rate)
        extreme = max(abs(binance_rate), abs(bybit_rate))
        
        alert_reason = []
        if extreme > 0.001:  # 0.1%
            alert_reason.append(f"Extreme funding: {extreme*100:.2f}%")
        if divergence > 0.0015:  # 0.15%
            alert_reason.append(f"Divergence: {divergence*100:.2f}%")
        
        message = (
            f"⚡ **FUNDING RATE ALERT**\n"
            f"━━━━━━━━━━━━━━\n"
            f"**Asset:** {symbol}\n"
            f"**Binance:** {binance_rate*100:+.3f}%\n"
            f"**Bybit:** {bybit_rate*100:+.3f}%\n"
            f"**Divergence:** {divergence*100:.3f}%\n"
            f"━━━━━━━━━━━━━━\n"
            f"📊 {' | '.join(alert_reason)}"
        )
        
        await self.notifier.send_message_raw(message)
        self._mark_sent(alert_key)
        logger.info(f"✅ Funding alert sent for {symbol}")
    
    def _is_on_cooldown(self, alert_key: str) -> bool:
        """Checks if alert is on cooldown."""
        import time
        if alert_key in self.last_alerts:
            elapsed = time.time() - self.last_alerts[alert_key]
            return elapsed < self.COOLDOWN_SECONDS
        return False
    
    def _mark_sent(self, alert_key: str):
        """Marks alert as sent."""
        import time
        self.last_alerts[alert_key] = time.time()
