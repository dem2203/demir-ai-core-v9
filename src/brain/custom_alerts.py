# -*- coding: utf-8 -*-
"""
DEMIR AI - CUSTOM PRICE ALERTS
==============================
Kullanıcı tanımlı fiyat ve indikatör uyarıları.

Kullanım:
- BTC $90K'yı geçerse uyar
- ETH RSI < 30 olursa uyar
- SOL funding > 0.05% olursa uyar
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger("CUSTOM_ALERTS")


@dataclass
class PriceAlert:
    """Custom price alert"""
    id: str
    symbol: str
    condition: str  # 'above', 'below'
    target_price: float
    triggered: bool = False
    created_at: str = ""
    triggered_at: str = ""


@dataclass
class IndicatorAlert:
    """Custom indicator alert"""
    id: str
    symbol: str
    indicator: str  # 'rsi', 'funding', 'ob_ratio'
    condition: str  # 'above', 'below'
    target_value: float
    triggered: bool = False
    created_at: str = ""
    triggered_at: str = ""


class CustomAlertManager:
    """Manages custom alerts"""
    
    def __init__(self):
        self.alerts_path = Path("src/brain/storage/custom_alerts.json")
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self.price_alerts: List[PriceAlert] = []
        self.indicator_alerts: List[IndicatorAlert] = []
        self._load_alerts()
    
    def _load_alerts(self):
        """Load alerts from file"""
        if self.alerts_path.exists():
            try:
                with open(self.alerts_path) as f:
                    data = json.load(f)
                    self.price_alerts = [PriceAlert(**a) for a in data.get('price', [])]
                    self.indicator_alerts = [IndicatorAlert(**a) for a in data.get('indicator', [])]
            except Exception as e:
                logger.error(f"Failed to load alerts: {e}")
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            data = {
                'price': [asdict(a) for a in self.price_alerts],
                'indicator': [asdict(a) for a in self.indicator_alerts]
            }
            with open(self.alerts_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def add_price_alert(self, symbol: str, condition: str, target: float) -> str:
        """
        Add a price alert
        
        Args:
            symbol: 'BTCUSDT'
            condition: 'above' or 'below'
            target: Target price
        
        Returns:
            Alert ID
        """
        alert_id = f"PA_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        alert = PriceAlert(
            id=alert_id,
            symbol=symbol,
            condition=condition,
            target_price=target,
            created_at=datetime.now().isoformat()
        )
        self.price_alerts.append(alert)
        self._save_alerts()
        logger.info(f"Added price alert: {symbol} {condition} ${target:,.0f}")
        return alert_id
    
    def add_indicator_alert(self, symbol: str, indicator: str, condition: str, target: float) -> str:
        """
        Add an indicator alert
        
        Args:
            symbol: 'BTCUSDT'
            indicator: 'rsi', 'funding', 'ob_ratio'
            condition: 'above' or 'below'
            target: Target value
        
        Returns:
            Alert ID
        """
        alert_id = f"IA_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        alert = IndicatorAlert(
            id=alert_id,
            symbol=symbol,
            indicator=indicator,
            condition=condition,
            target_value=target,
            created_at=datetime.now().isoformat()
        )
        self.indicator_alerts.append(alert)
        self._save_alerts()
        logger.info(f"Added indicator alert: {symbol} {indicator} {condition} {target}")
        return alert_id
    
    def check_price_alerts(self, current_prices: Dict[str, float]) -> List[PriceAlert]:
        """
        Check price alerts and return triggered ones
        
        Args:
            current_prices: {'BTCUSDT': 87500, 'ETHUSDT': 3100, ...}
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for alert in self.price_alerts:
            if alert.triggered:
                continue
            
            current = current_prices.get(alert.symbol, 0)
            if current <= 0:
                continue
            
            if alert.condition == 'above' and current >= alert.target_price:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
                logger.info(f"🔔 Price alert triggered: {alert.symbol} above ${alert.target_price:,.0f}")
            
            elif alert.condition == 'below' and current <= alert.target_price:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
                logger.info(f"🔔 Price alert triggered: {alert.symbol} below ${alert.target_price:,.0f}")
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def check_indicator_alerts(self, indicator_data: Dict[str, Dict]) -> List[IndicatorAlert]:
        """
        Check indicator alerts
        
        Args:
            indicator_data: {'BTCUSDT': {'rsi': 65, 'funding': 0.01, ...}, ...}
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for alert in self.indicator_alerts:
            if alert.triggered:
                continue
            
            symbol_data = indicator_data.get(alert.symbol, {})
            current = symbol_data.get(alert.indicator, None)
            
            if current is None:
                continue
            
            if alert.condition == 'above' and current >= alert.target_value:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
                logger.info(f"🔔 Indicator alert: {alert.symbol} {alert.indicator} above {alert.target_value}")
            
            elif alert.condition == 'below' and current <= alert.target_value:
                alert.triggered = True
                alert.triggered_at = datetime.now().isoformat()
                triggered.append(alert)
                logger.info(f"🔔 Indicator alert: {alert.symbol} {alert.indicator} below {alert.target_value}")
        
        if triggered:
            self._save_alerts()
        
        return triggered
    
    def get_active_alerts(self) -> Dict:
        """Get all active (non-triggered) alerts"""
        return {
            'price': [a for a in self.price_alerts if not a.triggered],
            'indicator': [a for a in self.indicator_alerts if not a.triggered]
        }
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert by ID"""
        for i, alert in enumerate(self.price_alerts):
            if alert.id == alert_id:
                del self.price_alerts[i]
                self._save_alerts()
                return True
        
        for i, alert in enumerate(self.indicator_alerts):
            if alert.id == alert_id:
                del self.indicator_alerts[i]
                self._save_alerts()
                return True
        
        return False
    
    def clear_triggered(self):
        """Clear all triggered alerts"""
        self.price_alerts = [a for a in self.price_alerts if not a.triggered]
        self.indicator_alerts = [a for a in self.indicator_alerts if not a.triggered]
        self._save_alerts()


# Singleton
_alert_manager: Optional[CustomAlertManager] = None


def get_custom_alert_manager() -> CustomAlertManager:
    """Get or create CustomAlertManager singleton"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = CustomAlertManager()
    return _alert_manager
