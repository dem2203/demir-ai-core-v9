# -*- coding: utf-8 -*-
"""
DEMIR AI - Notifications Package
7/24 Telegram bildirim sistemi.

UNIFIED: NotificationManager from src/utils/notifications.py is the main notifier.
"""

from .signal_tracker import SignalTracker, ActiveSignal, can_send_for_coin
from .market_monitor import MarketMonitor, start_monitor, get_monitor_status

# Re-export NotificationManager from utils for backwards compatibility
try:
    from src.utils.notifications import NotificationManager
    TelegramNotifier = NotificationManager  # Backwards compatibility alias
except ImportError:
    TelegramNotifier = None
    NotificationManager = None

__all__ = [
    'SignalTracker',
    'ActiveSignal', 
    'can_send_for_coin',
    'NotificationManager',
    'TelegramNotifier',  # Alias for backwards compatibility
    'MarketMonitor',
    'start_monitor',
    'get_monitor_status'
]
