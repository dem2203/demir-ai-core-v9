# -*- coding: utf-8 -*-
"""
DEMIR AI - 7/24 Market Monitor
Ana izleme döngüsü - tüm sistemi koordine eder.

PHASE 50: 24/7 Signal System
- Her dakika sinyal kontrolü
- Her 30 saniye aktif sinyal fiyat takibi
- Her 15 dakika haber tarama
- 1 saat sessizlikten sonra heartbeat
"""
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

logger = logging.getLogger("MARKET_MONITOR")


class MarketMonitor:
    """
    7/24 Piyasa İzleme Sistemi
    
    Koordine eder:
    - SignalOrchestrator (sinyal üretimi)
    - SignalTracker (aktif sinyal takibi)
    - TelegramNotifier (bildirimler)
    - ResearchAgent (araştırma)
    """
    
    COINS = ['BTCUSDT', 'ETHUSDT']
    
    # Zamanlama (saniye)
    SIGNAL_CHECK_INTERVAL = 60  # Her 1 dakika
    PRICE_CHECK_INTERVAL = 30   # Her 30 saniye
    NEWS_CHECK_INTERVAL = 900   # Her 15 dakika
    HEARTBEAT_INTERVAL = 3600   # Her 1 saat
    
    def __init__(self):
        self.running = False
        self.last_signal_check = datetime.now()
        self.last_price_check = datetime.now()
        self.last_news_check = datetime.now()
        self.last_heartbeat = datetime.now()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Bileşenleri başlat."""
        try:
            from src.brain.signal_orchestrator import SignalOrchestrator
            self.orchestrator = SignalOrchestrator()
        except Exception as e:
            logger.warning(f"SignalOrchestrator init failed: {e}")
            self.orchestrator = None
        
        try:
            from src.notifications.signal_tracker import SignalTracker
            self.tracker = SignalTracker()
        except Exception as e:
            logger.warning(f"SignalTracker init failed: {e}")
            self.tracker = None
        
        try:
            from src.utils.notifications import NotificationManager
            self.notifier = NotificationManager()
        except Exception as e:
            logger.warning(f"NotificationManager init failed: {e}")
            self.notifier = None
    
    async def check_for_signals(self):
        """Tüm coinler için sinyal kontrolü."""
        if not self.orchestrator:
            return
        
        for symbol in self.COINS:
            try:
                # Bu coin için sinyal gönderilebilir mi?
                if self.tracker:
                    can_send = self.tracker.can_send_signal(symbol)
                    if not can_send['allowed']:
                        logger.debug(f"{symbol}: {can_send['reason']}")
                        continue
                
                # Sinyal oluştur
                signal = await self.orchestrator.orchestrate(symbol)
                
                if signal:
                    # Telegram'a gönder
                    if self.notifier:
                        self.notifier.send_signal(
                            symbol=symbol,
                            direction=signal.direction,
                            entry=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            tp1=signal.take_profit,
                            tp2=signal.take_profit * 1.02 if signal.direction == 'LONG' else signal.take_profit * 0.98,
                            confidence=signal.confidence,
                            modules=signal.contributing_modules,
                            signal_id=f"{symbol[:3]}-{datetime.now().strftime('%m%d%H%M')}"
                        )
                    
                    # Tracker'a kaydet
                    if self.tracker:
                        self.tracker.register_signal(
                            symbol=symbol,
                            direction=signal.direction,
                            entry=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            tp1=signal.take_profit,
                            tp2=signal.take_profit * 1.02 if signal.direction == 'LONG' else signal.take_profit * 0.98,
                            confidence=signal.confidence
                        )
                    
                    logger.info(f"🎯 Signal sent: {symbol} {signal.direction}")
            
            except Exception as e:
                logger.error(f"Signal check error for {symbol}: {e}")
    
    def check_active_signals(self):
        """Aktif sinyallerin fiyat kontrolü."""
        if not self.tracker:
            return
        
        events = self.tracker.check_all_signals()
        
        for event in events:
            if self.notifier:
                signal = event['signal']
                
                # TP veya SL bildirimi gönder
                if 'TP' in event['event'] or 'SL' in event['event']:
                    # Süreyi hesapla
                    created = datetime.fromisoformat(signal['created_at'])
                    duration = (datetime.now() - created).total_seconds() / 3600
                    
                    self.notifier.send_result(
                        symbol=signal['symbol'],
                        direction=signal['direction'],
                        result_type=event['event'],
                        entry=signal['entry_price'],
                        exit_price=signal['take_profit_1'] if 'TP' in event['event'] else signal['stop_loss'],
                        profit_pct=event['profit_pct'],
                        duration_hours=duration,
                        signal_id=signal['signal_id']
                    )
                    
                    logger.info(f"📊 Result sent: {signal['symbol']} {event['event']}")
    
    def check_heartbeat(self):
        """Heartbeat gerekiyorsa gönder."""
        if not self.notifier:
            return
        
        if self.notifier.should_send_heartbeat():
            prices = self.notifier.get_current_prices()
            self.notifier.send_heartbeat(prices)
            self.last_heartbeat = datetime.now()
            logger.info("💚 Heartbeat sent")
    
    async def run_cycle(self):
        """Tek bir izleme döngüsü."""
        now = datetime.now()
        
        # 1. Sinyal kontrolü (her 60 saniye)
        if (now - self.last_signal_check).total_seconds() >= self.SIGNAL_CHECK_INTERVAL:
            await self.check_for_signals()
            self.last_signal_check = now
        
        # 2. Aktif sinyal fiyat kontrolü (her 30 saniye)
        if (now - self.last_price_check).total_seconds() >= self.PRICE_CHECK_INTERVAL:
            self.check_active_signals()
            self.last_price_check = now
        
        # 3. Heartbeat kontrolü (her 1 saat)
        if (now - self.last_heartbeat).total_seconds() >= self.HEARTBEAT_INTERVAL:
            self.check_heartbeat()
    
    async def run(self):
        """Ana izleme döngüsü."""
        logger.info("🚀 Market Monitor started")
        self.running = True
        
        while self.running:
            try:
                await self.run_cycle()
                await asyncio.sleep(10)  # Her 10 saniyede kontrol
            except Exception as e:
                logger.error(f"Monitor cycle error: {e}")
                await asyncio.sleep(30)
        
        logger.info("🛑 Market Monitor stopped")
    
    def start_background(self):
        """Arka planda başlat."""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run())
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        logger.info("📡 Monitor running in background")
        return thread
    
    def stop(self):
        """Durdurmak."""
        self.running = False
        logger.info("Monitor stopping...")
    
    def get_status(self) -> Dict:
        """İzleme durumu."""
        return {
            'running': self.running,
            'last_signal_check': self.last_signal_check.isoformat(),
            'last_price_check': self.last_price_check.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'active_signals': self.tracker.get_active_signals() if self.tracker else [],
            'stats': self.tracker.get_statistics() if self.tracker else {}
        }


# Convenience functions
def start_monitor() -> MarketMonitor:
    """Monitor başlat."""
    monitor = MarketMonitor()
    monitor.start_background()
    return monitor


def get_monitor_status() -> Dict:
    """Durum al."""
    monitor = MarketMonitor()
    return monitor.get_status()
