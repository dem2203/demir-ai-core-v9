# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ENHANCED MAIN ENGINE + EARLY SIGNAL
===================================================
Ana döngü - Early Signal Engine entegrasyonu.

PHASE 400: Early Signal Engine - Leading indicators kullanır.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.v10.data_hub import get_data_hub, MarketSnapshot
from src.v10.enhanced_predictor import get_enhanced_predictor, SignalType
from src.v10.smart_notifier import get_notifier
from src.v10.performance_tracker import get_performance_tracker
from src.v10.lstm_predictor import get_lstm_predictor
from src.v10.early_signal_engine import get_early_signal_engine

logger = logging.getLogger("V10_ENGINE")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


class V10Engine:
    """
    DEMIR AI v10 - ENHANCED Ana Motor + Early Signal Engine
    """
    
    SCAN_INTERVAL = 30
    SIGNAL_COOLDOWN = 30 * 60
    PERFORMANCE_REPORT_INTERVAL = 4 * 60 * 60
    
    def __init__(self):
        self.data_hub = get_data_hub()
        self.predictor = get_enhanced_predictor()
        self.notifier = get_notifier()
        self.performance_tracker = get_performance_tracker()
        self.lstm_predictor = get_lstm_predictor()
        self.early_signal_engine = None  # Lazy init
        
        self._last_signal_time: Dict[str, datetime] = {}
        self._last_performance_report = datetime.now()
        self._running = False
        self._cycle_count = 0
        self._signal_count = 0
        self._error_count = 0
        
        logger.info("[START] DEMIR AI v10 + EARLY SIGNAL ENGINE initialized")
    
    async def start(self):
        """Ana donguyu baslat"""
        self._running = True
        self.notifier.send_startup_message()
        logger.info("[RUN] V10 Engine started with Early Signal...")
        
        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"[ERROR] SCAN CYCLE: {e}")
                self._error_count += 1
                if self._error_count >= 3:
                    self.notifier.send_error_alert(f"Tekrarlayan hata: {e}")
            
            await asyncio.sleep(self.SCAN_INTERVAL)
    
    async def stop(self):
        """Motoru durdur"""
        self._running = False
        await self.data_hub.close()
        if self.early_signal_engine:
            await self.early_signal_engine.close()
        logger.info("[STOP] V10 Engine stopped")
    
    async def _scan_cycle(self):
        """Tek bir tarama dongusu"""
        self._cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"=== SCAN CYCLE #{self._cycle_count} ===")
        
        snapshots = await self.data_hub.get_all_snapshots()
        
        for symbol, snapshot in snapshots.items():
            try:
                await self._process_coin(symbol, snapshot)
            except Exception as e:
                logger.error(f"[ERROR] {symbol}: {e}")
        
        # Technical Analysis (15 dakikada bir)
        if not hasattr(self, '_last_ta_time'):
            self._last_ta_time = datetime.now() - timedelta(minutes=20)
        
        if (datetime.now() - self._last_ta_time).total_seconds() >= 15 * 60:
            try:
                logger.info("[TA] Sending Technical Analysis...")
                self.notifier.send_technical_analysis(snapshots)
                self._last_ta_time = datetime.now()
            except Exception as e:
                logger.error(f"[ERROR] Technical Analysis: {e}")
        
        # Performance report (4 saatte bir)
        if (datetime.now() - self._last_performance_report).total_seconds() >= self.PERFORMANCE_REPORT_INTERVAL:
            try:
                await self.performance_tracker.check_outcomes()
                report_msg = self.performance_tracker.format_report_message()
                self.notifier._send_message(report_msg)
                self._last_performance_report = datetime.now()
            except Exception as e:
                logger.error(f"[ERROR] Performance report: {e}")
        
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        stats = self.data_hub.get_stats()
        
        logger.info(
            f"[OK] Cycle #{self._cycle_count}: {cycle_time:.1f}s | "
            f"Signals: {self._signal_count} | Errors: {self._error_count}"
        )
    
    async def _process_coin(self, symbol: str, snapshot: MarketSnapshot):
        """Tek coin icin sinyal kontrolu - EARLY SIGNAL ENGINE"""
        
        if not snapshot.is_valid:
            if snapshot.errors:
                logger.warning(f"[WARN] {symbol}: Data errors - {snapshot.errors[:2]}")
            return
        
        if self._is_on_cooldown(symbol):
            return
        
        # PHASE 400: EARLY SIGNAL ENGINE
        try:
            if self.early_signal_engine is None:
                self.early_signal_engine = await get_early_signal_engine()
            
            early_signal = await self.early_signal_engine.analyze(symbol)
            
            if early_signal:
                direction = early_signal.leading_signal.direction.value
                logger.info(
                    f"[EARLY] {symbol} | {early_signal.action} | "
                    f"Conf: {early_signal.confidence:.0f}% | {direction}"
                )
                
                if early_signal.confidence >= 60 and early_signal.action != 'HOLD':
                    action_tr = "AL" if early_signal.action == "BUY" else "SAT"
                    tag = "[BUY]" if early_signal.action == "BUY" else "[SELL]"
                    
                    msg = f"""{tag} EARLY SIGNAL: {symbol}

Sinyal: {action_tr}
Guven: {early_signal.confidence:.0f}%
R/R: {early_signal.risk_reward:.1f}x

Giris: ${early_signal.entry_zone[0]:,.0f} - ${early_signal.entry_zone[1]:,.0f}
SL: ${early_signal.stop_loss:,.0f}
TP: ${early_signal.take_profit:,.0f}

Gostergeler: {early_signal.reasoning}"""
                    
                    self.notifier._send_message(msg)
                    self._last_signal_time[symbol] = datetime.now()
                    self._signal_count += 1
                    logger.info(f"[SIGNAL] Early Signal sent: {symbol}")
                    return
                    
        except Exception as e:
            logger.error(f"[ERROR] Early Signal: {symbol}: {e}")
        
        # LEGACY FALLBACK
        signal = await self.predictor.generate_signal_async(snapshot)
        
        if signal.is_valid:
            logger.info(f"[LEGACY] {symbol} {signal.signal_type.value} %{signal.confidence:.0f}")
            
            success = self.notifier.send_trading_signal(signal)
            
            if success:
                self._last_signal_time[symbol] = datetime.now()
                self._signal_count += 1
                try:
                    self.performance_tracker.record_signal(signal)
                except Exception as e:
                    logger.warning(f"Performance tracking error: {e}")
    
    def _is_on_cooldown(self, symbol: str) -> bool:
        if symbol not in self._last_signal_time:
            return False
        elapsed = (datetime.now() - self._last_signal_time[symbol]).total_seconds()
        return elapsed < self.SIGNAL_COOLDOWN
    
    async def scan_once(self) -> Dict[str, dict]:
        """Tek seferlik tarama (test icin)."""
        results = {}
        snapshots = await self.data_hub.get_all_snapshots()
        
        for symbol, snapshot in snapshots.items():
            if snapshot.is_valid:
                signal = self.predictor.generate_signal(snapshot)
                results[symbol] = {
                    'signal': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'is_valid': signal.is_valid
                }
            else:
                results[symbol] = {'signal': 'ERROR', 'errors': snapshot.errors}
        
        return results


_engine: Optional[V10Engine] = None

def get_v10_engine() -> V10Engine:
    global _engine
    if _engine is None:
        _engine = V10Engine()
    return _engine


async def run_v10():
    """V10 Engine'i baslat"""
    engine = get_v10_engine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(run_v10())
