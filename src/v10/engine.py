# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - ENHANCED MAIN ENGINE
====================================
Ana döngü - gelişmiş modüllerle tahmin üret.

YENİ ÖZELLİKLER:
- Enhanced Predictor (7 temel + 7 gelişmiş gösterge)
- Multi-Timeframe Confluence (5 TF)
- LSTM Fiyat Tahmini
- Performans Takibi

HATALAR YUTULMAZ - tüm hatalar loglanır ve bildirilir.
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

logger = logging.getLogger("V10_ENGINE")

# Configure logging EXPLICITLY - no silent errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


class V10Engine:
    """
    DEMIR AI v10 - ENHANCED Ana Motor
    
    Gelişmiş tahmin sistemi:
    - 14 teknik gösterge
    - 5 timeframe confluence
    - LSTM fiyat tahmini
    - Performans takibi
    
    HATA YUTULMAZ - her hata açıkça loglanır.
    """
    
    # Scan interval
    SCAN_INTERVAL = 30  # saniye
    
    # Spam prevention - same coin wait time
    SIGNAL_COOLDOWN = 30 * 60  # 30 dakika
    
    # Performance report interval
    PERFORMANCE_REPORT_INTERVAL = 4 * 60 * 60  # 4 saat
    
    def __init__(self):
        self.data_hub = get_data_hub()
        self.predictor = get_enhanced_predictor()  # Enhanced predictor with all modules
        self.notifier = get_notifier()
        self.performance_tracker = get_performance_tracker()  # Track signal accuracy
        self.lstm_predictor = get_lstm_predictor()  # Price prediction
        
        self._last_signal_time: Dict[str, datetime] = {}
        self._last_performance_report = datetime.now()
        self._running = False
        self._cycle_count = 0
        self._signal_count = 0
        self._error_count = 0
        
        logger.info("🚀 DEMIR AI v10 ENHANCED Engine initialized")
        logger.info("📊 Modules: Advanced Indicators + MTF + LSTM + Performance Tracking")
    
    async def start(self):
        """Ana döngüyü başlat"""
        self._running = True
        
        # Startup mesajı
        self.notifier.send_startup_message()
        logger.info("📡 V10 Engine started - scanning markets...")
        
        while self._running:
            try:
                await self._scan_cycle()
            except Exception as e:
                logger.error(f"❌ SCAN CYCLE ERROR: {e}")
                self._error_count += 1
                # Kritik hata - kullanıcıya bildir
                if self._error_count >= 3:
                    self.notifier.send_error_alert(f"Tekrarlayan hata: {e}")
            
            await asyncio.sleep(self.SCAN_INTERVAL)
    
    async def stop(self):
        """Motoru durdur"""
        self._running = False
        await self.data_hub.close()
        logger.info("🛑 V10 Engine stopped")
    
    async def _scan_cycle(self):
        """Tek bir tarama döngüsü"""
        self._cycle_count += 1
        cycle_start = datetime.now()
        
        logger.info(f"━━━ SCAN CYCLE #{self._cycle_count} ━━━")
        
        # 1. TÜM COİNLER İÇİN VERİ ÇEK
        snapshots = await self.data_hub.get_all_snapshots()
        
        # 2. HER COİN İÇİN SİNYAL ÜRET
        for symbol, snapshot in snapshots.items():
            try:
                await self._process_coin(symbol, snapshot)
            except Exception as e:
                logger.error(f"❌ {symbol} processing error: {e}")
        
        # 3. TEKNİK ANALİZ BİLDİRİMİ (15 dakikada bir)
        if not hasattr(self, '_last_ta_time'):
            self._last_ta_time = datetime.now() - timedelta(minutes=20)
        
        if (datetime.now() - self._last_ta_time).total_seconds() >= 15 * 60:
            try:
                logger.info("📈 Sending Technical Analysis...")
                self.notifier.send_technical_analysis(snapshots)
                self._last_ta_time = datetime.now()
            except Exception as e:
                logger.error(f"❌ Technical Analysis error: {e}")
        
        # 4. PERFORMANS KONTROLÜ (4 saatte bir)
        if (datetime.now() - self._last_performance_report).total_seconds() >= self.PERFORMANCE_REPORT_INTERVAL:
            try:
                logger.info("📊 Checking signal outcomes and sending performance report...")
                await self.performance_tracker.check_outcomes()
                report_msg = self.performance_tracker.format_report_message()
                self.notifier._send_message(report_msg)
                self._last_performance_report = datetime.now()
            except Exception as e:
                logger.error(f"❌ Performance report error: {e}")
        
        # 5. Döngü istatistikleri
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        stats = self.data_hub.get_stats()
        
        logger.info(
            f"✅ Cycle #{self._cycle_count} complete: {cycle_time:.1f}s | "
            f"Signals: {self._signal_count} | Errors: {self._error_count} | "
            f"Data success rate: {stats['success_rate']:.0f}%"
        )
    
    async def _process_coin(self, symbol: str, snapshot: MarketSnapshot):
        """Tek coin için sinyal kontrolü"""
        
        # 1. Veri kalitesi kontrolü
        if not snapshot.is_valid:
            if snapshot.errors:
                logger.warning(f"⚠️ {symbol}: Data errors - {snapshot.errors[:2]}")
                # Belirli aralıklarla kullanıcıya bildir
                if self._cycle_count % 10 == 0:
                    self.notifier.send_data_quality_alert(symbol, snapshot.errors)
            return
        
        # 2. Cooldown kontrolü (spam önleme)
        if self._is_on_cooldown(symbol):
            return
        
        # 3. Sinyal üret (async - enhanced predictor)
        signal = await self.predictor.generate_signal_async(snapshot)
        
        # 4. Geçerli sinyal varsa gönder
        if signal.is_valid:
            logger.info(f"🎯 VALID SIGNAL: {symbol} {signal.signal_type.value} %{signal.confidence:.0f}")
            
            success = self.notifier.send_trading_signal(signal)
            
            if success:
                self._last_signal_time[symbol] = datetime.now()
                self._signal_count += 1
                
                # Performance tracking - kaydet
                try:
                    self.performance_tracker.record_signal(signal)
                    logger.info(f"📊 Signal recorded for accuracy tracking")
                except Exception as e:
                    logger.warning(f"Performance tracking error: {e}")
        else:
            # Debug: neden sinyal üretilmedi
            if signal.warnings:
                logger.debug(f"⏸️ {symbol}: {signal.warnings[0]}")
    
    def _is_on_cooldown(self, symbol: str) -> bool:
        """Sinyal cooldown kontrolü"""
        if symbol not in self._last_signal_time:
            return False
        
        elapsed = (datetime.now() - self._last_signal_time[symbol]).total_seconds()
        return elapsed < self.SIGNAL_COOLDOWN
    
    async def scan_once(self) -> Dict[str, dict]:
        """
        Tek seferlik tarama (test için).
        Returns dict of signals for each coin.
        """
        results = {}
        snapshots = await self.data_hub.get_all_snapshots()
        
        for symbol, snapshot in snapshots.items():
            if snapshot.is_valid:
                signal = self.predictor.generate_signal(snapshot)
                results[symbol] = {
                    'signal': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'entry': f"${signal.entry_low:,.0f} - ${signal.entry_high:,.0f}",
                    'tp1': f"${signal.tp1:,.0f}",
                    'sl': f"${signal.sl:,.0f}",
                    'reasons': signal.reasons,
                    'is_valid': signal.is_valid
                }
            else:
                results[symbol] = {
                    'signal': 'ERROR',
                    'errors': snapshot.errors
                }
        
        return results


# Singleton
_engine: Optional[V10Engine] = None

def get_v10_engine() -> V10Engine:
    global _engine
    if _engine is None:
        _engine = V10Engine()
    return _engine


async def run_v10():
    """V10 Engine'i başlat (entry point)"""
    engine = get_v10_engine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


# Direct run
if __name__ == "__main__":
    asyncio.run(run_v10())
