# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - MAIN ENGINE
===========================
Ana döngü - veri topla, tahmin üret, sinyal gönder.

ÇALIŞMA MANTIĞI:
1. Her 30 saniyede tüm coinleri tara
2. Pattern/trend değişiklikleri tespit et
3. Yeterli güven (>60%) ve potansiyel (>$500) varsa sinyal gönder
4. Spam önleme: aynı coin için 30 dk bekleme
5. HATALAR YUTULMAZ - tüm hatalar loglanır ve bildirilir
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.v10.data_hub import get_data_hub, MarketSnapshot
from src.v10.predictor import get_predictor, SignalType
from src.v10.smart_notifier import get_notifier

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
    DEMIR AI v10 - Ana Motor
    
    Prediktif trading sinyalleri üretir.
    HATA YUTULMAZ - her hata açıkça loglanır.
    """
    
    # Scan interval
    SCAN_INTERVAL = 30  # saniye
    
    # Spam prevention - same coin wait time
    SIGNAL_COOLDOWN = 30 * 60  # 30 dakika
    
    def __init__(self):
        self.data_hub = get_data_hub()
        self.predictor = get_predictor()
        self.notifier = get_notifier()
        
        self._last_signal_time: Dict[str, datetime] = {}
        self._running = False
        self._cycle_count = 0
        self._signal_count = 0
        self._error_count = 0
        
        logger.info("🚀 DEMIR AI v10 Engine initialized")
    
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
        
        # 4. Döngü istatistikleri
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
        
        # 3. Sinyal üret
        signal = self.predictor.generate_signal(snapshot)
        
        # 4. Geçerli sinyal varsa gönder
        if signal.is_valid:
            logger.info(f"🎯 VALID SIGNAL: {symbol} {signal.signal_type.value} %{signal.confidence:.0f}")
            
            success = self.notifier.send_trading_signal(signal)
            
            if success:
                self._last_signal_time[symbol] = datetime.now()
                self._signal_count += 1
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
