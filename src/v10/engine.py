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
from src.v10.signal_history import record_early_signal

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
                    
                    # Dogrulama verileri topla
                    verification = await self._get_verification_data(symbol)
                    
                    msg = f"""{tag} EARLY SIGNAL: {symbol}

Sinyal: {action_tr} | Guven: {early_signal.confidence:.0f}%
R/R: {early_signal.risk_reward:.1f}x

Giris: ${early_signal.entry_zone[0]:,.0f} - ${early_signal.entry_zone[1]:,.0f}
SL: ${early_signal.stop_loss:,.0f}
TP: ${early_signal.take_profit:,.0f}

Oncu Gostergeler:
{early_signal.reasoning}

DOGRULAMA:
{verification}"""
                    
                    self.notifier._send_message(msg)
                    self._last_signal_time[symbol] = datetime.now()
                    self._signal_count += 1
                    
                    # Sinyal geçmişine kaydet
                    try:
                        record_early_signal(early_signal, symbol)
                    except Exception as e:
                        logger.warning(f"Signal history error: {e}")
                    
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
    
    async def _get_verification_data(self, symbol: str) -> str:
        """Sinyal icin dogrulama verileri topla."""
        try:
            import aiohttp
            
            lines = []
            warnings = []
            
            async with aiohttp.ClientSession() as session:
                # 1. Klines -> RSI, EMA
                url = f"https://fapi.binance.com/fapi/v1/klines"
                params = {"symbol": symbol, "interval": "1h", "limit": 50}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
                        closes = [float(k[4]) for k in klines]
                        current = closes[-1]
                        
                        # RSI
                        rsi = self._calc_rsi(closes)
                        if rsi:
                            status = "Dusuk" if rsi < 40 else "Yuksek" if rsi > 60 else "Notr"
                            lines.append(f"RSI: {rsi:.0f} ({status})")
                            if rsi > 70:
                                warnings.append("RSI overbought")
                            elif rsi < 30:
                                warnings.append("RSI oversold")
                        
                        # EMA Trend
                        ema21 = sum(closes[-21:]) / 21
                        trend = "Yukari" if current > ema21 else "Asagi"
                        lines.append(f"Trend: {trend} (vs EMA21)")
                
                # 2. Order Book
                url = f"https://fapi.binance.com/fapi/v1/depth"
                params = {"symbol": symbol, "limit": 20}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        ob = await resp.json()
                        bid = sum(float(b[1]) for b in ob['bids'])
                        ask = sum(float(a[1]) for a in ob['asks'])
                        imb = (bid - ask) / (bid + ask) * 100
                        status = "Alim agir" if imb > 10 else "Satim agir" if imb < -10 else "Dengeli"
                        lines.append(f"Order Book: {imb:+.0f}% ({status})")
                        if imb < -20:
                            warnings.append("Guclu satis baskisi")
                        elif imb > 20:
                            warnings.append("Guclu alim")
                
                # 3. Funding
                url = "https://fapi.binance.com/fapi/v1/fundingRate"
                params = {"symbol": symbol, "limit": 1}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            fr = float(data[-1]['fundingRate']) * 100
                            lines.append(f"Funding: {fr:.4f}%")
            
            result = "\n".join(f"  {l}" for l in lines)
            if warnings:
                result += "\n  [!] " + ", ".join(warnings)
            
            return result
            
        except Exception as e:
            return f"  Dogrulama hatasi: {e}"
    
    def _calc_rsi(self, closes, period=14):
        """RSI hesapla."""
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            c = closes[i] - closes[i-1]
            gains.append(c if c > 0 else 0)
            losses.append(abs(c) if c < 0 else 0)
        avg_g = sum(gains[-period:]) / period
        avg_l = sum(losses[-period:]) / period
        if avg_l == 0:
            return 100
        return 100 - (100 / (1 + avg_g / avg_l))
    
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
