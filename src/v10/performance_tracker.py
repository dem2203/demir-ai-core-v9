# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - PERFORMANCE TRACKER
===================================
Sinyal performansını izle ve accuracy hesapla.

ÖZELLİKLER:
1. Her sinyali kaydet (timestamp, entry, TP, SL)
2. Belirli aralıklarla outcome kontrol et
3. Win rate, accuracy, R/R hesapla
4. Telegram'a rapor gönder

KULLANIM:
    tracker = PerformanceTracker()
    tracker.record_signal(signal)
    await tracker.check_outcomes()
    report = tracker.get_report()
"""
import logging
import json
import os
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger("PERFORMANCE_TRACKER")


class SignalOutcome(Enum):
    """Sinyal sonucu"""
    PENDING = "PENDING"
    TP1_HIT = "TP1_HIT"
    TP2_HIT = "TP2_HIT"
    TP3_HIT = "TP3_HIT"
    SL_HIT = "SL_HIT"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class RecordedSignal:
    """Kaydedilen sinyal"""
    id: str
    symbol: str
    signal_type: str  # "LONG", "SHORT"
    entry_low: float
    entry_high: float
    tp1: float
    tp2: float
    tp3: float
    sl: float
    confidence: float
    timestamp: str
    outcome: str = "PENDING"
    outcome_price: float = 0
    outcome_time: str = ""
    pnl_pct: float = 0


@dataclass
class PerformanceReport:
    """Performans raporu"""
    total_signals: int
    completed_signals: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl: float
    best_trade: float
    worst_trade: float
    accuracy_by_coin: Dict[str, float]
    last_10_results: List[str]


class PerformanceTracker:
    """
    Sinyal performansını izle ve raporla.
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json"
    }
    
    # Check outcomes after these durations
    CHECK_INTERVALS = [1, 4, 12, 24]  # hours
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._signals: List[RecordedSignal] = []
        
        # Storage
        self._storage_path = "src/v10/performance"
        os.makedirs(self._storage_path, exist_ok=True)
        
        # Load existing signals
        self._load_signals()
        
        logger.info(f"📊 Performance Tracker initialized. {len(self._signals)} signals loaded.")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers=self.HEADERS
            )
        return self._session
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
    
    def record_signal(self, signal) -> str:
        """
        Yeni sinyal kaydet.
        
        Args:
            signal: TradingSignal from predictor
            
        Returns:
            Signal ID
        """
        signal_id = f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        recorded = RecordedSignal(
            id=signal_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            tp1=signal.tp1,
            tp2=signal.tp2,
            tp3=signal.tp3,
            sl=signal.sl,
            confidence=signal.confidence,
            timestamp=datetime.now().isoformat()
        )
        
        self._signals.append(recorded)
        self._save_signals()
        
        logger.info(f"📝 Signal recorded: {signal_id}")
        return signal_id
    
    async def check_outcomes(self) -> List[Dict]:
        """
        Bekleyen sinyallerin sonuçlarını kontrol et.
        """
        updates = []
        
        for signal in self._signals:
            if signal.outcome != "PENDING":
                continue
            
            # Check if enough time passed
            signal_time = datetime.fromisoformat(signal.timestamp)
            elapsed_hours = (datetime.now() - signal_time).total_seconds() / 3600
            
            # Minimum 1 saat bekle
            if elapsed_hours < 1:
                continue
            
            # Get current price
            try:
                current_price = await self._get_price(signal.symbol)
                if current_price <= 0:
                    continue
                
                # Check outcome
                outcome = self._check_signal_outcome(signal, current_price)
                
                if outcome != SignalOutcome.PENDING:
                    signal.outcome = outcome.value
                    signal.outcome_price = current_price
                    signal.outcome_time = datetime.now().isoformat()
                    
                    # Calculate PnL
                    entry_price = (signal.entry_low + signal.entry_high) / 2
                    
                    if signal.signal_type == "LONG":
                        signal.pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:  # SHORT
                        signal.pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    updates.append({
                        'id': signal.id,
                        'outcome': outcome.value,
                        'pnl': signal.pnl_pct
                    })
                    
                    logger.info(f"✅ Signal {signal.id}: {outcome.value} ({signal.pnl_pct:+.2f}%)")
                
                # Expire old signals (24h+)
                elif elapsed_hours >= 24:
                    signal.outcome = SignalOutcome.EXPIRED.value
                    signal.outcome_price = current_price
                    signal.outcome_time = datetime.now().isoformat()
                    
                    entry_price = (signal.entry_low + signal.entry_high) / 2
                    if signal.signal_type == "LONG":
                        signal.pnl_pct = (current_price - entry_price) / entry_price * 100
                    else:
                        signal.pnl_pct = (entry_price - current_price) / entry_price * 100
                    
                    updates.append({
                        'id': signal.id,
                        'outcome': 'EXPIRED',
                        'pnl': signal.pnl_pct
                    })
                    
            except Exception as e:
                logger.error(f"❌ Check outcome error {signal.id}: {e}")
        
        if updates:
            self._save_signals()
        
        return updates
    
    def _check_signal_outcome(self, signal: RecordedSignal, current_price: float) -> SignalOutcome:
        """
        Sinyalin sonucunu belirle.
        """
        if signal.signal_type == "LONG":
            if current_price >= signal.tp3:
                return SignalOutcome.TP3_HIT
            elif current_price >= signal.tp2:
                return SignalOutcome.TP2_HIT
            elif current_price >= signal.tp1:
                return SignalOutcome.TP1_HIT
            elif current_price <= signal.sl:
                return SignalOutcome.SL_HIT
                
        elif signal.signal_type == "SHORT":
            if current_price <= signal.tp3:
                return SignalOutcome.TP3_HIT
            elif current_price <= signal.tp2:
                return SignalOutcome.TP2_HIT
            elif current_price <= signal.tp1:
                return SignalOutcome.TP1_HIT
            elif current_price >= signal.sl:
                return SignalOutcome.SL_HIT
        
        return SignalOutcome.PENDING
    
    def get_report(self) -> PerformanceReport:
        """
        Performans raporu oluştur.
        """
        completed = [s for s in self._signals if s.outcome not in ["PENDING", "UNKNOWN"]]
        
        wins = [s for s in completed if s.outcome in ["TP1_HIT", "TP2_HIT", "TP3_HIT"]]
        losses = [s for s in completed if s.outcome == "SL_HIT"]
        
        if not completed:
            return PerformanceReport(
                total_signals=len(self._signals),
                completed_signals=0,
                wins=0,
                losses=0,
                win_rate=0,
                avg_pnl=0,
                best_trade=0,
                worst_trade=0,
                accuracy_by_coin={},
                last_10_results=[]
            )
        
        # Win rate
        win_rate = len(wins) / len(completed) * 100 if completed else 0
        
        # Average PnL
        pnls = [s.pnl_pct for s in completed if s.pnl_pct != 0]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        
        # Best/worst
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        
        # By coin
        accuracy_by_coin = {}
        for coin in set(s.symbol for s in completed):
            coin_signals = [s for s in completed if s.symbol == coin]
            coin_wins = [s for s in coin_signals if s.outcome in ["TP1_HIT", "TP2_HIT", "TP3_HIT"]]
            accuracy_by_coin[coin] = len(coin_wins) / len(coin_signals) * 100 if coin_signals else 0
        
        # Last 10
        last_10 = sorted(completed, key=lambda x: x.timestamp, reverse=True)[:10]
        last_10_results = [f"{s.symbol}: {s.outcome} ({s.pnl_pct:+.1f}%)" for s in last_10]
        
        return PerformanceReport(
            total_signals=len(self._signals),
            completed_signals=len(completed),
            wins=len(wins),
            losses=len(losses),
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            best_trade=best_trade,
            worst_trade=worst_trade,
            accuracy_by_coin=accuracy_by_coin,
            last_10_results=last_10_results
        )
    
    def format_report_message(self) -> str:
        """
        Telegram için rapor mesajı oluştur.
        """
        report = self.get_report()
        
        # If no signals in performance tracker, try signal_history
        if report.total_signals == 0:
            try:
                from src.v10.signal_history import get_statistics, get_recent_signals
                from src.brain.feedback_db import get_feedback_db
                
                # Get signal_history stats
                stats = get_statistics()
                
                # Get feedback_db stats (actual trade outcomes)
                feedback_db = get_feedback_db()
                feedback_stats = feedback_db.get_stats()
                
                lines = [
                    "📊 *PERFORMANS RAPORU*",
                    "━━━━━━━━━━━━━━━━━━━━━━━",
                    "",
                    f"📈 *SİNYAL İSTATİSTİKLERİ*",
                    f"Toplam Sinyal: {stats.get('total', 0)}",
                    f"Bekleyen: {stats.get('pending', 0)}",
                    f"Ort. Güven: {stats.get('avg_confidence', 0):.0f}%",
                    "",
                    f"💰 *PAPER TRADING*",
                    f"Toplam İşlem: {feedback_stats.get('total_trades', 0)}",
                    f"Win Rate: {feedback_stats.get('win_rate', 0) * 100:.1f}%",
                    f"Ort. PnL: ${feedback_stats.get('avg_pnl', 0):.2f}",
                    f"Toplam PnL: ${feedback_stats.get('total_pnl', 0):.2f}",
                    "",
                ]
                
                # Recent signals
                recent = get_recent_signals(5)
                if recent:
                    lines.append("━━━ *SON SİNYALLER* ━━━")
                    for sig in recent[:3]:
                        emoji = "🟢" if sig.get('action') == 'BUY' else "🔴" if sig.get('action') == 'SELL' else "⚪"
                        lines.append(f"{emoji} {sig.get('symbol')}: {sig.get('action')} ({sig.get('confidence', 0):.0f}%)")
                
                lines.extend([
                    "",
                    f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                    "📡 *DEMIR AI v10*"
                ])
                
                return "\n".join(lines)
            except Exception as e:
                logger.debug(f"Signal history stats error: {e}")
        
        # Original code for when we have performance tracker data
        lines = [
            "📊 *PERFORMANS RAPORU*",
            "━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"📈 *ÖZET*",
            f"Toplam Sinyal: {report.total_signals}",
            f"Tamamlanan: {report.completed_signals}",
            f"Kazanç: {report.wins} | Kayıp: {report.losses}",
            "",
            f"🎯 *Win Rate: {report.win_rate:.1f}%*",
            f"💰 Ort. PnL: {report.avg_pnl:+.2f}%",
            f"🚀 En İyi: {report.best_trade:+.2f}%",
            f"💀 En Kötü: {report.worst_trade:+.2f}%",
            "",
        ]
        
        # By coin
        if report.accuracy_by_coin:
            lines.append("━━━ *COİN BAZINDA* ━━━")
            for coin, acc in report.accuracy_by_coin.items():
                emoji = "🟢" if acc >= 60 else "🟡" if acc >= 40 else "🔴"
                lines.append(f"{emoji} {coin}: {acc:.0f}%")
            lines.append("")
        
        # Last 10
        if report.last_10_results:
            lines.append("━━━ *SON 10 SİNYAL* ━━━")
            for result in report.last_10_results[:5]:
                lines.append(result)
        
        lines.extend([
            "",
            f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "📡 *DEMIR AI v10*"
        ])
        
        return "\n".join(lines)
    
    async def _get_price(self, symbol: str) -> float:
        """Get current price from Binance Futures."""
        try:
            session = await self._get_session()
            url = f"{self.FUTURES_BASE}/fapi/v1/ticker/price?symbol={symbol}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return 0
                data = await resp.json()
                return float(data.get('price', 0))
        except Exception:
            return 0
    
    def _save_signals(self):
        """Sinyalleri dosyaya kaydet."""
        path = os.path.join(self._storage_path, "signals.json")
        data = [asdict(s) for s in self._signals[-500:]]  # Keep last 500
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_signals(self):
        """Sinyalleri dosyadan yükle."""
        path = os.path.join(self._storage_path, "signals.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                self._signals = [RecordedSignal(**d) for d in data]
            except Exception as e:
                logger.error(f"Failed to load signals: {e}")
                self._signals = []
    
    def get_stats(self) -> Dict:
        """Quick stats for monitoring."""
        report = self.get_report()
        return {
            'total': report.total_signals,
            'completed': report.completed_signals,
            'win_rate': report.win_rate,
            'avg_pnl': report.avg_pnl
        }


# Singleton
_tracker: Optional[PerformanceTracker] = None

def get_performance_tracker() -> PerformanceTracker:
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker
