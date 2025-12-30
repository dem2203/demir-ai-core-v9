# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - BACKTEST ENGINE
==============================
Geçmiş verilerle sinyal stratejisini test et.

ÖZELLİKLER:
- Son 30 gün simülasyon
- Faktör bazlı performans analizi
- Win rate hesaplama
- Drawdown analizi
"""
import logging
import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("BACKTEST_ENGINE")


@dataclass
class BacktestTrade:
    """Tek bir backtest trade'i"""
    symbol: str
    direction: str  # LONG, SHORT
    entry_time: datetime
    entry_price: float
    exit_time: datetime = None
    exit_price: float = 0
    tp_price: float = 0
    sl_price: float = 0
    pnl_pct: float = 0
    result: str = ""  # WIN, LOSS, OPEN
    exit_reason: str = ""  # TP, SL, TIME
    factors: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Backtest sonuçları"""
    symbol: str
    period_days: int
    total_signals: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    best_factors: List[Tuple[str, float]]  # (factor, win_rate)
    worst_factors: List[Tuple[str, float]]
    trades: List[BacktestTrade] = field(default_factory=list)


class BacktestEngine:
    """
    Backtest Engine
    
    Geçmiş verileri kullanarak sinyal stratejisini test eder.
    """
    
    FUTURES_BASE = "https://fapi.binance.com"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def run_backtest(self, symbol: str = "BTCUSDT", days: int = 30) -> BacktestResult:
        """
        Backtest çalıştır.
        
        Args:
            symbol: Test edilecek coin
            days: Kaç günlük veri
            
        Returns:
            BacktestResult
        """
        logger.info(f"🔄 Running backtest for {symbol} ({days} days)...")
        
        # 1. Tarihi veri çek
        klines = await self._fetch_historical_klines(symbol, days)
        if len(klines) < 100:
            logger.error("Insufficient data for backtest")
            return self._empty_result(symbol, days)
        
        logger.info(f"📊 Fetched {len(klines)} candles")
        
        # 2. Her mum için sinyal simüle et
        trades = []
        factor_performance = {}  # factor -> [wins, total]
        
        # Her 4 saatte bir sinyal kontrolü (15dk yerine)
        step = 4  # 4 saatlik mumlar
        
        for i in range(50, len(klines) - 20, step):
            # O andaki veri dilimi
            window = klines[max(0, i-50):i+1]
            
            # Sinyal simülasyonu
            signal = self._simulate_signal(window)
            
            if signal and signal['direction'] in ['LONG', 'SHORT']:
                entry_price = float(klines[i][4])  # Close
                entry_time = datetime.fromtimestamp(klines[i][0] / 1000)
                
                # TP/SL hesapla
                atr = self._calculate_atr(window)
                if signal['direction'] == 'LONG':
                    tp = entry_price * 1.03  # %3 TP
                    sl = entry_price * 0.985  # %1.5 SL
                else:
                    tp = entry_price * 0.97
                    sl = entry_price * 1.015
                
                # Trade sonucunu hesapla
                trade = BacktestTrade(
                    symbol=symbol,
                    direction=signal['direction'],
                    entry_time=entry_time,
                    entry_price=entry_price,
                    tp_price=tp,
                    sl_price=sl,
                    factors=signal.get('factors', [])
                )
                
                # İleriye bak - TP veya SL vuruldu mu?
                for j in range(i+1, min(i+20, len(klines))):  # Max 20 mum (80 saat)
                    high = float(klines[j][2])
                    low = float(klines[j][3])
                    close = float(klines[j][4])
                    
                    if signal['direction'] == 'LONG':
                        if high >= tp:
                            trade.exit_price = tp
                            trade.exit_time = datetime.fromtimestamp(klines[j][0] / 1000)
                            trade.result = "WIN"
                            trade.exit_reason = "TP"
                            trade.pnl_pct = (tp - entry_price) / entry_price * 100
                            break
                        elif low <= sl:
                            trade.exit_price = sl
                            trade.exit_time = datetime.fromtimestamp(klines[j][0] / 1000)
                            trade.result = "LOSS"
                            trade.exit_reason = "SL"
                            trade.pnl_pct = (sl - entry_price) / entry_price * 100
                            break
                    else:  # SHORT
                        if low <= tp:
                            trade.exit_price = tp
                            trade.exit_time = datetime.fromtimestamp(klines[j][0] / 1000)
                            trade.result = "WIN"
                            trade.exit_reason = "TP"
                            trade.pnl_pct = (entry_price - tp) / entry_price * 100
                            break
                        elif high >= sl:
                            trade.exit_price = sl
                            trade.exit_time = datetime.fromtimestamp(klines[j][0] / 1000)
                            trade.result = "LOSS"
                            trade.exit_reason = "SL"
                            trade.pnl_pct = (entry_price - sl) / entry_price * 100
                            break
                
                # Timeout - poz kapanmadı
                if trade.result == "":
                    trade.exit_price = float(klines[min(i+20, len(klines)-1)][4])
                    trade.exit_time = datetime.fromtimestamp(klines[min(i+20, len(klines)-1)][0] / 1000)
                    trade.result = "WIN" if trade.pnl_pct > 0 else "LOSS"
                    trade.exit_reason = "TIME"
                    if signal['direction'] == 'LONG':
                        trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
                    else:
                        trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100
                
                trades.append(trade)
                
                # Faktör performansı güncelle
                for factor in trade.factors:
                    if factor not in factor_performance:
                        factor_performance[factor] = [0, 0]
                    factor_performance[factor][1] += 1
                    if trade.result == "WIN":
                        factor_performance[factor][0] += 1
        
        # 3. Sonuçları hesapla
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        total = len(trades)
        
        win_rate = (wins / total * 100) if total > 0 else 0
        
        total_pnl = sum(t.pnl_pct for t in trades)
        avg_win = sum(t.pnl_pct for t in trades if t.result == "WIN") / wins if wins > 0 else 0
        avg_loss = sum(t.pnl_pct for t in trades if t.result == "LOSS") / losses if losses > 0 else 0
        
        gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown hesapla
        max_drawdown = self._calculate_max_drawdown(trades)
        
        # En iyi/kötü faktörler
        factor_win_rates = {
            f: (p[0] / p[1] * 100) if p[1] > 0 else 0 
            for f, p in factor_performance.items()
        }
        
        best_factors = sorted(factor_win_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        worst_factors = sorted(factor_win_rates.items(), key=lambda x: x[1])[:5]
        
        result = BacktestResult(
            symbol=symbol,
            period_days=days,
            total_signals=total,
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            best_factors=best_factors,
            worst_factors=worst_factors,
            trades=trades
        )
        
        logger.info(f"✅ Backtest complete: {total} trades, Win Rate: {win_rate:.1f}%")
        
        return result
    
    def _simulate_signal(self, klines: List) -> Optional[Dict]:
        """Sinyal simüle et (basitleştirilmiş)"""
        if len(klines) < 30:
            return None
        
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        
        factors = []
        bullish_score = 0
        bearish_score = 0
        
        # RSI
        rsi = self._calculate_rsi(closes)
        if rsi < 30:
            factors.append("RSI Oversold")
            bullish_score += 1
        elif rsi > 70:
            factors.append("RSI Overbought")
            bearish_score += 1
        
        # MACD
        macd_hist = self._calculate_macd_histogram(closes)
        if macd_hist > 0:
            factors.append("MACD Bullish")
            bullish_score += 1
        elif macd_hist < 0:
            factors.append("MACD Bearish")
            bearish_score += 1
        
        # EMA Cross
        ema20 = self._calculate_ema(closes, 20)
        ema50 = self._calculate_ema(closes, 50)
        if ema20 > ema50:
            factors.append("EMA Bullish")
            bullish_score += 1
        else:
            factors.append("EMA Bearish")
            bearish_score += 1
        
        # Momentum
        momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
        if momentum > 1:
            factors.append("Momentum Up")
            bullish_score += 1
        elif momentum < -1:
            factors.append("Momentum Down")
            bearish_score += 1
        
        # Volume
        volumes = [float(k[5]) for k in klines]
        avg_vol = sum(volumes[-14:]) / 14
        if volumes[-1] > avg_vol * 1.5:
            factors.append("Volume Spike")
            if closes[-1] > closes[-2]:
                bullish_score += 0.5
            else:
                bearish_score += 0.5
        
        # Karar
        if bullish_score >= 3:
            return {'direction': 'LONG', 'factors': factors}
        elif bearish_score >= 3:
            return {'direction': 'SHORT', 'factors': factors}
        
        return None
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd_histogram(self, prices: List[float]) -> float:
        if len(prices) < 26:
            return 0
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        return ema12 - ema26
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1]
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_atr(self, klines: List, period: int = 14) -> float:
        if len(klines) < period + 1:
            return 0
        
        trs = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i-1][4])
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return sum(trs[-period:]) / period
    
    def _calculate_max_drawdown(self, trades: List[BacktestTrade]) -> float:
        if not trades:
            return 0
        
        equity = 100
        peak = 100
        max_dd = 0
        
        for trade in trades:
            equity += trade.pnl_pct
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    async def _fetch_historical_klines(self, symbol: str, days: int) -> List:
        """Binance'dan tarihi veri çek"""
        session = await self._get_session()
        
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        limit = 1000
        interval = "4h"  # 4 saatlik mumlar
        
        # Kaç çağrı gerekli
        candles_needed = days * 6  # 6 x 4h = 24h
        calls_needed = (candles_needed // limit) + 1
        
        for _ in range(calls_needed):
            url = f"{self.FUTURES_BASE}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}&endTime={end_time}"
            
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        break
                    klines = await resp.json()
                    
                    if not klines:
                        break
                    
                    all_klines = klines + all_klines
                    end_time = int(klines[0][0]) - 1
                    
                    if len(all_klines) >= candles_needed:
                        break
            except Exception as e:
                logger.error(f"Klines fetch error: {e}")
                break
        
        return all_klines
    
    def _empty_result(self, symbol: str, days: int) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            period_days=days,
            total_signals=0,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            total_pnl=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_drawdown=0,
            best_factors=[],
            worst_factors=[]
        )
    
    def format_report(self, result: BacktestResult) -> str:
        """Telegram formatında backtest raporu"""
        
        best_factors_text = ""
        for i, (factor, wr) in enumerate(result.best_factors[:3], 1):
            best_factors_text += f"  {i}. {factor}: %{wr:.0f}\n"
        
        worst_factors_text = ""
        for i, (factor, wr) in enumerate(result.worst_factors[:3], 1):
            worst_factors_text += f"  {i}. {factor}: %{wr:.0f}\n"
        
        return f"""📊 *BACKTEST RAPORU*
━━━━━━━━━━━━━━━━━━━━━━━━

📅 *{result.symbol}* - Son {result.period_days} Gün

📈 *SONUÇLAR:*
  Toplam Trade: {result.total_trades}
  ✅ Kazanan: {result.wins}
  ❌ Kaybeden: {result.losses}
  📊 Win Rate: *%{result.win_rate:.1f}*

💰 *PERFORMANS:*
  Toplam PnL: {result.total_pnl:+.2f}%
  Ortalama Win: +{result.avg_win:.2f}%
  Ortalama Loss: {result.avg_loss:.2f}%
  Profit Factor: {result.profit_factor:.2f}
  Max Drawdown: %{result.max_drawdown:.1f}

🏆 *EN İYİ FAKTÖRLER:*
{best_factors_text}
📉 *EN KÖTÜ FAKTÖRLER:*
{worst_factors_text}
━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_engine: Optional[BacktestEngine] = None

def get_backtest_engine() -> BacktestEngine:
    global _engine
    if _engine is None:
        _engine = BacktestEngine()
    return _engine


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        engine = get_backtest_engine()
        
        print("Running backtest...")
        result = await engine.run_backtest("BTCUSDT", days=30)
        
        print(engine.format_report(result))
        
        await engine.close()
    
    asyncio.run(test())
