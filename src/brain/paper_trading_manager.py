# -*- coding: utf-8 -*-
"""
DEMIR AI - Paper Trading Manager
=================================
Premium sinyalleri paper trading ile takip eder.
SL/TP vurulduğunda bildirim gönderir.
Günlük detaylı rapor üretir.

Özellikler:
1. Sinyal -> Paper Trade Açma
2. TP/SL Vurulunca Bildirim
3. Günlük Performans Raporu
4. Güçlü/Zayıf Analizi
"""
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger("PAPER_TRADING_MANAGER")

# Storage
DATA_DIR = Path("src/brain/models/storage")
TRADES_FILE = DATA_DIR / "paper_trades.json"


@dataclass
class PaperTrade:
    """Paper trade kaydı"""
    id: str
    symbol: str
    direction: str  # LONG, SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: int
    
    # Factors at entry
    bullish_factors: List[str]
    bearish_factors: List[str]
    risk_factors: List[str]
    claude_analysis: str
    
    # Status
    status: str = "OPEN"  # OPEN, TP1_HIT, TP2_HIT, SL_HIT, EXPIRED
    opened_at: str = ""
    closed_at: str = ""
    exit_price: float = 0.0
    pnl_percent: float = 0.0
    duration_hours: float = 0.0
    
    # Learning
    what_worked: str = ""
    what_failed: str = ""


class PaperTradingManager:
    """
    Paper Trading Yöneticisi
    
    Premium sinyalleri paper trade olarak takip eder.
    TP/SL vurulduğunda analiz yapar ve bildirim gönderir.
    """
    
    TRADE_EXPIRY_HOURS = 72  # 3 gün
    
    def __init__(self):
        self.trades: List[PaperTrade] = []
        self.daily_stats = {}
        self._load_data()
        logger.info("📈 Paper Trading Manager initialized")
    
    def _load_data(self):
        """Kayıtlı trade'leri yükle"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if TRADES_FILE.exists():
                with open(TRADES_FILE, 'r') as f:
                    data = json.load(f)
                    self.trades = [PaperTrade(**t) for t in data.get('trades', [])]
                    self.daily_stats = data.get('daily_stats', {})
                logger.info(f"📂 Loaded {len(self.trades)} paper trades")
        except Exception as e:
            logger.warning(f"Trade load error: {e}")
            self.trades = []
    
    def _save_data(self):
        """Trade'leri kaydet"""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRADES_FILE, 'w') as f:
                json.dump({
                    'trades': [asdict(t) for t in self.trades],
                    'daily_stats': self.daily_stats
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Trade save error: {e}")
    
    def open_trade(self, signal) -> Optional[str]:
        """
        Premium sinyalden paper trade aç.
        
        Args:
            signal: PremiumSignal object
        
        Returns:
            trade_id veya None
        """
        if signal.direction == "BEKLE":
            return None
        
        trade_id = f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade = PaperTrade(
            id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            confidence=signal.confidence,
            bullish_factors=signal.bullish_factors,
            bearish_factors=signal.bearish_factors,
            risk_factors=signal.risk_factors,
            claude_analysis=signal.claude_analysis[:200],
            opened_at=datetime.now().isoformat()
        )
        
        self.trades.append(trade)
        self._save_data()
        
        logger.info(f"📈 Paper Trade Opened: {trade_id} ({signal.direction})")
        return trade_id
    
    def format_trade_open_message(self, signal) -> str:
        """Trade açılış bildirimi formatla"""
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        
        return f"""📈 *PAPER TRADE AÇILDI*
━━━━━━━━━━━━━━━━━━━━

{emoji} *{signal.symbol}* → *{signal.direction}*
💰 Entry: ${signal.entry_price:,.2f}
🎯 Güven: %{signal.confidence}

📍 *Seviyeler:*
  TP1: ${signal.take_profit_1:,.2f}
  TP2: ${signal.take_profit_2:,.2f}
  SL: ${signal.stop_loss:,.2f}

📊 Paper trading ile takip ediliyor...
⏰ {datetime.now().strftime('%H:%M:%S')}"""
    
    async def check_trades(self) -> List[Dict]:
        """
        Açık trade'leri kontrol et.
        TP/SL vuruldu mu bak.
        
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.trades:
            if trade.status != "OPEN":
                continue
            
            try:
                # Fiyat al
                current_price = await self._get_price(trade.symbol)
                if current_price == 0:
                    continue
                
                # TP/SL Kontrolü
                if trade.direction == "LONG":
                    # TP2 Hit
                    if current_price >= trade.take_profit_2:
                        pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
                        self._close_trade(trade, "TP2_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                    # TP1 Hit
                    elif current_price >= trade.take_profit_1:
                        pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
                        self._close_trade(trade, "TP1_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                    # SL Hit
                    elif current_price <= trade.stop_loss:
                        pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
                        self._close_trade(trade, "SL_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                
                elif trade.direction == "SHORT":
                    # TP2 Hit
                    if current_price <= trade.take_profit_2:
                        pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
                        self._close_trade(trade, "TP2_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                    # TP1 Hit
                    elif current_price <= trade.take_profit_1:
                        pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
                        self._close_trade(trade, "TP1_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                    # SL Hit
                    elif current_price >= trade.stop_loss:
                        pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
                        self._close_trade(trade, "SL_HIT", current_price, pnl)
                        closed_trades.append(self._analyze_trade(trade))
                
                # Expiry kontrolü
                opened = datetime.fromisoformat(trade.opened_at)
                age_hours = (datetime.now() - opened).total_seconds() / 3600
                
                if age_hours >= self.TRADE_EXPIRY_HOURS:
                    pnl = 0
                    if trade.direction == "LONG":
                        pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    else:
                        pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
                    self._close_trade(trade, "EXPIRED", current_price, pnl)
                    closed_trades.append(self._analyze_trade(trade))
                    
            except Exception as e:
                logger.debug(f"Trade check error for {trade.id}: {e}")
        
        if closed_trades:
            self._save_data()
            self._update_daily_stats()
        
        return closed_trades
    
    def _close_trade(self, trade: PaperTrade, status: str, exit_price: float, pnl: float):
        """Trade'i kapat"""
        trade.status = status
        trade.exit_price = exit_price
        trade.pnl_percent = pnl
        trade.closed_at = datetime.now().isoformat()
        
        opened = datetime.fromisoformat(trade.opened_at)
        trade.duration_hours = (datetime.now() - opened).total_seconds() / 3600
        
        logger.info(f"📊 Trade Closed: {trade.id} → {status} ({pnl:+.2f}%)")
        
        # 🧠 THINKING BRAIN FEEDBACK - Trade sonucundan öğren
        try:
            from src.brain.thinking_brain import get_thinking_brain
            thinking_brain = get_thinking_brain()
            
            was_correct = pnl > 0
            
            # Karar timestamp'ini bul (trade.opened_at kullan)
            thinking_brain.learn_from_outcome(
                decision_timestamp=trade.opened_at,
                pnl=pnl,
                was_correct=was_correct
            )
            
            logger.info(f"🧠 Thinking Brain learned from trade: {'✅' if was_correct else '❌'}")
        except Exception as e:
            logger.debug(f"Thinking Brain feedback error: {e}")
    
    def _analyze_trade(self, trade: PaperTrade) -> Dict:
        """Trade sonucunu analiz et - Ne işe yaradı, ne yaramadı?"""
        
        if trade.status in ["TP1_HIT", "TP2_HIT"]:
            # Başarılı trade - güçlü faktörleri belirle
            trade.what_worked = ", ".join(trade.bullish_factors[:2]) if trade.direction == "LONG" else ", ".join(trade.bearish_factors[:2])
            trade.what_failed = ""
        elif trade.status == "SL_HIT":
            # Başarısız trade - risk faktörlerini incele
            trade.what_worked = ""
            trade.what_failed = ", ".join(trade.risk_factors[:2]) if trade.risk_factors else "Risk faktörleri dikkate alınmadı"
        else:
            # Expired
            trade.what_worked = ""
            trade.what_failed = "Sinyal zamanında hareket etmedi"
        
        return {
            'trade': trade,
            'result': trade.status,
            'pnl': trade.pnl_percent,
            'what_worked': trade.what_worked,
            'what_failed': trade.what_failed
        }
    
    def format_trade_close_message(self, result: Dict) -> str:
        """Trade kapanış bildirimi formatla"""
        trade = result['trade']
        
        if trade.status in ["TP1_HIT", "TP2_HIT"]:
            emoji = "✅"
            result_text = f"TP{'1' if trade.status == 'TP1_HIT' else '2'} VURULDU!"
            color = "🟢"
        elif trade.status == "SL_HIT":
            emoji = "❌"
            result_text = "STOP LOSS VURULDU"
            color = "🔴"
        else:
            emoji = "⏰"
            result_text = "SÜRESİ DOLDU"
            color = "🟡"
        
        msg = f"""{emoji} *PAPER TRADE KAPANDI*
━━━━━━━━━━━━━━━━━━━━

{color} *{trade.symbol}* → {result_text}

📊 *Sonuç:*
  Entry: ${trade.entry_price:,.2f}
  Exit: ${trade.exit_price:,.2f}
  PnL: *{trade.pnl_percent:+.2f}%*
  Süre: {trade.duration_hours:.1f} saat

"""
        
        if trade.what_worked:
            msg += f"""💪 *Doğru Yapılan:*
  {trade.what_worked}

"""
        
        if trade.what_failed:
            msg += f"""📉 *Hata/Ders:*
  {trade.what_failed}

"""
        
        msg += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        return msg
    
    def _update_daily_stats(self):
        """Günlük istatistikleri güncelle"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'total_trades': 0,
                'tp_hits': 0,
                'sl_hits': 0,
                'expired': 0,
                'total_pnl': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'strong_factors': {},
                'weak_factors': {}
            }
        
        stats = self.daily_stats[today]
        
        # Bugünkü kapatılan trade'leri say
        for trade in self.trades:
            if trade.status == "OPEN":
                continue
            if not trade.closed_at:
                continue
            
            closed_date = trade.closed_at[:10]
            if closed_date != today:
                continue
            
            stats['total_trades'] += 1
            stats['total_pnl'] += trade.pnl_percent
            
            if trade.status in ["TP1_HIT", "TP2_HIT"]:
                stats['tp_hits'] += 1
                # Güçlü faktörleri kaydet
                for factor in trade.bullish_factors + trade.bearish_factors:
                    factor_key = factor[:30]
                    if factor_key not in stats['strong_factors']:
                        stats['strong_factors'][factor_key] = 0
                    stats['strong_factors'][factor_key] += 1
            elif trade.status == "SL_HIT":
                stats['sl_hits'] += 1
                # Zayıf faktörleri kaydet
                for factor in trade.risk_factors:
                    factor_key = factor[:30]
                    if factor_key not in stats['weak_factors']:
                        stats['weak_factors'][factor_key] = 0
                    stats['weak_factors'][factor_key] += 1
            else:
                stats['expired'] += 1
            
            # Best/Worst trade
            if stats['best_trade'] is None or trade.pnl_percent > stats['best_trade']['pnl']:
                stats['best_trade'] = {'symbol': trade.symbol, 'pnl': trade.pnl_percent}
            if stats['worst_trade'] is None or trade.pnl_percent < stats['worst_trade']['pnl']:
                stats['worst_trade'] = {'symbol': trade.symbol, 'pnl': trade.pnl_percent}
        
        self._save_data()
    
    async def _get_price(self, symbol: str) -> float:
        """Güncel fiyat al"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data.get('price', 0))
        except Exception as e:
            logger.debug(f"Price fetch error: {e}")
        return 0.0
    
    def get_open_trades_count(self) -> int:
        """Açık trade sayısı"""
        return len([t for t in self.trades if t.status == "OPEN"])
    
    def format_daily_report(self) -> str:
        """Günlük performans raporu"""
        today = datetime.now().strftime('%Y-%m-%d')
        stats = self.daily_stats.get(today, {})
        
        if not stats or stats.get('total_trades', 0) == 0:
            # Bugün trade yoksa son 7 günü göster
            return self._format_weekly_summary()
        
        total = stats.get('total_trades', 0)
        tp_hits = stats.get('tp_hits', 0)
        sl_hits = stats.get('sl_hits', 0)
        expired = stats.get('expired', 0)
        total_pnl = stats.get('total_pnl', 0)
        
        win_rate = (tp_hits / total * 100) if total > 0 else 0
        
        # Güçlü faktörler
        strong = stats.get('strong_factors', {})
        top_strong = sorted(strong.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Zayıf faktörler
        weak = stats.get('weak_factors', {})
        top_weak = sorted(weak.items(), key=lambda x: x[1], reverse=True)[:3]
        
        msg = f"""📊 *GÜNLÜK PAPER TRADING RAPORU*
━━━━━━━━━━━━━━━━━━━━━━━━

📅 *{datetime.now().strftime('%d.%m.%Y')}*

📈 *Özet:*
  Toplam Trade: {total}
  ✅ TP Vuruldu: {tp_hits}
  ❌ SL Vuruldu: {sl_hits}
  ⏰ Süresi Doldu: {expired}

📊 *Performans:*
  Win Rate: *%{win_rate:.1f}*
  Toplam PnL: *{total_pnl:+.2f}%*
"""
        
        if stats.get('best_trade'):
            msg += f"  🏆 En İyi: {stats['best_trade']['symbol']} ({stats['best_trade']['pnl']:+.2f}%)\n"
        if stats.get('worst_trade'):
            msg += f"  📉 En Kötü: {stats['worst_trade']['symbol']} ({stats['worst_trade']['pnl']:+.2f}%)\n"
        
        if top_strong:
            msg += f"""
💪 *GÜÇLÜ FAKTÖRLER:*
"""
            for factor, count in top_strong:
                msg += f"  • {factor} ({count}x)\n"
        
        if top_weak:
            msg += f"""
📉 *ZAYIF/HATALI FAKTÖRLER:*
"""
            for factor, count in top_weak:
                msg += f"  • {factor} ({count}x)\n"
        
        # Öğrenme önerileri
        msg += f"""
💡 *ÖNERİLER:*
"""
        if win_rate < 50:
            msg += "  • Win rate düşük - Güven eşiğini yükselt\n"
        if sl_hits > tp_hits:
            msg += "  • SL fazla - Risk faktörlerine daha çok dikkat et\n"
        if expired > 0:
            msg += "  • Expired trade var - Sinyal zamanlama optimize edilmeli\n"
        if win_rate >= 60:
            msg += "  • 🎯 İyi performans! Devam et.\n"
        
        msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""
        
        return msg
    
    def _format_weekly_summary(self) -> str:
        """Haftalık özet (bugün trade yoksa)"""
        total_trades = 0
        total_tp = 0
        total_sl = 0
        total_pnl = 0.0
        
        for date, stats in self.daily_stats.items():
            total_trades += stats.get('total_trades', 0)
            total_tp += stats.get('tp_hits', 0)
            total_sl += stats.get('sl_hits', 0)
            total_pnl += stats.get('total_pnl', 0)
        
        win_rate = (total_tp / total_trades * 100) if total_trades > 0 else 0
        
        open_count = self.get_open_trades_count()
        
        return f"""📊 *PAPER TRADING DURUM RAPORU*
━━━━━━━━━━━━━━━━━━━━━━━━

📍 *Aktif Trade:* {open_count}

📈 *Genel İstatistik:*
  Toplam Trade: {total_trades}
  ✅ TP: {total_tp} | ❌ SL: {total_sl}
  Win Rate: *%{win_rate:.1f}*
  Toplam PnL: *{total_pnl:+.2f}%*

⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}"""


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_manager: Optional[PaperTradingManager] = None

def get_paper_trading_manager() -> PaperTradingManager:
    """Get or create manager instance."""
    global _manager
    if _manager is None:
        _manager = PaperTradingManager()
    return _manager
