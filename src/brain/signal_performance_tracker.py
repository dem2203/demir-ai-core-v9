# -*- coding: utf-8 -*-
"""
DEMIR AI - Signal Performance Tracker
Win Rate Takibi - Sinyal doğruluğunu ölç.

PHASE 91: Win Rate Tracking System
- Her sinyali kaydet
- TP1/TP2/SL durumunu takip et
- Win rate hesapla
- Günlük rapor gönder
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

logger = logging.getLogger("WIN_RATE_TRACKER")


class SignalPerformanceTracker:
    """
    Sinyal Performans Takip Sistemi
    
    Her sinyalin TP/SL durumunu takip eder ve win rate hesaplar.
    """
    
    SIGNALS_FILE = "active_signals.json"
    HISTORY_FILE = "signal_history.json"
    
    def __init__(self):
        self.active_signals: List[Dict] = []
        self.signal_history: List[Dict] = []
        self._load_signals()
        logger.info("✅ Signal Performance Tracker initialized")
    
    def _load_signals(self):
        """Kayıtlı sinyalleri yükle."""
        try:
            if os.path.exists(self.SIGNALS_FILE):
                with open(self.SIGNALS_FILE, 'r') as f:
                    self.active_signals = json.load(f)
            
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r') as f:
                    self.signal_history = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load signals: {e}")
    
    def _save_signals(self):
        """Sinyalleri kaydet."""
        try:
            with open(self.SIGNALS_FILE, 'w') as f:
                json.dump(self.active_signals, f, indent=2)
            
            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(self.signal_history[-100:], f, indent=2)  # Son 100 sinyal
        except Exception as e:
            logger.warning(f"Failed to save signals: {e}")
    
    def record_signal(self, signal: Dict) -> str:
        """
        Yeni sinyal kaydet.
        
        Args:
            signal: {
                'symbol': 'BTCUSDT',
                'direction': 'LONG',
                'entry': 89000,
                'tp1': 90780,
                'tp2': 92560,
                'sl': 87680,
                'confidence': 65
            }
        
        Returns:
            signal_id: Benzersiz sinyal ID'si
        """
        signal_id = f"{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tracked_signal = {
            'id': signal_id,
            'symbol': signal.get('symbol', 'BTCUSDT'),
            'direction': signal.get('direction', 'LONG'),
            'entry': signal.get('entry', 0),
            'tp1': signal.get('tp1', 0),
            'tp2': signal.get('tp2', 0),
            'sl': signal.get('sl', 0),
            'confidence': signal.get('confidence', 50),
            'created_at': datetime.now().isoformat(),
            'status': 'ACTIVE',  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT, EXPIRED
            'tp1_hit': False,
            'tp2_hit': False,
            'sl_hit': False,
            'closed_at': None,
            'profit_pct': 0
        }
        
        self.active_signals.append(tracked_signal)
        self._save_signals()
        
        logger.info(f"📝 Signal recorded: {signal_id}")
        return signal_id
    
    def check_signals(self) -> List[Dict]:
        """
        Aktif sinyallerin TP/SL durumunu kontrol et.
        
        Returns:
            Durum değişen sinyallerin listesi
        """
        updated_signals = []
        
        for signal in self.active_signals:
            if signal['status'] != 'ACTIVE':
                continue
            
            try:
                # Mevcut fiyatı al
                current_price = self._get_current_price(signal['symbol'])
                if current_price == 0:
                    continue
                
                direction = signal['direction']
                entry = signal['entry']
                tp1 = signal['tp1']
                tp2 = signal['tp2']
                sl = signal['sl']
                
                # LONG pozisyon kontrolü
                if direction == 'LONG':
                    if current_price <= sl:
                        signal['status'] = 'SL_HIT'
                        signal['sl_hit'] = True
                        signal['profit_pct'] = ((sl - entry) / entry) * 100
                        signal['closed_at'] = datetime.now().isoformat()
                        updated_signals.append(signal)
                    elif current_price >= tp2:
                        signal['status'] = 'TP2_HIT'
                        signal['tp1_hit'] = True
                        signal['tp2_hit'] = True
                        signal['profit_pct'] = ((tp2 - entry) / entry) * 100
                        signal['closed_at'] = datetime.now().isoformat()
                        updated_signals.append(signal)
                    elif current_price >= tp1 and not signal['tp1_hit']:
                        signal['tp1_hit'] = True
                        signal['status'] = 'TP1_HIT'
                        updated_signals.append(signal)
                
                # SHORT pozisyon kontrolü
                elif direction == 'SHORT':
                    if current_price >= sl:
                        signal['status'] = 'SL_HIT'
                        signal['sl_hit'] = True
                        signal['profit_pct'] = ((entry - sl) / entry) * 100
                        signal['closed_at'] = datetime.now().isoformat()
                        updated_signals.append(signal)
                    elif current_price <= tp2:
                        signal['status'] = 'TP2_HIT'
                        signal['tp1_hit'] = True
                        signal['tp2_hit'] = True
                        signal['profit_pct'] = ((entry - tp2) / entry) * 100
                        signal['closed_at'] = datetime.now().isoformat()
                        updated_signals.append(signal)
                    elif current_price <= tp1 and not signal['tp1_hit']:
                        signal['tp1_hit'] = True
                        signal['status'] = 'TP1_HIT'
                        updated_signals.append(signal)
                
                # 24 saat sonra expire
                created = datetime.fromisoformat(signal['created_at'])
                if datetime.now() - created > timedelta(hours=24):
                    if signal['status'] == 'ACTIVE':
                        signal['status'] = 'EXPIRED'
                        signal['profit_pct'] = ((current_price - entry) / entry) * 100 if direction == 'LONG' else ((entry - current_price) / entry) * 100
                        signal['closed_at'] = datetime.now().isoformat()
                        updated_signals.append(signal)
                
            except Exception as e:
                logger.warning(f"Signal check failed for {signal['id']}: {e}")
        
        # Kapatılan sinyalleri history'ye taşı
        for signal in updated_signals:
            if signal['status'] in ['TP2_HIT', 'SL_HIT', 'EXPIRED']:
                self.signal_history.append(signal)
                self.active_signals.remove(signal)
        
        self._save_signals()
        return updated_signals
    
    def _get_current_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                f"https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def get_win_rate(self, days: int = 7) -> Dict:
        """
        Win rate hesapla.
        
        Returns:
            {
                'total_signals': 15,
                'winners': 9,
                'losers': 4,
                'expired': 2,
                'win_rate': 69.2,
                'avg_profit_pct': 1.5,
                'best_signal': {...},
                'worst_signal': {...}
            }
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        recent = [s for s in self.signal_history 
                  if datetime.fromisoformat(s['created_at']) > cutoff]
        
        if not recent:
            return {
                'total_signals': 0,
                'winners': 0,
                'losers': 0,
                'expired': 0,
                'win_rate': 0,
                'avg_profit_pct': 0,
                'period_days': days
            }
        
        winners = [s for s in recent if s['status'] in ['TP1_HIT', 'TP2_HIT']]
        losers = [s for s in recent if s['status'] == 'SL_HIT']
        expired = [s for s in recent if s['status'] == 'EXPIRED']
        
        total = len(recent)
        win_count = len(winners)
        lose_count = len(losers)
        
        win_rate = (win_count / (win_count + lose_count) * 100) if (win_count + lose_count) > 0 else 0
        
        profits = [s['profit_pct'] for s in recent if s.get('profit_pct')]
        avg_profit = sum(profits) / len(profits) if profits else 0
        
        best = max(recent, key=lambda x: x.get('profit_pct', -100)) if recent else None
        worst = min(recent, key=lambda x: x.get('profit_pct', 100)) if recent else None
        
        return {
            'total_signals': total,
            'winners': win_count,
            'losers': lose_count,
            'expired': len(expired),
            'active': len(self.active_signals),
            'win_rate': round(win_rate, 1),
            'avg_profit_pct': round(avg_profit, 2),
            'best_signal': best,
            'worst_signal': worst,
            'period_days': days
        }
    
    def format_daily_report(self) -> str:
        """Günlük performans raporu formatla."""
        stats = self.get_win_rate(days=1)
        stats_7d = self.get_win_rate(days=7)
        
        if stats['total_signals'] == 0 and stats_7d['total_signals'] == 0:
            return ""
        
        # Emoji belirleme
        if stats['win_rate'] >= 70:
            emoji = "🏆"
        elif stats['win_rate'] >= 50:
            emoji = "📊"
        else:
            emoji = "⚠️"
        
        # Best/Worst
        best_text = ""
        if stats_7d.get('best_signal'):
            b = stats_7d['best_signal']
            best_text = f"🏆 En İyi: {b['symbol']} {b['direction']} +{b['profit_pct']:.1f}%"
        
        worst_text = ""
        if stats_7d.get('worst_signal'):
            w = stats_7d['worst_signal']
            worst_text = f"💔 En Kötü: {w['symbol']} {w['direction']} {w['profit_pct']:.1f}%"
        
        msg = f"""
{emoji} **GÜNLÜK PERFORMANS RAPORU**
━━━━━━━━━━━━━━━━━━━━━━
**Bugün:**
├── ✅ Kazanan: {stats['winners']}
├── ❌ Kaybeden: {stats['losers']}
├── ⏳ Devam Eden: {stats['active']}
└── 📈 Win Rate: **%{stats['win_rate']}**

**Son 7 Gün:**
├── Toplam: {stats_7d['total_signals']} sinyal
├── Win Rate: **%{stats_7d['win_rate']}**
└── Ort. Kâr: {stats_7d['avg_profit_pct']:+.2f}%
━━━━━━━━━━━━━━━━━━━━━━
{best_text}
{worst_text}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
🧠 35 modül analizi
""".strip()
        
        return msg
    
    def format_signal_result(self, signal: Dict) -> str:
        """Sinyal sonuç mesajı formatla."""
        status = signal['status']
        symbol = signal['symbol']
        direction = signal['direction']
        profit = signal.get('profit_pct', 0)
        
        if status == 'TP1_HIT':
            emoji = "🎯"
            status_text = "TP1 VURULDU!"
        elif status == 'TP2_HIT':
            emoji = "🏆"
            status_text = "TP2 VURULDU!"
        elif status == 'SL_HIT':
            emoji = "🛑"
            status_text = "SL VURULDU"
        else:
            emoji = "⏱️"
            status_text = "EXPIRED"
        
        profit_emoji = "📈" if profit > 0 else "📉"
        
        msg = f"""
{emoji} **SİNYAL SONUCU**
━━━━━━━━━━━━━━━━━━━━━━
📍 {symbol} {direction}
📊 Durum: **{status_text}**
{profit_emoji} Sonuç: **{profit:+.2f}%**
━━━━━━━━━━━━━━━━━━━━━━
💰 Giriş: ${signal['entry']:,.2f}
🎯 TP1: ${signal['tp1']:,.2f} {'✅' if signal['tp1_hit'] else ''}
🎯 TP2: ${signal['tp2']:,.2f} {'✅' if signal['tp2_hit'] else ''}
🛡️ SL: ${signal['sl']:,.2f} {'❌' if signal['sl_hit'] else ''}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg


# Global instance
_tracker = None

def get_tracker() -> SignalPerformanceTracker:
    """Get or create tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = SignalPerformanceTracker()
    return _tracker
