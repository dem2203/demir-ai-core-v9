# -*- coding: utf-8 -*-
"""
DEMIR AI - Paper Trader
Gerçek para olmadan simüle trading.

PHASE 111: Paper Trading
- Virtual balance tracking
- Signal execution simulation
- Real-time PnL tracking
- Performance reporting
"""
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import requests

logger = logging.getLogger("PAPER_TRADER")


@dataclass
class PaperPosition:
    """Simüle pozisyon."""
    id: str
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float  # USDT
    leverage: int
    tp_price: float
    sl_price: float
    entry_time: datetime
    status: str  # 'OPEN', 'TP_HIT', 'SL_HIT', 'CLOSED'
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    pnl: float = 0


class PaperTrader:
    """
    Paper Trading Sistemi
    
    Gerçek para olmadan sinyal testi.
    """
    
    STATE_FILE = "paper_trader_state.json"
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: List[PaperPosition] = []
        self.closed_positions: List[PaperPosition] = []
        self.position_counter = 0
        
        self._load_state()
        logger.info(f"✅ Paper Trader initialized - Balance: ${self.balance:,.2f}")
    
    def _load_state(self):
        """State yükle."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', self.initial_balance)
                    self.position_counter = data.get('position_counter', 0)
                    
                    # Load closed positions
                    for p in data.get('closed_positions', []):
                        self.closed_positions.append(PaperPosition(
                            id=p['id'],
                            symbol=p['symbol'],
                            direction=p['direction'],
                            entry_price=p['entry_price'],
                            size=p['size'],
                            leverage=p['leverage'],
                            tp_price=p['tp_price'],
                            sl_price=p['sl_price'],
                            entry_time=datetime.fromisoformat(p['entry_time']),
                            status=p['status'],
                            exit_price=p.get('exit_price', 0),
                            exit_time=datetime.fromisoformat(p['exit_time']) if p.get('exit_time') else None,
                            pnl=p.get('pnl', 0)
                        ))
                    
                    # Load open positions
                    for p in data.get('positions', []):
                        self.positions.append(PaperPosition(
                            id=p['id'],
                            symbol=p['symbol'],
                            direction=p['direction'],
                            entry_price=p['entry_price'],
                            size=p['size'],
                            leverage=p['leverage'],
                            tp_price=p['tp_price'],
                            sl_price=p['sl_price'],
                            entry_time=datetime.fromisoformat(p['entry_time']),
                            status=p['status']
                        ))
                        
        except Exception as e:
            logger.warning(f"Paper trader state load failed: {e}")
    
    def _save_state(self):
        """State kaydet."""
        try:
            def pos_to_dict(p: PaperPosition) -> dict:
                return {
                    'id': p.id,
                    'symbol': p.symbol,
                    'direction': p.direction,
                    'entry_price': p.entry_price,
                    'size': p.size,
                    'leverage': p.leverage,
                    'tp_price': p.tp_price,
                    'sl_price': p.sl_price,
                    'entry_time': p.entry_time.isoformat(),
                    'status': p.status,
                    'exit_price': p.exit_price,
                    'exit_time': p.exit_time.isoformat() if p.exit_time else None,
                    'pnl': p.pnl
                }
            
            with open(self.STATE_FILE, 'w') as f:
                json.dump({
                    'balance': self.balance,
                    'position_counter': self.position_counter,
                    'positions': [pos_to_dict(p) for p in self.positions],
                    'closed_positions': [pos_to_dict(p) for p in self.closed_positions[-100:]],
                    'last_update': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Paper trader state save failed: {e}")
    
    def open_position(self, symbol: str, direction: str, 
                     size_pct: float = 10, leverage: int = 10,
                     tp_pct: float = 3.5, sl_pct: float = 1.5) -> Optional[PaperPosition]:
        """
        Simüle pozisyon aç.
        
        Args:
            symbol: BTCUSDT
            direction: LONG veya SHORT
            size_pct: Bakiyenin yüzdesi
            leverage: Kaldıraç
            tp_pct: Kar al yüzdesi
            sl_pct: Zarar kes yüzdesi
        """
        # Aynı sembol için açık pozisyon var mı?
        open_for_symbol = [p for p in self.positions if p.symbol == symbol and p.status == 'OPEN']
        if open_for_symbol:
            logger.warning(f"Already have open position for {symbol}")
            return None
        
        # Fiyat al
        current_price = self._get_price(symbol)
        if current_price == 0:
            return None
        
        # Size hesapla
        size = self.balance * (size_pct / 100)
        
        # TP/SL
        if direction == 'LONG':
            tp_price = current_price * (1 + tp_pct / 100)
            sl_price = current_price * (1 - sl_pct / 100)
        else:
            tp_price = current_price * (1 - tp_pct / 100)
            sl_price = current_price * (1 + sl_pct / 100)
        
        self.position_counter += 1
        
        position = PaperPosition(
            id=f"PAPER-{self.position_counter}",
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            size=size,
            leverage=leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            entry_time=datetime.now(),
            status='OPEN'
        )
        
        self.positions.append(position)
        self._save_state()
        
        logger.info(f"📝 PAPER: Opened {direction} {symbol} @ ${current_price:,.2f} | Size: ${size:,.2f}")
        
        return position
    
    def check_positions(self) -> List[Dict]:
        """Tüm açık pozisyonları kontrol et ve TP/SL tetikle."""
        results = []
        
        for pos in self.positions:
            if pos.status != 'OPEN':
                continue
            
            current_price = self._get_price(pos.symbol)
            if current_price == 0:
                continue
            
            result = None
            
            # TP/SL kontrol
            if pos.direction == 'LONG':
                if current_price >= pos.tp_price:
                    result = self._close_position(pos, pos.tp_price, 'TP_HIT')
                elif current_price <= pos.sl_price:
                    result = self._close_position(pos, pos.sl_price, 'SL_HIT')
            else:  # SHORT
                if current_price <= pos.tp_price:
                    result = self._close_position(pos, pos.tp_price, 'TP_HIT')
                elif current_price >= pos.sl_price:
                    result = self._close_position(pos, pos.sl_price, 'SL_HIT')
            
            if result:
                results.append(result)
        
        return results
    
    def _close_position(self, pos: PaperPosition, exit_price: float, status: str) -> Dict:
        """Pozisyonu kapat."""
        pos.exit_price = exit_price
        pos.exit_time = datetime.now()
        pos.status = status
        
        # PnL hesapla
        if pos.direction == 'LONG':
            pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        else:
            pnl_pct = ((pos.entry_price - exit_price) / pos.entry_price) * 100
        
        pnl_usd = pos.size * (pnl_pct / 100) * pos.leverage
        pos.pnl = pnl_usd
        
        # Bakiye güncelle
        self.balance += pnl_usd
        
        # Pozisyonu taşı
        self.positions.remove(pos)
        self.closed_positions.append(pos)
        
        self._save_state()
        
        result = {
            'id': pos.id,
            'symbol': pos.symbol,
            'direction': pos.direction,
            'status': status,
            'entry': pos.entry_price,
            'exit': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'new_balance': self.balance
        }
        
        logger.info(f"📝 PAPER: Closed {pos.id} | {status} | PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")
        
        return result
    
    def _get_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                return float(resp.json()['price'])
        except:
            pass
        return 0
    
    def get_stats(self) -> Dict:
        """İstatistikleri al."""
        closed = self.closed_positions
        
        if not closed:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'balance': self.balance
            }
        
        wins = len([p for p in closed if p.status == 'TP_HIT'])
        losses = len([p for p in closed if p.status == 'SL_HIT'])
        total = len(closed)
        
        total_pnl = sum(p.pnl for p in closed)
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_pnl': total_pnl,
            'total_pnl_pct': ((self.balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total) * 100 if total > 0 else 0,
            'open_positions': len([p for p in self.positions if p.status == 'OPEN'])
        }
    
    def format_stats_for_telegram(self) -> str:
        """Telegram formatında istatistikler."""
        s = self.get_stats()
        
        pnl_emoji = "📈" if s['total_pnl'] >= 0 else "📉"
        win_emoji = "✅" if s['win_rate'] >= 55 else "⚠️" if s['win_rate'] >= 45 else "❌"
        
        msg = f"""
📝 PAPER TRADİNG DURUMU
━━━━━━━━━━━━━━━━━━━━━━
💰 Başlangıç: ${s['initial_balance']:,.2f}
💰 Şu An: ${s['current_balance']:,.2f}
{pnl_emoji} Toplam Kar: ${s['total_pnl']:+,.2f} ({s['total_pnl_pct']:+.1f}%)
━━━━━━━━━━━━━━━━━━━━━━
📊 Toplam İşlem: {s['total_trades']}
{win_emoji} Kazanma Oranı: %{s['win_rate']:.1f}
✅ Kazanç: {s['wins']} | ❌ Kayıp: {s['losses']}
📂 Açık Pozisyon: {s['open_positions']}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%d.%m.%Y %H:%M')}
""".strip()
        
        return msg
    
    def reset(self):
        """Paper trader'ı sıfırla."""
        self.balance = self.initial_balance
        self.positions = []
        self.closed_positions = []
        self.position_counter = 0
        self._save_state()
        logger.info("📝 Paper Trader reset")


# Global instance
_paper_trader = None

def get_paper_trader() -> PaperTrader:
    """Get or create paper trader instance."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader
