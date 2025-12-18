import json
import os
import logging
from datetime import datetime

logger = logging.getLogger("PAPER_TRADER")

class PaperTrader:
    """
    DEMIR AI - PRO EXECUTION ENGINE (RISK MANAGED)
    
    Özellikler:
    1. Volatilite Bazlı Pozisyonlama (Risk Parity).
    2. Sabit Risk Kuralı (%1 Risk per Trade).
    3. Sanal Cüzdan Yönetimi.
    """
    
    DB_FILE = "portfolio.json"
    INITIAL_BALANCE = 10000.0 
    RISK_PER_TRADE = 0.01 # Her işlemde kasanın %1'ini riske atar

    def __init__(self):
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self):
        if os.path.exists(self.DB_FILE):
            try:
                with open(self.DB_FILE, 'r') as f:
                    return json.load(f)
            except: pass
        
        return {
            "balance": self.INITIAL_BALANCE,
            "equity": self.INITIAL_BALANCE,
            "positions": {}, 
            "history": []
        }

    def _save_portfolio(self):
        with open(self.DB_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=4)

    def calculate_position_size(self, entry_price, stop_loss_price, balance, risk_pct=0.01):
        """
        PROFESYONEL LOT HESABI
        Formül: (Kasa * Risk%) / (Giriş - Stop)
        Amaç: Stop olursam kasamın sadece Risk% kadarı gitsin.
        """
        if entry_price <= 0 or stop_loss_price <= 0: return 0.0
        
        # Hisse başına risk miktarı (Fiyat farkı)
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0: return 0.0
        
        # Riske atılacak toplam dolar (Örn: 10.000$ * 0.02 = 200$)
        money_at_risk = balance * risk_pct
        
        # Alınacak adet (Size)
        quantity = money_at_risk / risk_per_share
        
        # Toplam maliyet (USDT)
        trade_cost = quantity * entry_price
        
        # GÜVENLİK: Asla kasanın %95'inden fazlasını tek işleme bağlama
        if trade_cost > balance * 0.95:
            quantity = (balance * 0.95) / entry_price
            
        return quantity

    def execute_trade(self, signal):
        symbol = signal['symbol']
        price = float(signal['entry_price'])
        side = signal['side'] 
        
        # --- POZİSYON KAPATMA (SELL) ---
        if side == "SELL":
            if symbol in self.portfolio['positions']:
                pos = self.portfolio['positions'][symbol]
                amount = pos['amount']
                entry_price = pos['entry_price']
                
                # Kar/Zarar Hesapla
                pnl = (price - entry_price) * amount
                
                # Bakiyeye ekle (Ana para + Kar)
                self.portfolio['balance'] += (amount * entry_price) + pnl
                
                # Geçmişe kaydet
                self.portfolio['history'].append({
                    "symbol": symbol,
                    "action": "SELL",
                    "entry": entry_price,
                    "exit": price,
                    "amount": amount,
                    "pnl": pnl,
                    "time": datetime.now().isoformat()
                })
                
                del self.portfolio['positions'][symbol]
                self._save_portfolio()
                logger.info(f"🧻 SOLD {symbol} at {price}. PnL: ${pnl:.2f}")
                return True

        # --- POZİSYON AÇMA (BUY) ---
        elif side == "BUY":
            # Zaten açık pozisyon varsa ekleme yapma
            if symbol in self.portfolio['positions']: 
                return False
            
            # Sinyalden gelen Stop Loss'u kullan
            sl_price = float(signal.get('sl_price', price * 0.98))
            
            # Profesyonel Büyüklük Hesabı
            # Kelly Size varsa onu kullan (Yüzde olarak gelir, örn 2.5 -> 0.025)
            risk_pct = signal.get('kelly_size', 1.0) / 100.0
            if risk_pct <= 0: risk_pct = 0.01 # Güvenlik
            
            quantity = self.calculate_position_size(price, sl_price, self.portfolio['balance'], risk_pct)
            trade_cost = quantity * price
            
            # Min işlem limiti (10$)
            if trade_cost < 10: 
                logger.warning(f"Calculated trade size too small (${trade_cost:.2f}), skipping.")
                return False
            
            # Bakiyeden düş
            self.portfolio['balance'] -= trade_cost
            
            # Pozisyonu aç
            self.portfolio['positions'][symbol] = {
                "entry_price": price,
                "sl_price": sl_price,
                "amount": quantity,
                "cost": trade_cost,
                "time": datetime.now().isoformat()
            }
            
            self._save_portfolio()
            logger.info(f"🧻 BOUGHT {symbol} at {price}. Size: ${trade_cost:.2f} (Risk Adjusted)")
            return True
            
        return False

    def get_portfolio_status(self, current_prices={}):
        """
        Dashboard için canlı durum raporu.
        """
        equity = self.portfolio['balance']
        open_positions = []
        
        for symbol, pos in self.portfolio['positions'].items():
            # Canlı fiyat varsa kullan, yoksa giriş fiyatını al
            current_price = current_prices.get(symbol, pos['entry_price'])
            
            market_value = pos['amount'] * current_price
            unrealized_pnl = market_value - pos['cost']
            
            # 0'a bölme hatasını önle
            if pos['cost'] > 0:
                pnl_pct = (unrealized_pnl / pos['cost']) * 100
            else:
                pnl_pct = 0.0
            
            equity += market_value
            
            open_positions.append({
                "symbol": symbol,
                "entry": pos['entry_price'],
                "current": current_price,
                "amount": pos['amount'],
                "pnl": unrealized_pnl,
                "pnl_pct": pnl_pct
            })
            
        self.portfolio['equity'] = equity
        self._save_portfolio() # Her güncellemede kaydet ki veri kaybolmasın
        
        return {
            "balance": self.portfolio['balance'],
            "equity": equity,
            "positions": open_positions,
            "history": self.portfolio['history'][-10:] # Son 10 işlem
        }
    
    def get_stats(self):
        """İstatistikleri al."""
        history = self.portfolio.get('history', [])
        
        if not history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'balance': self.portfolio.get('balance', self.INITIAL_BALANCE)
            }
        
        wins = len([t for t in history if t.get('pnl', 0) > 0])
        losses = len([t for t in history if t.get('pnl', 0) <= 0])
        total = len(history)
        
        total_pnl = sum(t.get('pnl', 0) for t in history)
        
        return {
            'initial_balance': self.INITIAL_BALANCE,
            'current_balance': self.portfolio.get('balance', self.INITIAL_BALANCE),
            'total_pnl': total_pnl,
            'total_pnl_pct': ((self.portfolio.get('balance', self.INITIAL_BALANCE) - self.INITIAL_BALANCE) / self.INITIAL_BALANCE) * 100,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total) * 100 if total > 0 else 0,
            'open_positions': len(self.portfolio.get('positions', {}))
        }
    
    def format_stats_for_telegram(self):
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
        self.portfolio = {
            "balance": self.INITIAL_BALANCE,
            "equity": self.INITIAL_BALANCE,
            "positions": {},
            "history": []
        }
        self._save_portfolio()
        logger.info("📝 Paper Trader reset")


# Global instance
_paper_trader = None

def get_paper_trader():
    """Get or create paper trader instance."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader

