import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional

from src.execution.dca_module import get_dca_module

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
        self.dca_module = get_dca_module()
        self.trailing_stops = {}  # {symbol: {'initial_sl': X, 'highest_price': Y}}

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
    
    def _notify_trade_close(self, symbol: str, side: str, entry: float, exit_price: float, 
                            pnl: float, pnl_pct: float, duration_mins: int, result: str, emoji: str):
        """
        Telegram üzerinden trade kapanış bildirimi gönder.
        """
        try:
            # Lazy import to avoid circular dependency
            from src.v10.smart_notifier import _notifier
            
            if _notifier:
                msg = f"""
{emoji} *PAPER TRADE KAPANDI* - {symbol}

📊 Sonuç: **{result}**
💰 P/L: ${pnl:.2f} ({pnl_pct:+.1f}%)

📈 Giriş: ${entry:,.0f}
📉 Çıkış: ${exit_price:,.0f}
📏 Pozisyon: {side}
⏱ Süre: {duration_mins} dakika

💼 Güncel Bakiye: ${self.portfolio['balance']:,.2f}
"""
                _notifier._send_message(msg)
                logger.info(f"📢 Trade close notification sent: {symbol}")
        except Exception as e:
            logger.warning(f"Could not send close notification: {e}")

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
                side_type = pos.get('side', 'LONG')
                
                # Kar/Zarar Hesapla (LONG vs SHORT)
                if side_type == 'LONG':
                    pnl = (price - entry_price) * amount
                    self.portfolio['balance'] += (amount * entry_price) + pnl
                else:  # SHORT
                    pnl = (entry_price - price) * amount
                    collateral = pos.get('collateral', amount * price * 0.5)
                    self.portfolio['balance'] += collateral + pnl
                
                # FEEDBACK LOOP - Trade outcome kaydet
                try:
                    from src.brain.feedback_db import get_feedback_db
                    feedback_db = get_feedback_db()
                    
                    entry_time = datetime.fromisoformat(pos['timestamp'])
                    duration_mins = (datetime.now() - entry_time).seconds // 60
                    pnl_pct = (pnl / (amount * entry_price)) * 100
                    
                    feedback_db.save_trade_outcome({
                        'symbol': symbol,
                        'side': side_type,
                        'entry_features': pos.get('entry_snapshot', {}),
                        'predicted_action': pos.get('predicted_action', 'UNKNOWN'),
                        'actual_pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'duration_minutes': duration_mins,
                        'entry_price': entry_price,
                        'exit_price': price
                    })
                    logger.info(f"✅ Feedback saved: {symbol} PnL ${pnl:.2f}")
                except Exception as e:
                    logger.warning(f"Feedback save error: {e}")
                
                # Geçmişe kaydet
                self.portfolio['history'].append({
                    "symbol": symbol,
                    "action": "SELL",
                    "side": side_type,
                    "entry": entry_price,
                    "exit": price,
                    "amount": amount,
                    "pnl": pnl,
                    "time": datetime.now().isoformat()
                })
                
                del self.portfolio['positions'][symbol]
                self._save_portfolio()
                
                # TELEGRAM KAPANIŞ BİLDİRİMİ
                pnl_emoji = "🟢" if pnl > 0 else "🔴"
                result = "KAZANÇ" if pnl > 0 else "KAYIP"
                self._notify_trade_close(symbol, side_type, entry_price, price, pnl, pnl_pct, duration_mins, result, pnl_emoji)
                
                logger.info(f"💰 CLOSED {side_type} {symbol}: PnL ${pnl:.2f}")
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
            
            amount = self.calculate_position_size(price, sl_price, self.portfolio['balance'], risk_pct)
            
            if amount > 0:
                cost = amount * price
                if cost > self.portfolio['balance']:
                    logger.warning("Yetersiz bakiye")
                    return False
                
                self.portfolio['balance'] -= cost
                self.portfolio['positions'][symbol] = {
                    "entry_price": price,
                    "sl_price": sl_price,
                    "amount": amount,
                    "cost": cost,
                    "side": "LONG",
                    "timestamp": datetime.now().isoformat(),
                    "predicted_action": signal.get('side', 'BUY'),
                    "entry_snapshot": {
                        'rsi': signal.get('rsi', -1),
                        'ob_ratio': signal.get('ob_ratio', -1),
                        'funding': signal.get('funding', 0),
                        'volatility': signal.get('volatility', 0),
                        'regime': signal.get('regime', 'UNKNOWN')
                    }
                }
                
                self._save_portfolio()
                logger.info(f"🛒 BOUGHT {symbol}: {amount:.4f} units @ ${price}")
                return True
        
        # --- SHORT POZİSYON AÇMA (SELL) ---
        elif side == "SELL" or side == "SHORT":
            # Zaten açık pozisyon varsa ekleme yapma
            if symbol in self.portfolio['positions']:
                # Eğer long pozisyon varsa kapat, sonra short aç
                pos = self.portfolio['positions'][symbol]
                if pos.get('side') == 'LONG':
                    logger.info(f"Closing LONG before opening SHORT on {symbol}")
                    # Pozisyonu kapat (quick close)
                    amount = pos['amount']
                    entry = pos['entry_price']
                    pnl = (price - entry) * amount
                    self.portfolio['balance'] += (amount * entry) + pnl
                    
                    self.portfolio['history'].append({
                        "symbol": symbol,
                        "action": "CLOSE_LONG",
                        "entry": entry,
                        "exit": price,
                        "amount": amount,
                        "pnl": pnl,
                        "time": datetime.now().isoformat()
                    })
                    del self.portfolio['positions'][symbol]
                    logger.info(f"💰 CLOSED LONG {symbol}: PnL ${pnl:.2f}")
                else:
                    # Zaten SHORT varsa yeni short açma (veya kapatma sinyali ise kapat)
                    # Eğer sinyal sadece "SELL" ise ve mevcut pozisyon SHORT ise kapat
                    if side == "SELL" and pos.get('side') == 'SHORT':
                        amount = pos['amount']
                        entry = pos['entry_price']
                        collateral = pos['collateral']
                        
                        # PnL hesapla (SHORT için ters)
                        pnl = (entry - price) * amount
                        
                        # Bakiyeye geri ekle (collateral + PnL)
                        self.portfolio['balance'] += collateral + pnl
                        
                        self.portfolio['history'].append({
                            "symbol": symbol,
                            "action": "SELL_SHORT", # Kapatılan SHORT
                            "entry": entry,
                            "exit": price,
                            "amount": amount,
                            "pnl": pnl,
                            "time": datetime.now().isoformat()
                        })
                        del self.portfolio['positions'][symbol]
                        self._save_portfolio()
                        logger.info(f"💰 CLOSED SHORT {symbol}: PnL ${pnl:.2f}")
                        return True
                    else:
                        # Zaten SHORT varsa ve sinyal "SHORT" ise, yeni short açma
                        return False
            
            # SHORT pozisyon aç
            sl_price = float(signal.get('sl_price', price * 1.02))  # SHORT için SL yukarıda
            risk_pct = signal.get('kelly_size', 1.0) / 100.0
            if risk_pct <= 0: risk_pct = 0.01
            
            # SHORT için position size hesaplama (ters yönde)
            risk_per_unit = abs(sl_price - price)
            if risk_per_unit > 0:
                max_risk = self.portfolio['balance'] * risk_pct
                amount = max_risk / risk_per_unit
                
                # Max %10 balance short (örnek bir limit)
                # Bu limit, kaldıraçlı işlemlerde marjin gereksinimini simüle eder.
                # Gerçek bir borsada marjin hesaplaması daha karmaşık olacaktır.
                max_amount_based_on_balance = (self.portfolio['balance'] * 0.10) / price # %10'u kadar collateral ile
                amount = min(amount, max_amount_based_on_balance)
            else:
                amount = 0
            
            if amount > 0:
                # SHORT: Collateral lock (simulate margin)
                # Basit bir 2x kaldıraç simülasyonu, %50 collateral
                collateral = amount * price * 0.5  
                if collateral > self.portfolio['balance']:
                    logger.warning("Yetersiz bakiye (SHORT)")
                    return False
                
                self.portfolio['balance'] -= collateral
                self.portfolio['positions'][symbol] = {
                    "entry_price": price,
                    "sl_price": sl_price,
                    "amount": amount,
                    "side": "SHORT",  # Pozisyon yönü
                    "collateral": collateral,
                    "timestamp": datetime.now().isoformat()
                }
                
                self._save_portfolio()
                logger.info(f"📉 SHORTED {symbol}: {amount:.4f} units @ ${price}")
                return True
                
        return False
    
    def update_positions(self, current_prices: Dict[str, float]) -> None:
        """
        AKILLI POZİSYON YÖNETİMİ
        
        Her döngüde çalışır:
        1. Trailing Stop günceller
        2. DCA fırsatlarını kontrol eder
        3. SL/TP kontrolü yapar (otomatik kapat)
        
        Args:
            current_prices: {symbol: current_price}
        """
        closed_positions = []
        
        for symbol in list(self.portfolio['positions'].keys()):
            if symbol not in current_prices:
                continue
                
            current_price = current_prices[symbol]
            pos = self.portfolio['positions'][symbol]
            entry_price = pos['entry_price']
            current_sl = pos['sl_price']
            
            # 1. TRAILING STOP UPDATE
            self._update_trailing_stop(symbol, current_price, entry_price, current_sl)
            
            # 2. DCA CHECK
            self._check_and_execute_dca(symbol, current_price, entry_price, pos['amount'])
            
            # 3. SL/TP CHECK (Auto-close)
            # Refresh SL (may have been updated by trailing stop)
            current_sl = self.portfolio['positions'][symbol]['sl_price']
            side = pos.get('side', 'LONG')
            
            # Check Stop Loss (LONG vs SHORT logic)
            if side == 'LONG':
                # LONG: SL below entry, TP above
                if current_price <= current_sl:
                    logger.info(f"🛑 STOP LOSS HIT (LONG): {symbol} @ ${current_price:.2f}")
                    self.execute_trade({
                        "symbol": symbol,
                        "side": "SELL",
                        "entry_price": current_price
                    })
                    closed_positions.append(symbol)
                    continue
            else:  # SHORT
                # SHORT: SL above entry, TP below
                if current_price >= current_sl:
                    logger.info(f"🛑 STOP LOSS HIT (SHORT): {symbol} @ ${current_price:.2f}")
                    self.execute_trade({
                        "symbol": symbol,
                        "side": "SELL",  # Close SHORT
                        "entry_price": current_price
                    })
                    closed_positions.append(symbol)
                    continue
            
            # Simple TP check (if we had TP field)
            # For now, manual TP via Early Signal confidence
        
        # Clean up trailing stop data for closed positions
        for symbol in closed_positions:
            if symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
            if symbol in self.dca_module.positions:
                self.dca_module.close_position(symbol)
    
    def _update_trailing_stop(self, symbol: str, current_price: float, 
                              entry_price: float, current_sl: float) -> None:
        """
        Trailing Stop: Kârda iken SL'i yukarı çek.
        """
        # Initialize tracking
        if symbol not in self.trailing_stops:
            self.trailing_stops[symbol] = {
                'initial_sl': current_sl,
                'highest_price': entry_price
            }
        
        ts = self.trailing_stops[symbol]
        
        # Update highest seen price
        if current_price > ts['highest_price']:
            ts['highest_price'] = current_price
        
        # Calculate profit %
        profit_pct = ((current_price / entry_price) - 1) * 100
        
        # Adjust SL based on profit tiers
        new_sl = current_sl
        
        if profit_pct >= 10:  # +10% profit
            new_sl = entry_price * 1.05  # Move SL to +5% profit (lock in gains)
        elif profit_pct >= 5:   # +5% profit
            new_sl = entry_price  # Move SL to break-even (risk-free)
        elif profit_pct >= 2:   # +2% profit
            new_sl = max(current_sl, entry_price * 0.995)  # Tighten SL slightly
        
        # Update if improved (never move SL down!)
        if new_sl > current_sl:
            self.portfolio['positions'][symbol]['sl_price'] = new_sl
            self._save_portfolio()
            logger.info(
                f"📈 TRAILING STOP: {symbol} | "
                f"Profit: +{profit_pct:.1f}% | "
                f"SL ${current_sl:.0f} → ${new_sl:.0f}"
            )
    
    def _check_and_execute_dca(self, symbol: str, current_price: float,
                                entry_price: float, quantity: float) -> None:
        """
        DCA: Kayıptayken akıllı ek alım.
        """
        # Create DCA tracking if not exists
        if symbol not in self.dca_module.positions:
            self.dca_module.create_position(
                symbol=symbol,
                entry_price=entry_price,
                quantity=quantity
            )
        
        # Check DCA opportunity
        dca_signal = self.dca_module.check_dca_opportunity(
            symbol=symbol,
            current_price=current_price,
            market_data=None  # Could pass RSI, trend, etc.
        )
        
        if dca_signal:
            # Execute DCA buy
            logger.info(
                f"💰 DCA TRIGGERED: {symbol} Level {dca_signal['dca_level']} | "
                f"Drop: {dca_signal['drop_pct']:.1f}% | "
                f"Buy: {dca_signal['buy_quantity']:.4f} units"
            )
            
            # Add to position (simple buy, no new SL calc)
            pos = self.portfolio['positions'][symbol]
            add_cost = current_price * dca_signal['buy_quantity']
            
            if add_cost <= self.portfolio['balance']:
                # Update position
                new_total_qty = pos['amount'] + dca_signal['buy_quantity']
                new_total_cost = pos['cost'] + add_cost
                new_avg_price = new_total_cost / new_total_qty
                
                self.portfolio['balance'] -= add_cost
                pos['amount'] = new_total_qty
                pos['cost'] = new_total_cost
                pos['entry_price'] = new_avg_price  # Update avg
                
                self._save_portfolio()
                
                # Record in DCA module
                self.dca_module.execute_dca(symbol, current_price, dca_signal['buy_quantity'])
                
                logger.info(
                    f"✅ DCA EXECUTED: {symbol} | "
                    f"New Avg: ${new_avg_price:.2f} | "
                    f"Total: {new_total_qty:.4f} units"
                )
            else:
                logger.warning(f"⚠️ DCA skipped: Insufficient balance")

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
        self.portfolio = {
            "balance": self.INITIAL_BALANCE,
            "equity": self.INITIAL_BALANCE,
            "positions": {},
            "history": []
        }
        self._save_portfolio()
        logger.info("📝 Paper Trader reset")

    def get_trade_history(self):
        """Get list of past trades"""
        return self.portfolio.get('history', [])
    
    def get_open_positions(self):
        """Get currently open positions"""
        return self.portfolio.get('positions', {})
    
    def get_balance(self):
        """Get current balance"""
        return self.portfolio.get('balance', 0.0)


# Global instance
_paper_trader = None

def get_paper_trader():
    """Get or create paper trader instance."""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader

