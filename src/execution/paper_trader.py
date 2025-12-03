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

    def calculate_position_size(self, entry_price, stop_loss_price, balance):
        """
        PROFESYONEL LOT HESABI
        Formül: (Kasa * Risk%) / (Giriş - Stop)
        Amaç: Stop olursam kasamın sadece %1'i gitsin.
        """
        if entry_price <= 0 or stop_loss_price <= 0: return 0.0
        
        # Hisse başına risk miktarı (Fiyat farkı)
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0: return 0.0
        
        # Riske atılacak toplam dolar (Örn: 10.000$ * 0.01 = 100$)
        money_at_risk = balance * self.RISK_PER_TRADE
        
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
            quantity = self.calculate_position_size(price, sl_price, self.portfolio['balance'])
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
