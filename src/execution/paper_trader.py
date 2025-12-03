import json
import os
import logging
from datetime import datetime

logger = logging.getLogger("PAPER_TRADER")

class PaperTrader:
    """
    DEMIR AI - LIVE SIMULATION ENGINE
    
    Gerçek para kullanmadan canlı piyasada işlem yapar.
    Tüm işlemleri ve bakiyeyi 'portfolio.json' dosyasında tutar.
    """
    
    DB_FILE = "portfolio.json"
    INITIAL_BALANCE = 10000.0 # 10k Sanal Dolar ile başlıyoruz

    def __init__(self):
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self):
        """Cüzdan dosyasını yükler, yoksa oluşturur."""
        if os.path.exists(self.DB_FILE):
            try:
                with open(self.DB_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Yeni Cüzdan Başlat
        return {
            "balance": self.INITIAL_BALANCE,
            "equity": self.INITIAL_BALANCE,
            "positions": {}, # { 'BTC/USDT': { 'entry': 50000, 'size': 0.1, ... } }
            "history": []
        }

    def _save_portfolio(self):
        """Cüzdanı kaydeder."""
        with open(self.DB_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=4)

    def execute_trade(self, signal):
        """
        Sinyal geldiğinde işlem açar veya kapatır.
        """
        symbol = signal['symbol']
        price = float(signal['entry_price'])
        side = signal['side'] # BUY veya SELL
        
        # 1. POZİSYON KAPATMA (SELL)
        if side == "SELL":
            if symbol in self.portfolio['positions']:
                pos = self.portfolio['positions'][symbol]
                amount = pos['amount']
                entry_price = pos['entry_price']
                
                # Kar/Zarar Hesapla
                # (Satış Fiyatı - Alış Fiyatı) * Adet
                pnl = (price - entry_price) * amount
                
                # Bakiyeye ekle
                self.portfolio['balance'] += (amount * entry_price) + pnl
                
                # Geçmişe kaydet
                self.portfolio['history'].append({
                    "symbol": symbol,
                    "action": "SELL",
                    "entry": entry_price,
                    "exit": price,
                    "pnl": pnl,
                    "time": datetime.now().isoformat()
                })
                
                del self.portfolio['positions'][symbol]
                self._save_portfolio()
                logger.info(f"🧻 PAPER TRADE: Sold {symbol} at {price}. PnL: ${pnl:.2f}")
                return True

        # 2. POZİSYON AÇMA (BUY)
        elif side == "BUY":
            # Zaten pozisyon varsa alma (Piramitleme yok)
            if symbol in self.portfolio['positions']:
                return False
            
            # Kasanın %20'si ile işlem aç (Risk Yönetimi)
            trade_amount_usdt = self.portfolio['balance'] * 0.20
            
            if trade_amount_usdt < 10: return False # Min işlem tutarı
            
            quantity = trade_amount_usdt / price
            
            # Bakiyeden düş
            self.portfolio['balance'] -= trade_amount_usdt
            
            # Pozisyonu kaydet
            self.portfolio['positions'][symbol] = {
                "entry_price": price,
                "amount": quantity,
                "cost": trade_amount_usdt,
                "time": datetime.now().isoformat()
            }
            
            self._save_portfolio()
            logger.info(f"🧻 PAPER TRADE: Bought {symbol} at {price}. Size: ${trade_amount_usdt:.2f}")
            return True
            
        return False

    def get_portfolio_status(self, current_prices={}):
        """
        Canlı PnL ve Equity (Varlık) değerini hesaplar.
        """
        equity = self.portfolio['balance']
        open_positions = []
        
        for symbol, pos in self.portfolio['positions'].items():
            current_price = current_prices.get(symbol, pos['entry_price'])
            market_value = pos['amount'] * current_price
            unrealized_pnl = market_value - pos['cost']
            pnl_pct = (unrealized_pnl / pos['cost']) * 100
            
            equity += market_value
            
            open_positions.append({
                "symbol": symbol,
                "entry": pos['entry_price'],
                "current": current_price,
                "pnl": unrealized_pnl,
                "pnl_pct": pnl_pct
            })
            
        self.portfolio['equity'] = equity
        return {
            "balance": self.portfolio['balance'],
            "equity": equity,
            "positions": open_positions,
            "history": self.portfolio['history'][-10:] # Son 10 işlem
        }
