import os
import time
import logging
import pandas as pd
import ccxt
import requests
from datetime import datetime
from typing import Dict, List, Optional

# --- ÖNCEKİ MODÜLLERİN ENTEGRASYONU ---
# Not: Bu importların çalışması için ilgili dosyaların proje klasöründe olması gerekir.
# Biz burada simüle edilmiş importlar yerine sınıfı doğrudan kullanacak şekilde tasarlıyoruz
# veya dosya yapısının tam olduğunu varsayıyoruz.
from src.core.risk_manager import RiskManager
from src.brain.exit_strategy import SmartExitStrategy

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeEngine")

# --- TELEGRAM MESSENGER ---
class TelegramBot:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    def send_message(self, message: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram token veya Chat ID eksik! Mesaj gönderilemedi.")
            return
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram Hatası: {str(e)}")

# --- MAIN ENGINE ---
class TradeEngine:
    """
    Sistemin Kalbi. 7/24 Çalışan Ana Döngü.
    Zero-Mock Policy: Tüm veriler ccxt üzerinden canlı çekilir.
    """
    def __init__(self, symbol: str = 'BTC/USDT', timeframe: str = '15m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_running = False
        
        # API Bağlantıları
        self.exchange = self._connect_exchange()
        self.telegram = TelegramBot()
        
        # Alt Modüller
        self.risk_manager = RiskManager(max_risk_per_trade=0.01) # %1 Risk
        self.smart_exit = SmartExitStrategy()
        
        # Paper Trading Cüzdanı (Gerçek para öncesi simülasyon)
        self.paper_balance = 10000.0 # Başlangıç Bakiyesi
        self.active_position = None # { 'entry_price': float, 'amount': float, 'start_time': datetime }
        
        logger.info(f"MOTOR BAŞLATILDI: {self.symbol} | {self.timeframe} | Bakiye: {self.paper_balance}$")
        self.telegram.send_message(f"🚀 <b>DEMIR AI BAŞLATILDI</b>\nParite: {symbol}\nMod: Paper Trading\nBakiye: {self.paper_balance}$")

    def _connect_exchange(self):
        """Binance Bağlantısı"""
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key:
            logger.warning("BINANCE API KEY bulunamadı! Public verilerle çalışılacak.")
        
        return ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })

    def fetch_live_data(self) -> pd.DataFrame:
        """Canlı mum verisi çeker ve indikatörleri hesaplar."""
        try:
            # 1. Veri Çekme (Limit 100 yeterli, analiz için)
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 2. Data Integrity Check
            if df.empty or df['close'].iloc[-1] <= 0:
                raise ValueError("Bozuk veya boş veri alındı.")

            # 3. İndikatör Hesaplamaları (Optimize edilmiş parametreler varsayılıyor)
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # ATR (Volatilite için)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Veri Hatası: {str(e)}")
            return pd.DataFrame()

    def run_forever(self):
        """Sonsuz Döngü"""
        self.is_running = True
        logger.info("Döngü başladı. Piyasa izleniyor...")
        
        while self.is_running:
            try:
                # 1. Veriyi Al
                df = self.fetch_live_data()
                if df.empty:
                    time.sleep(60)
                    continue
                
                last_row = df.iloc[-1]
                current_price = float(last_row['close'])
                current_rsi = float(last_row['rsi'])
                current_atr = float(last_row['atr'])
                
                # Konsola bilgi bas (Railway logs)
                logger.info(f"Fiyat: {current_price} | RSI: {current_rsi:.2f} | ATR: {current_atr:.2f}")

                # 2. AKTİF POZİSYON YÖNETİMİ (Smart Exit)
                if self.active_position:
                    entry_price = self.active_position['entry_price']
                    bars_held = (datetime.now() - self.active_position['start_time']).seconds // 900 # 15dk'lık bar sayısı tahmini
                    
                    should_exit, reason, stop_price = self.smart_exit.calculate_exit_signal(
                        current_price=current_price,
                        entry_price=entry_price,
                        current_atr=current_atr,
                        rsi_value=current_rsi,
                        bars_held=bars_held
                    )
                    
                    if should_exit:
                        self._execute_sell(current_price, reason)
                
                # 3. YENİ GİRİŞ SİNYALİ ARAMA (Entry Strategy)
                # Basit RSI aşırı satım stratejisi (Optimizer'dan gelen mantık buraya konur)
                elif not self.active_position:
                    if current_rsi < 30: # Aşırı Satım Bölgesi -> FIRSAT
                        # Ekstra teyit: Fiyat düşüyor ama momentum (RSI) yükseliyorsa (Pozitif Uyumsuzluk)
                        logger.info("RSI < 30. Alım fırsatı aranıyor...")
                        
                        # Risk Yönetimi: Pozisyon büyüklüğü hesapla
                        stop_loss = current_price - (current_atr * 2) # 2 ATR altı stop
                        pos_size = self.risk_manager.calculate_position_size(
                            account_balance=self.paper_balance,
                            entry_price=current_price,
                            stop_loss_price=stop_loss,
                            volatility_atr=current_atr
                        )
                        
                        if pos_size > 0:
                            self._execute_buy(current_price, pos_size, stop_loss)

                # 4. Bekleme (API limitlerine saygı)
                time.sleep(60) # Her dakika kontrol et
                
            except KeyboardInterrupt:
                logger.info("Bot manuel olarak durduruldu.")
                self.is_running = False
            except Exception as e:
                logger.error(f"Döngü Hatası: {str(e)}")
                time.sleep(30)

    def _execute_buy(self, price, amount, stop_loss):
        """Paper Buy İşlemi"""
        cost = price * amount
        if cost > self.paper_balance:
            amount = self.paper_balance / price
            cost = self.paper_balance
            
        self.paper_balance -= cost
        self.active_position = {
            'entry_price': price,
            'amount': amount,
            'start_time': datetime.now(),
            'stop_loss': stop_loss
        }
        
        # Smart Exit'i sıfırla
        self.smart_exit.reset()
        self.smart_exit.trailing_stop_price = stop_loss # İlk stop seviyesini ata
        
        msg = (f"🟢 <b>ALIM SİNYALİ (BUY)</b>\n"
               f"Fiyat: {price}\n"
               f"Miktar: {amount:.4f}\n"
               f"Stop Loss: {stop_loss:.2f}\n"
               f"Bakiye: {self.paper_balance:.2f}$")
        logger.info(msg)
        self.telegram.send_message(msg)

    def _execute_sell(self, price, reason):
        """Paper Sell İşlemi"""
        amount = self.active_position['amount']
        entry_price = self.active_position['entry_price']
        revenue = price * amount
        profit = revenue - (entry_price * amount)
        profit_pct = (profit / (entry_price * amount)) * 100
        
        self.paper_balance += revenue
        self.active_position = None
        
        icon = "💰" if profit > 0 else "🛑"
        msg = (f"{icon} <b>SATIŞ İŞLEMİ (SELL)</b>\n"
               f"Çıkış Fiyatı: {price}\n"
               f"Sebep: {reason}\n"
               f"Kar/Zarar: {profit:.2f}$ (%{profit_pct:.2f})\n"
               f"Yeni Bakiye: {self.paper_balance:.2f}$")
        logger.info(msg)
        self.telegram.send_message(msg)

if __name__ == "__main__":
    # Railway'de bu dosya doğrudan çalıştırılacak
    engine = TradeEngine(symbol='BTC/USDT', timeframe='15m')
    engine.run_forever()
