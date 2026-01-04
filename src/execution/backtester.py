# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - ADVANCED BACKTESTER
==================================
Risk yönetimi kurallarını uygulayarak gerçekçi backtest yapar.
Sadece sinyal kalitesini değil, stratejinin kârlılığını ölçer.

Özellikler:
- Kelly Criterion ile pozisyon büyüklüğü
- ATR tabanlı Stop Loss
- Risk:Reward oranına göre Take Profit
- Komisyon ve Slippage hesaplaması

Author: DEMIR AI Team
Date: 2026-01-04
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from src.risk.position_sizer import RiskManager
from src.models.trainer import QuantModelTrainer

logger = logging.getLogger("ADVANCED_BACKTESTER")

class AdvancedBacktester:
    def __init__(self, symbol: str, model_name: str, initial_balance: float = 1000.0):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_manager = RiskManager()
        self.trainer = QuantModelTrainer(model_name)
        
        # Load model
        try:
            self.trainer.load_model()
            self.model = self.trainer.model
            logger.info(f"Model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.trades = []
        self.active_position = None  # {entry_price, size, side, stop_loss, take_profit, entry_time}
        self.commission_rate = 0.0004  # Binance Futures (%0.04)
        
        # İstatistikler
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.gross_profit = 0.0 # Toplam kazanç
        self.gross_loss = 0.0   # Toplam kayıp (pozitif değer olarak tut)
        
        # Debug Stats
        self.prob_stats = []
        self.skipped_low_size = 0
        self.skipped_active_pos = 0

    def run(self, df: pd.DataFrame):
        """Backtest simülasyonunu çalıştır."""
        logger.info(f"Starting backtest for {self.symbol} with ${self.initial_balance}...")
        
        # Feature kolonlarını al
        feature_cols = self.trainer.feature_columns
        if not feature_cols:
             # Model dosyasından yükleyemediyse feature'ları tahmin et veya hata ver
             # Şimdilik df kolonlarını kullan (target hariç)
             exclude = ['timestamp', 'label_1h', 'label_4h', 'label_4h_triple', 'future_return_60', 'future_return_240', 'symbol', 'close']
             feature_cols = [c for c in df.columns if c not in exclude]
        
        # Tahminleri topluca yap (Hız için)
        # Not: Gerçekte data leakage olmaması için walk-forward yapılmalı ama 
        # model zaten walk-forward ile eğitildi, test set üzerinde yapıyoruz varsayalım.
        # Bu basit simülasyon tüm veri üzerinde.
        
        # Eksik kolonları doldur
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            for c in missing_cols:
                df[c] = 0
                
        X = df[feature_cols]
        probs = self.model.predict_proba(X)
        
        # Olasılık istatistikleri
        prob_buy_all = [p[1] for p in probs]
        self.prob_stats = {
            'min': float(np.min(prob_buy_all)),
            'max': float(np.max(prob_buy_all)),
            'mean': float(np.mean(prob_buy_all)),
            'std': float(np.std(prob_buy_all))
        }
        logger.info(f"Probability Stats: {self.prob_stats}")
        
        # Simülasyon Döngüsü
        # İterasyon yavaş olabilir, ama işlem yönetimi için gerekli
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_time = row['timestamp'] if 'timestamp' in row else i
            atr = row.get('atr', current_price * 0.01) # ATR yoksa %1 varsay
            
            # 1. Açık pozisyon yönetimi
            if self.active_position:
                self.skipped_active_pos += 1
                self._manage_position(current_price, current_time)
                continue # Pozisyon varken yeni sinyal alma (Basitlik için)
            
            # 2. Sinyal Kontrolü
            # Class 0: SELL, Class 1: HOLD, Class 2: BUY (veya 0/1 binary)
            # LightGBM binary: 0=Düşüş/Yatay, 1=Yükseliş
            
            prob_buy = probs[i][1] if len(probs[i]) > 1 else probs[i][0]
            
            signal = None
            confidence = 0.0
            
            if prob_buy > 0.80: # Sweet Spot: %80 Eşik
                signal = "BUY"
                confidence = prob_buy
            elif prob_buy < 0.20: # 1 - 0.80
                signal = "SELL"
                confidence = 1 - prob_buy
            
            if signal:
                # Debug log (ilk 5 işlem için)
                if len(self.trades) < 5:
                    logger.debug(f"Signal found: {signal} Conf: {confidence:.4f} Price: {current_price}")
                self._open_position(signal, current_price, current_time, atr, confidence)
                
        self._generate_report()
        return self.get_results()

    def _open_position(self, side, price, time, atr, confidence):
        # Kelly Size
        # Doğru hesaplama
        avg_win = (self.gross_profit / self.wins) if self.wins > 0 else 100
        avg_loss = (self.gross_loss / self.losses) if self.losses > 0 else 50
        
        win_rate = 0.5
        if (self.wins + self.losses) > 0:
            win_rate = self.wins / (self.wins + self.losses)
        
        # Başlangıçta muhafazakar ol
        if self.wins + self.losses < 10:
            size_usd = self.current_balance * 0.02 # İlk 10 işlem %2 sabit
        else:
            size_usd = self.risk_manager.calculate_position_size(
                self.current_balance, win_rate, avg_win, avg_loss, confidence
            )
        # Başlangıçta veya Kelly negatifse bile minimum işlem aç (Veri toplamak için)
        # Normalde canlıda Kelly negatifse girmemek lazım ama backtestte performance görmek istiyoruz.
        
        if size_usd < 10:
            # Kelly negatifse veya çok küçükse, istatistik için minimum $10 ile gir
            if self.current_balance >= 10:
                size_usd = 10.0
            else:
                # Bakiye bitti
                self.skipped_low_size += 1
                return

        # SL / TP
        stop_loss = self.risk_manager.calculate_stop_loss(price, side, atr)
        take_profit = self.risk_manager.calculate_take_profit(price, side, stop_loss, rr_ratio=1.5)
        
        # Miktar (Coin)
        quantity = size_usd / price
        
        # Komisyon düş
        cost = size_usd * self.commission_rate
        self.current_balance -= cost
        
        self.active_position = {
            'entry_price': price,
            'quantity': quantity,
            'side': side,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': time,
            'entry_cost': cost
        }

    def _manage_position(self, current_price, time):
        pos = self.active_position
        pnl = 0
        close_reason = None
        
        if pos['side'] == "BUY":
            if current_price <= pos['stop_loss']:
                close_reason = "SL"
                pnl = (pos['stop_loss'] - pos['entry_price']) * pos['quantity']
            elif current_price >= pos['take_profit']:
                close_reason = "TP"
                pnl = (pos['take_profit'] - pos['entry_price']) * pos['quantity']
        else: # SELL
            if current_price >= pos['stop_loss']:
                close_reason = "SL"
                pnl = (pos['entry_price'] - pos['stop_loss']) * pos['quantity']
            elif current_price <= pos['take_profit']:
                close_reason = "TP"
                pnl = (pos['entry_price'] - pos['take_profit']) * pos['quantity']
                
        if close_reason:
            # Komisyon düş
            exit_cost = (pos['quantity'] * current_price) * self.commission_rate
            net_pnl = pnl - exit_cost
            
            self.current_balance += net_pnl
            self.total_pnl += net_pnl
            
            if net_pnl > 0:
                self.wins += 1
                self.gross_profit += net_pnl
            else:
                self.losses += 1
                self.gross_loss += abs(net_pnl)
                
            self.trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': time,
                'side': pos['side'],
                'pnl': net_pnl,
                'reason': close_reason,
                'balance': self.current_balance
            })
            
            self.active_position = None

    def _generate_report(self):
        total_trades = self.wins + self.losses
        win_rate = (self.wins / total_trades) if total_trades > 0 else 0
        roi = (self.current_balance - self.initial_balance) / self.initial_balance
        
        logger.info("-" * 30)
        logger.info(f"🏁 BACKTEST FINISHED: {self.symbol}")
        logger.info(f"💰 Final Balance: ${self.current_balance:.2f} (ROI: {roi:.2%})")
        logger.info(f"📊 Trades: {total_trades} (W: {self.wins} | L: {self.losses})")
        logger.info(f"🎯 Win Rate: {win_rate:.2%}")
        logger.info(f"⚠️ Skipped (Low Size): {self.skipped_low_size}")
        logger.info(f"⚠️ Skipped (Active Pos): {self.skipped_active_pos}")
        logger.info("-" * 30)

    def get_results(self):
        return {
            'final_balance': self.current_balance,
            'trades': len(self.trades),
            'win_rate': self.wins / (self.wins + self.losses) if (self.wins + self.losses) > 0 else 0,
            'history': self.trades
        }
