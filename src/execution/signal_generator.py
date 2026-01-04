# -*- coding: utf-8 -*-
"""
DEMIR AI v11 - SIGNAL GENERATOR (LIVE)
======================================
Canlı piyasa verisiyle anlık sinyal üretir.
Faz 5 entegrasyonunun kalbidir.

İşleyiş:
1. Collector ile son veriyi çek
2. Feature'ları hesapla (Technical Analysis)
3. Model ile yön tahmini yap (%80 Eşik)
4. Risk Manager ile SL/TP ve Pozisyon büyüklüğü hesapla
5. Sinyal objesi döndür

Author: DEMIR AI Team
Date: 2026-01-04
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.data_pipeline.collector import get_data_collector
from src.features.technical import TechFeatureEngineer
from src.models.trainer import QuantModelTrainer
from src.risk.position_sizer import RiskManager

logger = logging.getLogger("SIGNAL_GENERATOR")

class SignalGenerator:
    def __init__(self, symbols: list):
        self.symbols = symbols
        self.collector = get_data_collector()
        self.risk_manager = RiskManager()
        self.feature_eng = TechFeatureEngineer()
        
        # Modelleri önbelleğe al
        self.models = {}
        self.trainers = {}
        
        for symbol in symbols:
            model_name = f"quant_{symbol.lower().replace('usdt', '')}"
            trainer = QuantModelTrainer(model_name)
            try:
                trainer.load_model()
                self.models[symbol] = trainer.model
                self.trainers[symbol] = trainer
                logger.info(f"✅ Model loaded for {symbol}")
            except Exception as e:
                logger.error(f"❌ Failed to load model for {symbol}: {e}")

    async def check_for_signals(self) -> list:
        """
        Tüm sembolleri tara ve sinyal üret.
        """
        signals = []
        
        for symbol in self.symbols:
            if symbol not in self.models:
                continue
                
            try:
                # 1. Veri Güncelle (Incremental)
                # Live modda sadece son veriye ihtiyacımız var ama indikatörler için
                # biraz geçmiş veriye de ihtiyaç var. Parquet'i yükleyip update ediyoruz.
                df = await self.collector.update_symbol(symbol, interval="1m")
                
                if df is None or len(df) < 100:
                    logger.warning(f"Not enough data for {symbol}")
                    continue
                
                # 2. Feature Calculation
                # Tüm data üzerinde feature çıkar (en son satır önemli)
                # İleride performans için sadece son 1000 satırı alabiliriz
                df_features = self.feature_eng.generate_features(df)
                
                # 3. Model Hazırlığı
                # Trainer'daki column listesini kullan (böylece eğitimdeki ile aynı olur)
                feature_cols = self.trainers[symbol].feature_columns
                if not feature_cols:
                    # Fallback (eğer modelde kayıtlı değilse)
                    exclude = ['timestamp', 'label_1h', 'label_4h', 'label_4h_triple', 'future_return_60', 'future_return_240', 'symbol', 'close']
                    feature_cols = [c for c in df_features.columns if c not in exclude]
                
                # Son satırı al (Live Signal)
                last_row = df_features.iloc[[-1]] # DataFrame olarak kalmalı
                current_price = float(last_row['close'].values[0])
                current_time = last_row['timestamp'].values[0]
                
                # Eksik kolon kontrolü ve doldurma
                missing_cols = set(feature_cols) - set(last_row.columns)
                for c in missing_cols:
                    last_row[c] = 0
                
                X_live = last_row[feature_cols]
                
                # 4. Predict
                probs = self.models[symbol].predict_proba(X_live)
                prob_buy = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                
                # 5. Sinyal Kararı
                signal_side = None
                confidence = 0.0
                
                # EŞİK: %80 (SWEET SPOT)
                THRESHOLD = 0.80
                
                if prob_buy > THRESHOLD:
                    signal_side = "BUY"
                    confidence = prob_buy
                elif prob_buy < (1 - THRESHOLD): # < 0.20
                    signal_side = "SELL"
                    confidence = 1 - prob_buy
                
                if signal_side:
                    # 6. Risk Yönetimi
                    atr = float(last_row.get('atr', current_price * 0.01).values[0])
                    
                    # Backtest'te istatistik topladık, canlıda başlangıç değerleri verebiliriz
                    # Veya risk manager'a bir state verebiliriz.
                    # Şimdilik Safe Kelly için varsayılan değerler:
                    # Win Rate %45, Win/Loss 1.5 (Backtest sonuçlarına göre)
                    
                    size_usd = self.risk_manager.calculate_position_size(
                        account_balance=1000, # Örnek bakiye, API'den çekilmeli
                        win_rate=0.45,
                        avg_win=1.5,
                        avg_loss=1.0,
                        confidence=confidence
                    )
                    
                    # Stop/TP
                    sl = self.risk_manager.calculate_stop_loss(current_price, signal_side, atr)
                    tp = self.risk_manager.calculate_take_profit(current_price, signal_side, sl)
                    
                    signal_data = {
                        "symbol": symbol,
                        "side": signal_side,
                        "price": current_price,
                        "time": current_time,
                        "confidence": confidence,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "size_usd": size_usd,
                        "risk_ratio": abs(current_price - sl) / current_price * 100
                    }
                    
                    signals.append(signal_data)
                    logger.info(f"🚨 SIGNAL FOUND: {symbol} {signal_side} ({confidence:.2f})")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                
        return signals

