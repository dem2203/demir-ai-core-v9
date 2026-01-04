# -*- coding: utf-8 -*-
"""
DEMIR AI v11.1 - SIGNAL GENERATOR (LIVE + ADVANCED)
====================================================
Canlı piyasa verisiyle anlık sinyal üretir.
FAZ 6: Whale, Liquidation ve Sentiment entegrasyonu eklendi.

ÖNEMLI: GERÇEK ZAMANLI FİYAT KULLANILIR - PARQUET ESKİ VERİSİ DEĞİL!

İşleyiş:
1. Binance API'den GERÇEK ZAMANLI fiyat al
2. Collector ile son veriyi çek (feature için)
3. Feature'ları hesapla (Technical Analysis)
4. Model ile yön tahmini yap (%80 Eşik)
5. [YENİ] Advanced modüller ile boost/block kontrolü
6. Risk Manager ile SL/TP ve Pozisyon büyüklüğü hesapla
7. Sinyal objesi döndür

KURALLAR:
- ASLA MOCK/TEST/FALLBACK VERİ KULLANMA
- Her zaman Binance API'den gerçek fiyat al
- Sinyal cooldown: Aynı coin için 1 saat bekle

Author: DEMIR AI Team
Date: 2026-01-04
"""
import pandas as pd
import numpy as np
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.data_pipeline.collector import get_data_collector
from src.features.technical import TechnicalFeatures
from src.models.trainer import QuantModelTrainer
from src.risk.position_sizer import RiskManager
from src.execution.advanced_signals import get_advanced_enhancer

logger = logging.getLogger("SIGNAL_GENERATOR")


class SignalGenerator:
    # Binance Futures API
    BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
    
    # Sinyal cooldown (1 saat)
    SIGNAL_COOLDOWN = timedelta(hours=1)
    
    def __init__(self, symbols: list, use_advanced: bool = True):
        """
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            use_advanced: Enable whale/liquidation/sentiment integration
        """
        self.symbols = symbols
        self.collector = get_data_collector()
        self.risk_manager = RiskManager()
        self.feature_eng = TechnicalFeatures()
        self.use_advanced = use_advanced
        
        # Advanced Enhancer (Whale + Liquidation + Sentiment)
        self.enhancer = get_advanced_enhancer() if use_advanced else None
        
        # Sinyal geçmişi (spam engellemek için)
        self._last_signals: Dict[str, dict] = {}
        
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

    async def initialize(self):
        """Initialize async components."""
        if self.enhancer:
            await self.enhancer.initialize()
            logger.info("🚀 Advanced Signal Enhancer initialized")

    async def _get_realtime_price(self, symbol: str) -> Optional[float]:
        """Binance API'den GERÇEK ZAMANLI fiyat al."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BINANCE_PRICE_URL, 
                    params={"symbol": symbol},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data.get("price", 0))
                        if price > 0:
                            return price
                    logger.error(f"Binance API error: {resp.status}")
        except Exception as e:
            logger.error(f"Failed to get realtime price for {symbol}: {e}")
        return None

    def _is_signal_on_cooldown(self, symbol: str, side: str) -> bool:
        """Sinyal cooldown kontrolü - aynı yönde 1 saat bekle."""
        if symbol not in self._last_signals:
            return False
        
        last = self._last_signals[symbol]
        
        # Aynı yönde ve cooldown süresi dolmadıysa
        if last.get("side") == side:
            time_diff = datetime.now() - last.get("time", datetime.min)
            if time_diff < self.SIGNAL_COOLDOWN:
                remaining = (self.SIGNAL_COOLDOWN - time_diff).seconds // 60
                logger.info(f"⏳ {symbol} {side} signal on cooldown ({remaining} min remaining)")
                return True
        
        return False

    def _record_signal(self, symbol: str, side: str):
        """Sinyal kaydı yap (cooldown için)."""
        self._last_signals[symbol] = {
            "side": side,
            "time": datetime.now()
        }

    async def check_for_signals(self) -> list:
        """
        Tüm sembolleri tara ve sinyal üret.
        """
        signals = []
        
        for symbol in self.symbols:
            if symbol not in self.models:
                continue
                
            try:
                # 1. GERÇEK ZAMANLI FİYAT AL (KRİTİK!)
                realtime_price = await self._get_realtime_price(symbol)
                if realtime_price is None:
                    logger.warning(f"❌ Could not get realtime price for {symbol}, SKIPPING!")
                    continue
                
                logger.info(f"💵 {symbol} Realtime Price: ${realtime_price:,.2f}")
                
                # 2. Veri Güncelle (Feature hesaplama için)
                df = await self.collector.update_symbol(symbol, interval="1m")
                
                if df is None or len(df) < 100:
                    logger.warning(f"Not enough data for {symbol}")
                    continue
                
                # 3. Feature Calculation
                df_features = self.feature_eng.calculate_all(df)
                
                # 4. Model Hazırlığı
                feature_cols = self.trainers[symbol].feature_columns
                if not feature_cols:
                    exclude = ['timestamp', 'label_1h', 'label_4h', 'label_4h_triple', 
                               'future_return_60', 'future_return_240', 'symbol', 'close']
                    feature_cols = [c for c in df_features.columns if c not in exclude]
                
                # Son satırı al
                last_row = df_features.iloc[[-1]]
                
                # ÖNEMLI: Parquet'ten gelen fiyatı KULLANMA, gerçek zamanlı fiyatı kullan!
                current_price = realtime_price
                current_time = datetime.now()
                
                # Eksik kolon kontrolü
                missing_cols = set(feature_cols) - set(last_row.columns)
                for c in missing_cols:
                    last_row[c] = 0
                
                X_live = last_row[feature_cols]
                
                # 5. Predict
                probs = self.models[symbol].predict_proba(X_live)
                prob_buy = probs[0][1] if len(probs[0]) > 1 else probs[0][0]
                
                # 6. Sinyal Kararı - ADVANCED INTEGRATION
                signal_side = None
                confidence = 0.0
                advanced_data = None
                
                # Base eşik: %80
                BASE_THRESHOLD = 0.80
                
                # Advanced modüllerden boost al
                if self.use_advanced and self.enhancer:
                    boost = await self.enhancer.get_boost_score(symbol)
                    effective_threshold = BASE_THRESHOLD - boost
                    effective_threshold = max(0.70, min(0.85, effective_threshold))
                else:
                    effective_threshold = BASE_THRESHOLD
                
                if prob_buy > effective_threshold:
                    signal_side = "BUY"
                    confidence = prob_buy
                elif prob_buy < (1 - effective_threshold):
                    signal_side = "SELL"
                    confidence = 1 - prob_buy
                
                if signal_side:
                    # 6.1 Cooldown Check - SPAM ENGELLEME
                    if self._is_signal_on_cooldown(symbol, signal_side):
                        continue
                    
                    # 6.2 Advanced Block Check
                    if self.use_advanced and self.enhancer:
                        blocked, reason = await self.enhancer.should_block_signal(symbol, signal_side)
                        if blocked:
                            logger.warning(f"⛔ Signal BLOCKED for {symbol}: {reason}")
                            continue
                        
                        # Get enhanced data
                        advanced_data = await self.enhancer.get_enhanced_signal_data(
                            symbol, signal_side, confidence
                        )
                        confidence = advanced_data['final_confidence']
                    
                    # 7. Risk Yönetimi - GERÇEK ZAMANLI FİYAT KULLAN
                    atr = float(last_row.get('atr', current_price * 0.01).values[0])
                    
                    size_usd = self.risk_manager.calculate_position_size(
                        account_balance=1000,
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
                        "risk_ratio": abs(current_price - sl) / current_price * 100,
                        # Advanced data
                        "advanced": {
                            "whale_score": advanced_data['whale_score'] if advanced_data else 0,
                            "liq_score": advanced_data['liq_score'] if advanced_data else 0,
                            "sentiment_score": advanced_data['sentiment_score'] if advanced_data else 0,
                            "boost": advanced_data['boost'] if advanced_data else 0
                        } if self.use_advanced else None
                    }
                    
                    # Sinyali kaydet (cooldown için)
                    self._record_signal(symbol, signal_side)
                    
                    signals.append(signal_data)
                    
                    # Enhanced logging
                    if advanced_data:
                        logger.info(f"🚨 SIGNAL: {symbol} {signal_side} @ ${current_price:,.2f} ({confidence:.2f}) "
                                    f"[W:{advanced_data['whale_score']:.2f} L:{advanced_data['liq_score']:.2f} S:{advanced_data['sentiment_score']:.2f}]")
                    else:
                        logger.info(f"🚨 SIGNAL: {symbol} {signal_side} @ ${current_price:,.2f} ({confidence:.2f})")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                
        return signals

    async def close(self):
        """Cleanup."""
        if self.enhancer:
            await self.enhancer.close()
