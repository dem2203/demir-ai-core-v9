import logging
import pandas as pd
from typing import Dict, List, Optional
from src.brain.feature_engineering import FeatureEngineer
from src.validation.validator import SignalValidator

logger = logging.getLogger("MARKET_ANALYZER")

class MarketAnalyzer:
    """
    DEMIR AI V9.5 - STRATEGY ENGINE
    
    Çoklu Strateji Karar Mekanizması:
    1. Trend Takibi (Ichimoku + ADX)
    2. Momentum (RSI + MACD)
    3. Volatilite (Bollinger Sıkışması)
    
    Tüm bu faktörleri puanlar ve 60 puanın üzerindeyse işleme girer.
    """

    def __init__(self):
        # Modeller şimdilik kapalı (Matematik motoru devrede)
        pass
        
    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        """
        Ham veriyi işler ve AL/SAT/BEKLE kararı üretir.
        """
        # 1. Öznitelik Mühendisliği (Matematiksel Analiz)
        df = FeatureEngineer.process_data(raw_data)
        if df is None or df.empty:
            logger.warning(f"Insufficient data for analysis: {symbol}")
            return None

        # Son anlık veriyi al (Canlı Karar)
        current_data = df.iloc[-1]
        
        # 2. Strateji Motorunu Çalıştır
        signal = self._advanced_math_strategy(symbol, current_data)
        
        # 3. Sinyal Doğrulama (Validation Layer)
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        
        return None

    def _advanced_math_strategy(self, symbol: str, row: pd.Series) -> Dict:
        """
        Çok Faktörlü Puanlama Sistemi.
        """
        score = 0
        reasons = []
        
        # --- A. TREND ANALİZİ (Ichimoku & ADX) ---
        # Fiyat Bulutun Üstünde mi? (Bullish)
        if row['close'] > row['span_a'] and row['close'] > row['span_b']:
            score += 20
            reasons.append("Price above Cloud")
            
        # Trend Güçlü mü? (ADX > 25)
        if row['adx'] > 25:
            score += 10
            reasons.append("Strong Trend")

        # --- B. MOMENTUM ANALİZİ (MACD & RSI) ---
        # MACD Al Sinyali (Line > Signal)
        if row['macd'] > row['macd_signal']:
            score += 15
            reasons.append("MACD Cross Up")
            
        # RSI Kontrolü
        if 45 < row['rsi'] < 70: # Sağlıklı yükseliş bölgesi
            score += 10
        elif row['rsi'] < 30: # Aşırı Satım (Diğer şartlar uyarsa tepki alımı)
            score += 30 
            reasons.append("RSI Oversold")

        # --- C. VOLATİLİTE (Bollinger) ---
        # Fiyat Üst Bandı zorluyor mu?
        # (Basit hesap: close > sma + (2*std) * 0.95) - Yaklaşık hesap
        # Burada z-score kullanabiliriz. Z-Score > 1 ise fiyat yükseliyor demektir.
        if row['z_score'] > 1.0:
            score += 10

        # --- KARAR MEKANİZMASI ---
        decision = "NEUTRAL"
        confidence = 0.0
        
        # Eğer Puan 50'den yüksekse AL
        if score >= 50:
            decision = "BUY"
            confidence = min(score, 99.0) # Max %99
        
        # Eğer Puan çok düşükse SAT (Short)
        elif score <= 10:
            # Short mantığı (Tersi)
            if row['close'] < row['span_a'] and row['macd'] < row['macd_signal']:
                decision = "SELL"
                confidence = 80.0

        # Stop Loss ve Take Profit Hesaplama (ATR Bazlı)
        price = row['close']
        atr = row['atr']
        
        if decision == "BUY":
            tp = price + (atr * 3.0) # Hedef: 3x ATR yukarı
            sl = price - (atr * 1.5) # Stop: 1.5x ATR aşağı
        elif decision == "SELL":
            tp = price - (atr * 3.0)
            sl = price + (atr * 1.5)
        else:
            tp = 0.0
            sl = 0.0
            confidence = 0.0

        # Loglama (Analizin nedenini görmek için)
        if decision != "NEUTRAL":
            logger.info(f"STRATEGY HIT ({symbol}): Score {score} -> {reasons}")

        return {
            "symbol": symbol,
            "side": decision,
            "entry_price": price,
            "tp_price": tp,
            "sl_price": sl,
            "confidence": confidence,
            "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
        }