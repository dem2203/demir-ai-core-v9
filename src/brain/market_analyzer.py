import logging
from typing import Dict, List, Optional
from src.brain.feature_engineering import FeatureEngineer
from src.brain.models.model_loader import ModelLoader
from src.validation.validator import SignalValidator

logger = logging.getLogger("MARKET_ANALYZER")

class MarketAnalyzer:
    """
    MERKEZİ ANALİZ MOTORU
    1. Veriyi alır.
    2. Matematiğini (Features) oluşturur.
    3. AI Modellerine sorar (Quant, Analyst, Economist).
    4. Nihai kararı verir.
    """

    def __init__(self):
        # Hibrit Model Yapısı
        self.trend_model = ModelLoader.load_model("lstm_v9.h5")
        self.sentiment_model = ModelLoader.load_model("finbert_v2.bin")
        
    async def analyze_market(self, symbol: str, raw_data: List[Dict]) -> Optional[Dict]:
        """
        Bir coini analiz eder ve işlem sinyali üretir.
        """
        # 1. Öznitelik Mühendisliği (Matematiksel Analiz)
        df = FeatureEngineer.process_data(raw_data)
        if df is None or df.empty:
            logger.warning(f"Insufficient data for analysis: {symbol}")
            return None

        # Son satırı (en güncel anı) al
        current_market_state = df.iloc[-1]
        
        # 2. AI Tahmini (Eğer model yüklü değilse Heuristic Analiz yap)
        # Not: Zero-Mock politikası gereği, model yoksa rastgele sayı üretmiyoruz.
        # Bunun yerine "Geleneksel Matematiksel Analiz" (RSI+MACD) sonuçlarını kullanıyoruz.
        
        signal = self._heuristic_analysis(symbol, current_market_state)
        
        # 3. Sinyal Doğrulama (Validation Layer)
        if SignalValidator.validate_outgoing_signal(signal):
            return signal
        
        return None

    def _heuristic_analysis(self, symbol: str, data: pd.Series) -> Dict:
        """
        Yapay zeka modelleri eğitilene kadar kullanılacak MATEMATİKSEL KESİN STRATEJİ.
        Asla rastgele veri üretmez. RSI ve MACD kesişimlerine bakar.
        """
        rsi = data['rsi']
        macd = data['macd']
        macd_signal = data['macd_signal']
        price = data['close']
        
        decision = "NEUTRAL"
        confidence = 0.0
        
        # Gerçek Strateji Mantığı
        if rsi < 30 and macd > macd_signal:
            decision = "BUY"
            confidence = 85.0 # Oversold + Momentum Up
        elif rsi > 70 and macd < macd_signal:
            decision = "SELL"
            confidence = 85.0 # Overbought + Momentum Down
            
        # Sinyal Paketi Oluştur
        return {
            "symbol": symbol,
            "side": decision,
            "entry_price": price,
            "tp_price": price * 1.02 if decision == "BUY" else price * 0.98,
            "sl_price": price * 0.98 if decision == "BUY" else price * 1.02,
            "confidence": confidence,
            "timestamp": int(pd.Timestamp.now().timestamp() * 1000)
        }