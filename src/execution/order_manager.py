import logging
from typing import Dict, Optional
from src.execution.risk_manager import RiskManager
from src.validation.validator import SignalValidator

logger = logging.getLogger("ORDER_MANAGER")

class OrderManager:
    """
    EXECUTION SNIPER
    Analizden gelen sinyali alır, Risk Yöneticisine onaylatır ve emre dönüştürür.
    """
    
    def __init__(self):
        self.risk_manager = RiskManager()

    async def prepare_order(self, signal: Dict, current_balance: float, atr_value: float) -> Optional[Dict]:
        """
        Sinyali alır, üzerine para yönetimi (SL/TP/Size) ekleyerek 
        borsaya gitmeye hazır, PROFESYONEL bir emir paketine dönüştürür.
        """
        symbol = signal.get('symbol')
        side = signal.get('side')
        entry = float(signal.get('entry_price'))
        confidence = float(signal.get('confidence', 0))

        # 1. Dinamik Stop Loss & Take Profit Hesapla (ATR ile)
        stops = self.risk_manager.calculate_dynamic_stops(entry, side, atr_value)
        sl_price = stops['sl']
        tp_price = stops['tp']

        # 2. Akıllı Pozisyon Büyüklüğü Hesapla (Kelly & Risk)
        position_size_usdt = self.risk_manager.calculate_position_size(
            balance=current_balance,
            entry_price=entry,
            sl_price=sl_price,
            ai_confidence=confidence
        )

        if position_size_usdt < 10: # Binance min emir tutarı (genelde 10$)
            logger.warning(f"Order Ignored: Position size (${position_size_usdt:.2f}) too small for exchange limits.")
            return None

        # 3. Nihai Emir Paketi
        final_order = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT", # Profesyoneller Market emri kullanmaz, Limit kullanır (Slippage koruması)
            "quantity": position_size_usdt / entry, # Coin adedi
            "price": entry,
            "params": {
                "stopLoss": sl_price,
                "takeProfit": tp_price,
            },
            "confidence": confidence,
            "metadata": "AI_V9_SUPERHUMAN_EXECUTION"
        }

        # 4. Son Güvenlik Kontrolü (Validator Layer)
        if SignalValidator.validate_outgoing_signal({
            "side": side, "entry_price": entry, "tp_price": tp_price, "sl_price": sl_price, "confidence": confidence
        }):
            logger.info(f"ORDER PREPARED: {side} {symbol} | Size: ${position_size_usdt:.2f} | R:R Ratio Protected")
            return final_order
        else:
            return None