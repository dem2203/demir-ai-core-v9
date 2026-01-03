# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - SIGNAL VALIDATOR
================================
Sinyallerin doğruluğunu takip eder ve performans raporu oluşturur.

HER SİNYAL İÇİN:
1. Giriş fiyatı kaydet
2. 1 saat sonra fiyat kontrol et
3. 4 saat sonra fiyat kontrol et
4. Yön doğru muydu? hesapla

Author: DEMIR AI Team
Date: 2026-01-03
"""
import json
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("SIGNAL_VALIDATOR")


@dataclass
class SignalRecord:
    """Bir sinyalin kaydı"""
    id: str
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: int
    timestamp: str
    
    # Sonuç verileri (sonradan dolacak)
    price_1h: Optional[float] = None
    price_4h: Optional[float] = None
    result_1h: Optional[str] = None  # "WIN", "LOSS", "PENDING"
    result_4h: Optional[str] = None
    profit_pct_1h: Optional[float] = None
    profit_pct_4h: Optional[float] = None
    checked_1h: bool = False
    checked_4h: bool = False


class SignalValidator:
    """
    Sinyal Doğrulama Sistemi
    
    Her sinyali kaydeder, belirli aralıklarla fiyat kontrol eder,
    doğruluk hesaplar ve rapor oluşturur.
    """
    
    DB_PATH = Path("src/v10/storage/signal_validation.json")
    
    def __init__(self):
        self.signals: List[SignalRecord] = []
        self._load()
        logger.info(f"📊 Signal Validator initialized with {len(self.signals)} records")
    
    def record_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int
    ) -> str:
        """Yeni bir sinyali kaydet"""
        signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        record = SignalRecord(
            id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
        self.signals.append(record)
        self._save()
        
        logger.info(f"📝 Signal recorded: {signal_id} | {direction} @ ${entry_price:,.2f}")
        return signal_id
    
    async def check_pending_signals(self) -> Dict:
        """Bekleyen sinyallerin fiyatlarını kontrol et"""
        now = datetime.now()
        checked_count = 0
        results = {"1h": [], "4h": []}
        
        for signal in self.signals:
            signal_time = datetime.fromisoformat(signal.timestamp)
            hours_passed = (now - signal_time).total_seconds() / 3600
            
            # 1 saat kontrolü
            if not signal.checked_1h and hours_passed >= 1:
                current_price = await self._get_current_price(signal.symbol)
                if current_price:
                    signal.price_1h = current_price
                    signal.checked_1h = True
                    
                    # Sonuç hesapla
                    if signal.direction == "BUY":
                        profit_pct = ((current_price - signal.entry_price) / signal.entry_price) * 100
                        signal.result_1h = "WIN" if current_price > signal.entry_price else "LOSS"
                    else:  # SELL
                        profit_pct = ((signal.entry_price - current_price) / signal.entry_price) * 100
                        signal.result_1h = "WIN" if current_price < signal.entry_price else "LOSS"
                    
                    signal.profit_pct_1h = profit_pct
                    results["1h"].append({
                        "symbol": signal.symbol,
                        "result": signal.result_1h,
                        "profit": profit_pct
                    })
                    checked_count += 1
            
            # 4 saat kontrolü
            if not signal.checked_4h and hours_passed >= 4:
                current_price = await self._get_current_price(signal.symbol)
                if current_price:
                    signal.price_4h = current_price
                    signal.checked_4h = True
                    
                    # Sonuç hesapla
                    if signal.direction == "BUY":
                        profit_pct = ((current_price - signal.entry_price) / signal.entry_price) * 100
                        signal.result_4h = "WIN" if current_price > signal.entry_price else "LOSS"
                    else:  # SELL
                        profit_pct = ((signal.entry_price - current_price) / signal.entry_price) * 100
                        signal.result_4h = "WIN" if current_price < signal.entry_price else "LOSS"
                    
                    signal.profit_pct_4h = profit_pct
                    results["4h"].append({
                        "symbol": signal.symbol,
                        "result": signal.result_4h,
                        "profit": profit_pct
                    })
                    checked_count += 1
        
        if checked_count > 0:
            self._save()
            logger.info(f"✅ Checked {checked_count} pending signals")
        
        return results
    
    def get_accuracy_report(self, hours: int = 24) -> Dict:
        """Son N saat için doğruluk raporu oluştur"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_signals = [
            s for s in self.signals 
            if datetime.fromisoformat(s.timestamp) > cutoff
        ]
        
        if not recent_signals:
            return {
                "total": 0,
                "win_rate_1h": 0,
                "win_rate_4h": 0,
                "by_symbol": {}
            }
        
        # 1 saat sonuçları
        checked_1h = [s for s in recent_signals if s.checked_1h]
        wins_1h = sum(1 for s in checked_1h if s.result_1h == "WIN")
        win_rate_1h = (wins_1h / len(checked_1h) * 100) if checked_1h else 0
        
        # 4 saat sonuçları
        checked_4h = [s for s in recent_signals if s.checked_4h]
        wins_4h = sum(1 for s in checked_4h if s.result_4h == "WIN")
        win_rate_4h = (wins_4h / len(checked_4h) * 100) if checked_4h else 0
        
        # Symbol bazlı analiz
        by_symbol = {}
        for symbol in set(s.symbol for s in recent_signals):
            sym_signals = [s for s in recent_signals if s.symbol == symbol]
            sym_checked = [s for s in sym_signals if s.checked_4h]
            sym_wins = sum(1 for s in sym_checked if s.result_4h == "WIN")
            by_symbol[symbol] = {
                "total": len(sym_signals),
                "checked": len(sym_checked),
                "wins": sym_wins,
                "win_rate": (sym_wins / len(sym_checked) * 100) if sym_checked else 0
            }
        
        return {
            "total": len(recent_signals),
            "checked_1h": len(checked_1h),
            "wins_1h": wins_1h,
            "win_rate_1h": win_rate_1h,
            "checked_4h": len(checked_4h),
            "wins_4h": wins_4h,
            "win_rate_4h": win_rate_4h,
            "by_symbol": by_symbol
        }
    
    def get_last_n_accuracy(self, n: int = 10) -> float:
        """Son N sinyalin doğruluk oranını döndür"""
        checked = [s for s in self.signals if s.checked_4h]
        if len(checked) < n:
            return 0.0
        
        last_n = checked[-n:]
        wins = sum(1 for s in last_n if s.result_4h == "WIN")
        return wins / n * 100
    
    def format_telegram_report(self, hours: int = 24) -> str:
        """Telegram için formatlanmış rapor oluştur"""
        report = self.get_accuracy_report(hours)
        
        if report["total"] == 0:
            return "📊 Son 24 saatte sinyal kaydı yok."
        
        # Symbol bazlı en iyi performansı bul
        best_symbol = ""
        best_rate = 0
        for sym, data in report["by_symbol"].items():
            if data["checked"] >= 2 and data["win_rate"] > best_rate:
                best_rate = data["win_rate"]
                best_symbol = sym
        
        msg = f"""📊 SİNYAL DOĞRULAMA RAPORU
━━━━━━━━━━━━━━━━━━━━━━━

📈 Son {hours} Saat:
  Toplam Sinyal: {report['total']}
  
⏱️ 1 Saat Sonunda:
  Kontrol Edilen: {report['checked_1h']}
  Doğru Yön: {report['wins_1h']} ({report['win_rate_1h']:.0f}%)
  
⏱️ 4 Saat Sonunda:
  Kontrol Edilen: {report['checked_4h']}
  Doğru Yön: {report['wins_4h']} ({report['win_rate_4h']:.0f}%)

💰 Coin Bazlı Performans:"""
        
        for sym, data in report["by_symbol"].items():
            if data["checked"] > 0:
                emoji = "🟢" if data["win_rate"] >= 60 else "🟡" if data["win_rate"] >= 40 else "🔴"
                msg += f"\n  {emoji} {sym}: {data['wins']}/{data['checked']} ({data['win_rate']:.0f}%)"
        
        if best_symbol:
            msg += f"\n\n🏆 En İyi: {best_symbol} ({best_rate:.0f}%)"
        
        msg += f"\n\n⏰ Rapor: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        msg += "\n━━━ DEMIR AI v10 ━━━"
        
        return msg
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Binance'dan güncel fiyat al"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data["price"])
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {e}")
        return None
    
    def _load(self):
        """Veritabanını yükle"""
        try:
            if self.DB_PATH.exists():
                with open(self.DB_PATH, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalRecord(**s) for s in data]
        except Exception as e:
            logger.warning(f"Load error: {e}")
            self.signals = []
    
    def _save(self):
        """Veritabanını kaydet"""
        try:
            self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.DB_PATH, 'w') as f:
                json.dump([asdict(s) for s in self.signals], f, indent=2)
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def cleanup_old(self, keep_days: int = 7):
        """Eski kayıtları temizle"""
        cutoff = datetime.now() - timedelta(days=keep_days)
        before = len(self.signals)
        self.signals = [
            s for s in self.signals 
            if datetime.fromisoformat(s.timestamp) > cutoff
        ]
        removed = before - len(self.signals)
        if removed > 0:
            self._save()
            logger.info(f"🗑️ Removed {removed} old signal records")


# Singleton
_validator: Optional[SignalValidator] = None


def get_signal_validator() -> SignalValidator:
    """Get or create SignalValidator singleton"""
    global _validator
    if _validator is None:
        _validator = SignalValidator()
    return _validator


# Test
if __name__ == "__main__":
    async def test():
        validator = get_signal_validator()
        
        # Test kayıt
        validator.record_signal(
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=90000,
            stop_loss=88500,
            take_profit=93000,
            confidence=85
        )
        
        # Test rapor
        print(validator.format_telegram_report())
    
    asyncio.run(test())
