# -*- coding: utf-8 -*-
"""
DEMIR AI - TECHNICAL ANALYSIS SCANNER
Phase 131: 15 Dakikalık Teknik Tarama

ÖZELLİKLER:
1. 4 coin taraması (BTC, ETH, LTC, SOL)
2. Tüm teknik modüller (%70+ olanları listele)
3. LONG/SHORT yönü
4. Entry, SL, TP seviyeleri
"""
import logging
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("TECHNICAL_SCANNER")


@dataclass
class ModuleSignal:
    """Tek modül sinyali."""
    name: str
    direction: str  # LONG, SHORT, NEUTRAL
    confidence: float  # 0-100
    reason: str = ""


@dataclass 
class TechnicalScan:
    """Teknik tarama sonucu."""
    symbol: str
    direction: str  # LONG, SHORT, NEUTRAL
    overall_confidence: float
    strong_modules: List[ModuleSignal]  # %70+ modüller
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)


class TechnicalScanner:
    """
    TEKNİK ANALİZ TARAYICISI
    
    Tüm teknik modülleri tarar ve %70+ güven veren sinyalleri toplar.
    Entry, SL, TP hesaplar.
    """
    
    SYMBOLS = ['BTCUSDT', 'ETHUSDT']
    CONFIDENCE_THRESHOLD = 70  # Minimum %70 güven
    
    def __init__(self):
        logger.info("✅ Technical Scanner initialized")
    
    async def scan_all_coins(self) -> List[TechnicalScan]:
        """Tüm coinleri tara."""
        results = []
        
        for symbol in self.SYMBOLS:
            try:
                scan = await self.scan_coin(symbol)
                if scan and scan.overall_confidence >= 55:  # Min %55 overall
                    results.append(scan)
            except Exception as e:
                logger.warning(f"Scan failed for {symbol}: {e}")
        
        return results
    
    async def scan_coin(self, symbol: str) -> Optional[TechnicalScan]:
        """Tek coin taraması."""
        try:
            # Mevcut fiyatı al
            price = self._get_current_price(symbol)
            if not price:
                return None
            
            # Tüm modülleri çalıştır
            signals = await self._run_all_modules(symbol)
            
            # %70+ modülleri filtrele
            strong_long = [s for s in signals if s.confidence >= self.CONFIDENCE_THRESHOLD and s.direction == 'LONG']
            strong_short = [s for s in signals if s.confidence >= self.CONFIDENCE_THRESHOLD and s.direction == 'SHORT']
            
            # Yön belirle
            if len(strong_long) > len(strong_short) and len(strong_long) >= 2:
                direction = 'LONG'
                strong_modules = strong_long
            elif len(strong_short) > len(strong_long) and len(strong_short) >= 2:
                direction = 'SHORT'
                strong_modules = strong_short
            else:
                direction = 'NEUTRAL'
                strong_modules = []
            
            if direction == 'NEUTRAL':
                return None
            
            # Ortalama güven
            avg_confidence = sum(s.confidence for s in strong_modules) / len(strong_modules) if strong_modules else 0
            
            # Entry, SL, TP hesapla
            entry, sl, tp, rr = self._calculate_levels(symbol, price, direction)
            
            return TechnicalScan(
                symbol=symbol,
                direction=direction,
                overall_confidence=avg_confidence,
                strong_modules=strong_modules,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                rr_ratio=rr
            )
            
        except Exception as e:
            logger.error(f"Coin scan failed: {e}")
            return None
    
    async def _run_all_modules(self, symbol: str) -> List[ModuleSignal]:
        """Tüm teknik modülleri çalıştır."""
        signals = []
        
        # 1. RSI
        try:
            from src.brain.technical_analyzer import TechnicalAnalyzer
            ta = TechnicalAnalyzer()
            rsi = ta.get_rsi(symbol)
            if rsi:
                if rsi < 30:
                    signals.append(ModuleSignal("RSI", "LONG", 75, f"Aşırı satım ({rsi:.0f})"))
                elif rsi > 70:
                    signals.append(ModuleSignal("RSI", "SHORT", 75, f"Aşırı alım ({rsi:.0f})"))
                else:
                    signals.append(ModuleSignal("RSI", "NEUTRAL", 40, f"Nötr ({rsi:.0f})"))
        except:
            pass
        
        # 2. MACD
        try:
            macd = ta.get_macd(symbol)
            if macd:
                if macd.get('histogram', 0) > 0:
                    conf = min(80, 60 + abs(macd['histogram']) * 5)
                    signals.append(ModuleSignal("MACD", "LONG", conf, "Pozitif histogram"))
                else:
                    conf = min(80, 60 + abs(macd['histogram']) * 5)
                    signals.append(ModuleSignal("MACD", "SHORT", conf, "Negatif histogram"))
        except:
            pass
        
        # 3. Bollinger Squeeze
        try:
            from src.brain.bollinger_squeeze import BollingerSqueezeDetector
            bb = BollingerSqueezeDetector()
            result = bb.detect_squeeze(symbol)
            if result.get('squeeze_detected'):
                direction = result.get('direction', 'NEUTRAL')
                conf = result.get('confidence', 60)
                signals.append(ModuleSignal("Bollinger Squeeze", direction, conf, "Sıkışma tespit"))
        except:
            pass
        
        # 4. Volume Spike
        try:
            from src.brain.volume_spike import detect_volume_spike
            spike = detect_volume_spike(symbol)
            if spike.get('spike_detected'):
                direction = spike.get('direction', 'NEUTRAL')
                conf = spike.get('confidence', 60)
                signals.append(ModuleSignal("Volume Spike", direction, conf, f"{spike.get('spike_strength', 1):.1f}x hacim"))
        except:
            pass
        
        # 5. SMC (Smart Money Concept)
        try:
            from src.brain.smc_analyzer import SMCAnalyzer
            smc = SMCAnalyzer()
            result = smc.analyze(symbol)
            if result.get('signal') != 'NEUTRAL':
                signals.append(ModuleSignal("SMC", result['signal'], result.get('confidence', 60), result.get('reason', '')))
        except:
            pass
        
        # 6. Fibonacci
        try:
            from src.brain.fibonacci_analyzer import FibonacciAnalyzer
            fib = FibonacciAnalyzer()
            result = fib.analyze(symbol)
            if result.get('signal') != 'NEUTRAL':
                signals.append(ModuleSignal("Fibonacci", result['signal'], result.get('confidence', 60), result.get('level', '')))
        except:
            pass
        
        # 7. Pivot Points
        try:
            from src.brain.pivot_points import PivotPointsAnalyzer
            pivot = PivotPointsAnalyzer()
            result = pivot.analyze(symbol)
            if result.get('signal') != 'NEUTRAL':
                signals.append(ModuleSignal("Pivot Points", result['signal'], result.get('confidence', 60), result.get('level', '')))
        except:
            pass
        
        # 8. MTF (Multi-Timeframe)
        try:
            from src.brain.mtf_analyzer import MTFAnalyzer
            mtf = MTFAnalyzer()
            result = mtf.analyze(symbol)
            if result.get('direction') != 'NEUTRAL':
                signals.append(ModuleSignal("MTF Analiz", result['direction'], result.get('confidence', 60), result.get('alignment', '')))
        except:
            pass
        
        # 9. Pattern Engine (Wyckoff, etc.)
        try:
            from src.brain.pattern_engine import PatternEngine
            pe = PatternEngine()
            patterns = pe.detect_patterns(symbol)
            for p in patterns[:2]:  # Max 2 pattern
                if p.get('confidence', 0) >= 60:
                    signals.append(ModuleSignal(f"Pattern: {p['name']}", p['direction'], p['confidence'], p.get('reason', '')))
        except:
            pass
        
        # 10. Candle Patterns
        try:
            from src.brain.candle_patterns import CandlePatterns
            cp = CandlePatterns()
            result = cp.analyze(symbol)
            if result.get('pattern'):
                signals.append(ModuleSignal(f"Mum: {result['pattern']}", result['direction'], result.get('confidence', 65), ""))
        except:
            pass
        
        # 11. Funding Rate
        try:
            from src.brain.coinglass_funding import get_funding_rate
            funding = get_funding_rate(symbol.replace('USDT', ''))
            if funding.get('available'):
                rate = funding.get('funding_rate', 0)
                if rate > 0.05:
                    signals.append(ModuleSignal("Funding Rate", "SHORT", 70, f"Yüksek pozitif ({rate:.3f})"))
                elif rate < -0.03:
                    signals.append(ModuleSignal("Funding Rate", "LONG", 70, f"Negatif funding ({rate:.3f})"))
        except:
            pass
        
        # 12. L/S Ratio
        try:
            from src.brain.coinglass_ls_ratio import get_ls_ratio
            ls = get_ls_ratio(symbol.replace('USDT', ''))
            if ls.get('available'):
                ratio = ls.get('ratio', 1)
                if ratio > 2:
                    signals.append(ModuleSignal("L/S Ratio", "SHORT", 65, f"Long kalabalık ({ratio:.2f})"))
                elif ratio < 0.5:
                    signals.append(ModuleSignal("L/S Ratio", "LONG", 65, f"Short kalabalık ({ratio:.2f})"))
        except:
            pass
        
        # 13. OI Delta
        try:
            from src.brain.coinglass_oi_delta import get_oi_delta
            oi = get_oi_delta(symbol)
            if oi.get('available') and oi.get('velocity') != 'STABLE':
                signals.append(ModuleSignal("OI Delta", oi['direction'], oi.get('confidence', 60), f"{oi['velocity']}"))
        except:
            pass
        
        # 14. Liquidation Risk
        try:
            from src.brain.coinglass_liquidation import get_liquidation_levels
            liq = get_liquidation_levels(symbol.replace('USDT', ''))
            if liq.get('cascade_risk') == 'HIGH':
                signals.append(ModuleSignal("Liquidation", liq['direction'], 75, "Cascade riski YÜKSEK"))
        except:
            pass
        
        return signals
    
    def _get_current_price(self, symbol: str) -> float:
        """Mevcut fiyatı al."""
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={'symbol': symbol},
                timeout=5
            )
            return float(resp.json()['price'])
        except:
            return 0
    
    def _calculate_levels(self, symbol: str, price: float, direction: str) -> Tuple[float, float, float, float]:
        """Entry, SL, TP hesapla."""
        # ATR bazlı hesaplama yerine basit yüzde kullan
        if 'BTC' in symbol:
            sl_pct = 0.015  # %1.5
            tp_pct = 0.03   # %3
        elif 'ETH' in symbol:
            sl_pct = 0.02   # %2
            tp_pct = 0.04   # %4
        else:
            sl_pct = 0.025  # %2.5
            tp_pct = 0.05   # %5
        
        entry = price
        
        if direction == 'LONG':
            sl = price * (1 - sl_pct)
            tp = price * (1 + tp_pct)
        else:  # SHORT
            sl = price * (1 + sl_pct)
            tp = price * (1 - tp_pct)
        
        rr = tp_pct / sl_pct  # Risk/Reward
        
        return entry, sl, tp, rr
    
    def format_scan_message(self, scan: TechnicalScan) -> str:
        """Telegram mesajı formatla."""
        direction_emoji = "🟢📈" if scan.direction == "LONG" else "🔴📉"
        
        # Modül listesi
        module_list = ""
        for m in scan.strong_modules[:6]:  # Max 6 modül göster
            mod_emoji = "🟢" if m.direction == "LONG" else "🔴"
            module_list += f"  {mod_emoji} {m.name}: %{m.confidence:.0f}\n"
            if m.reason:
                module_list += f"     └─ {m.reason}\n"
        
        msg = f"""
📊 TEKNİK TARAMA - {scan.symbol}
━━━━━━━━━━━━━━━━━━━━━━
{direction_emoji} SİNYAL: {scan.direction}
📈 Güven: %{scan.overall_confidence:.0f}
━━━━━━━━━━━━━━━━━━━━━━
🎯 %70+ ONAY VEREN MODÜLLER:
{module_list}
━━━━━━━━━━━━━━━━━━━━━━
💰 Entry: ${scan.entry_price:,.2f}
🛡️ Stop Loss: ${scan.stop_loss:,.2f}
🎯 Take Profit: ${scan.take_profit:,.2f}
📊 R:R = 1:{scan.rr_ratio:.1f}
━━━━━━━━━━━━━━━━━━━━━━
⏰ {scan.timestamp.strftime('%d.%m.%Y %H:%M')}
"""
        return msg


# Global instance
_scanner = None

def get_technical_scanner() -> TechnicalScanner:
    """Get or create scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = TechnicalScanner()
    return _scanner


async def run_technical_scan() -> List[str]:
    """Quick scan - returns formatted messages."""
    scanner = get_technical_scanner()
    scans = await scanner.scan_all_coins()
    return [scanner.format_scan_message(s) for s in scans]
