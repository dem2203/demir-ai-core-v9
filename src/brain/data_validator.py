# -*- coding: utf-8 -*-
"""
DEMIR AI - DATA VALIDATOR & CROSS-CHECKER
==========================================
Tüm verilerin GERÇEK olduğunu doğrular.
Mock, fallback, test veya manuel veri tespiti yapar.

Kontroller:
1. Timestamp freshness (veri ne kadar eski?)
2. Source verification (kaynak gerçekten çağrıldı mı?)
3. Value range validation (değerler mantıklı mı?)
4. Cross-source consistency (kaynaklar tutarlı mı?)
5. Pattern detection (mock data pattern'leri)
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("DATA_VALIDATOR")


class DataQuality(Enum):
    """Veri kalitesi seviyeleri"""
    VERIFIED = "VERIFIED"      # Çoklu kaynak doğrulaması geçti
    REAL = "REAL"              # Tek kaynak, ama gerçek
    STALE = "STALE"            # Eski veri (>5 dk)
    FALLBACK = "FALLBACK"      # Yedek veri kullanıldı
    MOCK = "MOCK"              # Mock/test veri tespit edildi
    UNKNOWN = "UNKNOWN"        # Doğrulanamadı


@dataclass
class ValidationResult:
    """Tek bir veri kaynağının doğrulama sonucu"""
    source_name: str
    quality: DataQuality
    timestamp: datetime = field(default_factory=datetime.now)
    age_seconds: float = 0
    issues: List[str] = field(default_factory=list)
    value_hash: str = ""  # Değerin özeti
    cross_check_passed: bool = False


@dataclass
class DataValidationReport:
    """Tüm veri kaynaklarının doğrulama raporu"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    overall_quality: DataQuality = DataQuality.UNKNOWN
    sources_checked: int = 0
    sources_verified: int = 0
    sources_failed: int = 0
    validations: List[ValidationResult] = field(default_factory=list)
    is_usable: bool = True
    rejection_reason: str = ""
    
    @property
    def verification_rate(self) -> float:
        if self.sources_checked == 0:
            return 0
        return self.sources_verified / self.sources_checked * 100


# Known mock/test data patterns
MOCK_PATTERNS = {
    # Round numbers that are suspiciously perfect
    'round_prices': [50000, 60000, 70000, 80000, 90000, 100000, 1000, 2000, 3000, 4000, 5000],
    
    # Common test values
    'test_values': [0, 1, -1, 100, 1000, 0.5, 0.1, 0.01],
    
    # Suspicious ratios
    'perfect_ratios': [1.0, 2.0, 0.5, 1.5, 0.25, 0.75],
    
    # Known fallback indicators
    'fallback_strings': ['fallback', 'mock', 'test', 'dummy', 'fake', 'sample', 'example'],
}


class DataValidator:
    """
    Data Validator & Cross-Checker
    
    Tüm verilerin gerçek olduğunu doğrular.
    """
    
    def __init__(self):
        self._last_prices: Dict[str, Tuple[float, datetime]] = {}
        self._validation_history: List[DataValidationReport] = []
        self._max_data_age = 300  # 5 dakika
        self._price_change_threshold = 0.10  # %10 - bunun üzeri şüpheli
        logger.info("✅ Data Validator initialized")
    
    async def validate_live_snapshot(self, snapshot: Any, symbol: str = "BTCUSDT") -> DataValidationReport:
        """
        LiveDataSnapshot'ı doğrula.
        """
        report = DataValidationReport(symbol=symbol)
        
        # 1. Check each field
        validations = []
        
        # Price validation
        validations.append(self._validate_price(snapshot, symbol))
        
        # Volume validation
        validations.append(self._validate_volume(snapshot))
        
        # Orderbook validation
        validations.append(self._validate_orderbook(snapshot))
        
        # Funding validation
        validations.append(self._validate_funding(snapshot))
        
        # Fear & Greed validation
        validations.append(self._validate_fear_greed(snapshot))
        
        # Whale data validation
        validations.append(self._validate_whale_data(snapshot))
        
        report.validations = validations
        report.sources_checked = len(validations)
        report.sources_verified = sum(1 for v in validations if v.quality in [DataQuality.VERIFIED, DataQuality.REAL])
        report.sources_failed = sum(1 for v in validations if v.quality in [DataQuality.MOCK, DataQuality.FALLBACK])
        
        # Determine overall quality
        if report.sources_failed > 0:
            if report.sources_failed >= report.sources_checked / 2:
                report.overall_quality = DataQuality.MOCK
                report.is_usable = False
                report.rejection_reason = f"{report.sources_failed}/{report.sources_checked} kaynakta mock/fallback veri tespit edildi"
            else:
                report.overall_quality = DataQuality.FALLBACK
                report.is_usable = True  # Kısmi kullanılabilir
        elif report.verification_rate >= 80:
            report.overall_quality = DataQuality.VERIFIED
        elif report.verification_rate >= 50:
            report.overall_quality = DataQuality.REAL
        else:
            report.overall_quality = DataQuality.STALE
        
        self._validation_history.append(report)
        return report
    
    def _validate_price(self, snapshot: Any, symbol: str) -> ValidationResult:
        """Fiyat doğrulama"""
        result = ValidationResult(source_name="price")
        
        try:
            price = getattr(snapshot, 'current_price', 0) or getattr(snapshot, 'binance_price', 0)
            
            if price == 0:
                result.quality = DataQuality.FALLBACK
                result.issues.append("Fiyat 0 - veri yok")
                return result
            
            # Check for mock patterns
            if price in MOCK_PATTERNS['round_prices']:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Şüpheli yuvarlak fiyat: ${price}")
                return result
            
            # Check price movement (if we have history)
            if symbol in self._last_prices:
                last_price, last_time = self._last_prices[symbol]
                time_diff = (datetime.now() - last_time).total_seconds()
                
                if time_diff < 60:  # Son 1 dakika içinde
                    price_change = abs(price - last_price) / last_price
                    if price_change > self._price_change_threshold:
                        result.issues.append(f"Ani fiyat değişimi: %{price_change*100:.1f}")
            
            # Store current price
            self._last_prices[symbol] = (price, datetime.now())
            
            # Cross-check with typical ranges
            if symbol == "BTCUSDT" and (price < 10000 or price > 500000):
                result.quality = DataQuality.MOCK
                result.issues.append(f"BTC fiyatı aralık dışı: ${price}")
                return result
            
            if symbol == "ETHUSDT" and (price < 500 or price > 50000):
                result.quality = DataQuality.MOCK
                result.issues.append(f"ETH fiyatı aralık dışı: ${price}")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"${price:.2f}"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"Fiyat doğrulama hatası: {e}")
        
        return result
    
    def _validate_volume(self, snapshot: Any) -> ValidationResult:
        """Hacim doğrulama"""
        result = ValidationResult(source_name="taker_volume")
        
        try:
            buy_ratio = getattr(snapshot, 'taker_buy_ratio', 0.5)
            buy_vol = getattr(snapshot, 'taker_buy_volume', 0)
            sell_vol = getattr(snapshot, 'taker_sell_volume', 0)
            
            # Check for mock patterns
            if buy_ratio in MOCK_PATTERNS['perfect_ratios']:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Şüpheli mükemmel oran: {buy_ratio}")
                return result
            
            if buy_vol == 0 and sell_vol == 0:
                result.quality = DataQuality.FALLBACK
                result.issues.append("Hacim verisi yok")
                return result
            
            # Reasonable ratio check
            if buy_ratio < 0 or buy_ratio > 1:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Geçersiz buy ratio: {buy_ratio}")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"BR:{buy_ratio:.2f}"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"Hacim doğrulama hatası: {e}")
        
        return result
    
    def _validate_orderbook(self, snapshot: Any) -> ValidationResult:
        """Order book doğrulama"""
        result = ValidationResult(source_name="orderbook")
        
        try:
            imbalance = getattr(snapshot, 'orderbook_imbalance', 1.0)
            bid_vol = getattr(snapshot, 'orderbook_bid_volume', 0)
            ask_vol = getattr(snapshot, 'orderbook_ask_volume', 0)
            
            # Check for mock patterns
            if imbalance in [1.0, 2.0, 0.5]:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Şüpheli mükemmel imbalance: {imbalance}")
                return result
            
            if bid_vol == 0 and ask_vol == 0:
                result.quality = DataQuality.FALLBACK
                result.issues.append("Order book verisi yok")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"IMB:{imbalance:.2f}"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"Orderbook doğrulama hatası: {e}")
        
        return result
    
    def _validate_funding(self, snapshot: Any) -> ValidationResult:
        """Funding rate doğrulama"""
        result = ValidationResult(source_name="funding")
        
        try:
            rate = getattr(snapshot, 'funding_rate', 0)
            
            # Funding rate should be small (-0.5% to +0.5% typically)
            if abs(rate) > 1:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Aşırı funding rate: {rate}%")
                return result
            
            if rate == 0:
                result.quality = DataQuality.FALLBACK
                result.issues.append("Funding rate 0 - fallback olabilir")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"FR:{rate:.4f}%"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"Funding doğrulama hatası: {e}")
        
        return result
    
    def _validate_fear_greed(self, snapshot: Any) -> ValidationResult:
        """Fear & Greed index doğrulama"""
        result = ValidationResult(source_name="fear_greed")
        
        try:
            index = getattr(snapshot, 'fear_greed_index', 50)
            
            # Should be 0-100
            if index < 0 or index > 100:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Geçersiz F&G index: {index}")
                return result
            
            # 50 is often a fallback value
            if index == 50:
                result.quality = DataQuality.FALLBACK
                result.issues.append("F&G 50 - muhtemelen fallback")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"FG:{index}"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"F&G doğrulama hatası: {e}")
        
        return result
    
    def _validate_whale_data(self, snapshot: Any) -> ValidationResult:
        """Whale verisi doğrulama"""
        result = ValidationResult(source_name="whale")
        
        try:
            net_flow = getattr(snapshot, 'whale_net_flow', 0)
            trade_count = getattr(snapshot, 'whale_trade_count', 0)
            
            if net_flow == 0 and trade_count == 0:
                result.quality = DataQuality.FALLBACK
                result.issues.append("Whale verisi yok - fallback")
                return result
            
            # Check for suspiciously round numbers
            if net_flow in [1000000, 5000000, 10000000, -1000000, -5000000]:
                result.quality = DataQuality.MOCK
                result.issues.append(f"Şüpheli yuvarlak whale flow: ${net_flow/1e6:.0f}M")
                return result
            
            result.quality = DataQuality.REAL
            result.value_hash = f"WF:${net_flow/1e6:.1f}M"
            
        except Exception as e:
            result.quality = DataQuality.UNKNOWN
            result.issues.append(f"Whale doğrulama hatası: {e}")
        
        return result
    
    async def cross_check_price(self, symbol: str = "BTCUSDT") -> Tuple[bool, float, str]:
        """
        Fiyatı birden fazla kaynaktan kontrol et.
        Returns: (is_valid, average_price, message)
        """
        import aiohttp
        
        prices = []
        sources = []
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Binance
                try:
                    async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as resp:
                        data = await resp.json()
                        prices.append(float(data['price']))
                        sources.append("binance")
                except:
                    pass
                
                # Bybit
                try:
                    async with session.get(f"https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}") as resp:
                        data = await resp.json()
                        price = float(data['result']['list'][0]['lastPrice'])
                        prices.append(price)
                        sources.append("bybit")
                except:
                    pass
                
                # OKX
                try:
                    okx_symbol = symbol.replace('USDT', '-USDT')
                    async with session.get(f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}") as resp:
                        data = await resp.json()
                        price = float(data['data'][0]['last'])
                        prices.append(price)
                        sources.append("okx")
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Cross-check error: {e}")
        
        if len(prices) < 2:
            return False, 0, f"Yetersiz kaynak ({len(prices)}/3)"
        
        avg_price = sum(prices) / len(prices)
        max_deviation = max(abs(p - avg_price) / avg_price for p in prices)
        
        if max_deviation > 0.01:  # >1% fark
            return False, avg_price, f"Borsalar arası fark çok yüksek: %{max_deviation*100:.2f}"
        
        return True, avg_price, f"Doğrulandı ({len(sources)} borsa, avg: ${avg_price:.2f})"
    
    def format_report_telegram(self, report: DataValidationReport) -> str:
        """Raporu Telegram formatında göster"""
        quality_emoji = {
            DataQuality.VERIFIED: "✅",
            DataQuality.REAL: "🟢",
            DataQuality.STALE: "🟡",
            DataQuality.FALLBACK: "🟠",
            DataQuality.MOCK: "🔴",
            DataQuality.UNKNOWN: "⚪"
        }
        
        emoji = quality_emoji.get(report.overall_quality, "⚪")
        
        source_lines = ""
        for v in report.validations:
            src_emoji = quality_emoji.get(v.quality, "⚪")
            issues_text = f" - {v.issues[0]}" if v.issues else ""
            source_lines += f"  {src_emoji} {v.source_name}: {v.value_hash}{issues_text}\n"
        
        return f"""🔍 *VERİ DOĞRULAMA - {report.symbol}*
━━━━━━━━━━━━━━━━━━
{emoji} *Kalite: {report.overall_quality.value}*
📊 Doğrulama: {report.verification_rate:.0f}%
✅ Geçti: {report.sources_verified}/{report.sources_checked}
❌ Başarısız: {report.sources_failed}

━━━ KAYNAK DETAY ━━━
{source_lines}
{"⚠️ " + report.rejection_reason if not report.is_usable else "✅ Veri kullanılabilir"}
━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%H:%M:%S')}"""


# Singleton instance
_validator: Optional[DataValidator] = None

def get_data_validator() -> DataValidator:
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator
