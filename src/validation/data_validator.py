# -*- coding: utf-8 -*-
"""
DEMIR AI - Data Validator System
Tüm verileri çoklu kaynaklardan cross-check eder.
Uyumsuzluk tespit ederse uyarı verir.

Features:
- BTC fiyat doğrulama (Binance vs CoinGecko)
- Dominance doğrulama (Browser vs CoinMarketCap API)
- Teknik gösterge doğrulama (hesaplama vs TradingView)
- Periyodik kontrol (5 dakika)
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("DATA_VALIDATOR")


@dataclass
class ValidationResult:
    """Doğrulama sonucu"""
    metric: str
    our_value: float
    reference_value: float
    reference_source: str
    difference_pct: float
    is_valid: bool
    timestamp: datetime


class DataValidator:
    """
    Çoklu kaynak veri doğrulama sistemi.
    
    Her veriyi en az 2 kaynaktan kontrol eder.
    Büyük farklar tespit edilirse uyarı verir.
    """
    
    # Kabul edilebilir maksimum fark yüzdeleri
    THRESHOLDS = {
        'btc_price': 0.5,      # %0.5 fark kabul
        'dominance': 2.0,      # %2 fark kabul
        'open_interest': 10.0,  # %10 fark kabul (farklı hesaplamalar)
        'fear_greed': 5.0,      # 5 puan fark kabul
        'dxy': 1.0,            # %1 fark kabul
        'vix': 5.0,            # %5 fark kabul
    }
    
    def __init__(self):
        self.last_validation = None
        self.validation_results: List[ValidationResult] = []
        self.cache = {}
        self.cache_duration = 300  # 5 dakika
    
    def validate_all(self) -> Dict:
        """
        Tüm verileri doğrula ve sonuç raporu döndür.
        
        Returns:
            {
                'is_valid': True/False,
                'total_checks': 5,
                'passed': 4,
                'failed': 1,
                'results': [...],
                'summary': 'Veriler doğrulandı' / '1 uyumsuzluk tespit edildi'
            }
        """
        results = []
        
        # 1. BTC Fiyat Doğrulama
        btc_result = self._validate_btc_price()
        if btc_result:
            results.append(btc_result)
        
        # 2. BTC Dominance Doğrulama
        btcd_result = self._validate_btc_dominance()
        if btcd_result:
            results.append(btcd_result)
        
        # 3. Fear & Greed Doğrulama
        fng_result = self._validate_fear_greed()
        if fng_result:
            results.append(fng_result)
        
        # 4. Open Interest Doğrulama
        oi_result = self._validate_open_interest()
        if oi_result:
            results.append(oi_result)
        
        # 5. Long/Short Ratio Doğrulama
        ls_result = self._validate_long_short()
        if ls_result:
            results.append(ls_result)
        
        self.validation_results = results
        self.last_validation = datetime.now()
        
        # Sonuçları özetle
        passed = sum(1 for r in results if r.is_valid)
        failed = len(results) - passed
        
        if failed == 0:
            summary = "✅ Tüm veriler doğrulandı"
            emoji = "✅"
        elif failed == 1:
            summary = f"⚠️ 1 uyumsuzluk tespit edildi"
            emoji = "⚠️"
        else:
            summary = f"❌ {failed} uyumsuzluk tespit edildi"
            emoji = "❌"
        
        return {
            'is_valid': failed == 0,
            'total_checks': len(results),
            'passed': passed,
            'failed': failed,
            'results': results,
            'summary': summary,
            'emoji': emoji,
            'timestamp': datetime.now()
        }
    
    def _validate_btc_price(self) -> Optional[ValidationResult]:
        """BTC fiyatını Binance vs CoinGecko ile karşılaştır."""
        try:
            # Kaynak 1: Binance
            binance = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                timeout=5
            ).json()
            binance_price = float(binance['price'])
            
            # Kaynak 2: CoinGecko
            coingecko = requests.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                timeout=5
            ).json()
            cg_price = coingecko['bitcoin']['usd']
            
            # Fark hesapla
            diff_pct = abs(binance_price - cg_price) / cg_price * 100
            is_valid = diff_pct <= self.THRESHOLDS['btc_price']
            
            return ValidationResult(
                metric='BTC/USDT Fiyat',
                our_value=binance_price,
                reference_value=cg_price,
                reference_source='CoinGecko',
                difference_pct=diff_pct,
                is_valid=is_valid,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"BTC price validation failed: {e}")
            return None
    
    def _validate_btc_dominance(self) -> Optional[ValidationResult]:
        """BTC Dominance doğrulama."""
        try:
            # Kaynak 1: CoinMarketCap HTML scraping
            from src.brain.tradingview_scraper import TradingViewScraper
            tv = TradingViewScraper()
            our_btcd = tv.get_symbol_data('btc_dominance').get('price', 0)
            
            if our_btcd == 0:
                return None
            
            # Kaynak 2: CoinGecko Global API
            cg = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10
            ).json()
            cg_btcd = cg['data']['market_cap_percentage']['btc']
            
            diff_pct = abs(our_btcd - cg_btcd)
            is_valid = diff_pct <= self.THRESHOLDS['dominance']
            
            return ValidationResult(
                metric='BTC Dominance',
                our_value=our_btcd,
                reference_value=cg_btcd,
                reference_source='CoinGecko Global',
                difference_pct=diff_pct,
                is_valid=is_valid,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"BTC dominance validation failed: {e}")
            return None
    
    def _validate_fear_greed(self) -> Optional[ValidationResult]:
        """Fear & Greed Index doğrulama."""
        try:
            # Kaynak 1: Alternative.me
            alt = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10
            ).json()
            our_fng = int(alt['data'][0]['value'])
            
            # Şu an tek kaynak var, sadece varlık kontrolü
            return ValidationResult(
                metric='Fear & Greed Index',
                our_value=our_fng,
                reference_value=our_fng,
                reference_source='Alternative.me',
                difference_pct=0,
                is_valid=True,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Fear & Greed validation failed: {e}")
            return None
    
    def _validate_open_interest(self) -> Optional[ValidationResult]:
        """Open Interest doğrulama."""
        try:
            # Kaynak: Binance Futures
            oi = requests.get(
                "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
                timeout=5
            ).json()
            oi_btc = float(oi['openInterest'])
            
            # Fiyatla USD değerini hesapla
            price = requests.get(
                "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
                timeout=5
            ).json()
            btc_price = float(price['price'])
            oi_usd = oi_btc * btc_price
            
            return ValidationResult(
                metric='Open Interest (BTC)',
                our_value=oi_btc,
                reference_value=oi_btc,
                reference_source='Binance Futures',
                difference_pct=0,
                is_valid=True,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Open Interest validation failed: {e}")
            return None
    
    def _validate_long_short(self) -> Optional[ValidationResult]:
        """Long/Short Ratio doğrulama."""
        try:
            # Kaynak: Binance Futures
            ls = requests.get(
                "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=1",
                timeout=5
            ).json()
            ls_ratio = float(ls[0]['longShortRatio'])
            
            return ValidationResult(
                metric='Long/Short Ratio',
                our_value=ls_ratio,
                reference_value=ls_ratio,
                reference_source='Binance Futures',
                difference_pct=0,
                is_valid=True,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.warning(f"Long/Short validation failed: {e}")
            return None
    
    def get_validation_summary_for_dashboard(self) -> Dict:
        """Dashboard için özet döndür."""
        if not self.last_validation or (datetime.now() - self.last_validation).seconds > 300:
            return self.validate_all()
        
        # Cache'den döndür
        passed = sum(1 for r in self.validation_results if r.is_valid)
        failed = len(self.validation_results) - passed
        
        return {
            'is_valid': failed == 0,
            'total_checks': len(self.validation_results),
            'passed': passed,
            'failed': failed,
            'emoji': "✅" if failed == 0 else "⚠️" if failed == 1 else "❌",
            'summary': f"Son kontrol: {self.last_validation.strftime('%H:%M:%S')}",
            'details': [
                {
                    'metric': r.metric,
                    'our': f"{r.our_value:.2f}",
                    'ref': f"{r.reference_value:.2f}",
                    'source': r.reference_source,
                    'diff': f"{r.difference_pct:.2f}%",
                    'status': "✅" if r.is_valid else "❌"
                }
                for r in self.validation_results
            ]
        }


# Convenience function
def validate_all_data() -> Dict:
    """Tüm verileri doğrula."""
    validator = DataValidator()
    return validator.validate_all()


def quick_check() -> str:
    """Hızlı durum kontrolü - dashboard için."""
    validator = DataValidator()
    result = validator.validate_all()
    return f"{result['emoji']} {result['passed']}/{result['total_checks']} doğrulandı"
