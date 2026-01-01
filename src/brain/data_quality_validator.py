# -*- coding: utf-8 -*-
"""
DEMIR AI v10 - DATA VALIDATOR
=============================
AI'lara gönderilecek verilerin doğruluğunu kontrol eder.

Kontroller:
1. Range check (değer aralığı)
2. Staleness check (veri yaşı)
3. Consistency check (çelişki tespiti)
4. Anomaly detection (ani değişim tespiti)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("DATA_VALIDATOR")


class ValidationSeverity(Enum):
    INFO = "INFO"       # Bilgilendirme
    WARNING = "WARNING" # Dikkat
    ERROR = "ERROR"     # Hata - veri kullanılmamalı
    CRITICAL = "CRITICAL"  # Kritik - sinyal üretilmemeli


@dataclass
class ValidationIssue:
    """Tek bir validasyon sorunu"""
    field: str
    severity: ValidationSeverity
    message: str
    value: Optional[float] = None
    expected_range: Optional[Tuple[float, float]] = None


@dataclass
class ValidationResult:
    """Validasyon sonucu"""
    is_valid: bool
    quality_score: float  # 0-100
    issues: List[ValidationIssue] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def has_critical_issues(self) -> bool:
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)
    
    def has_errors(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)
    
    def get_issues_summary(self) -> str:
        """Sorunların özeti"""
        if not self.issues:
            return "✅ No issues"
        
        lines = []
        for issue in self.issues:
            emoji = {
                ValidationSeverity.INFO: "ℹ️",
                ValidationSeverity.WARNING: "⚠️",
                ValidationSeverity.ERROR: "❌",
                ValidationSeverity.CRITICAL: "🚨"
            }.get(issue.severity, "❓")
            lines.append(f"{emoji} [{issue.field}] {issue.message}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Veri Validatörü
    
    AI'lara gönderilen verilerin doğruluğunu kontrol eder.
    """
    
    # Değer aralıkları
    VALUE_RANGES = {
        'rsi': (0, 100),
        'fear_greed': (0, 100),
        'btc_dominance': (20, 80),
        'funding_rate': (-0.5, 0.5),
        'ls_ratio': (0.1, 10),
        'orderbook_score': (-100, 100),
        'whale_score': (-100, 100),
        'oi_change': (-50, 50),
        'confidence': (0, 100),
    }
    
    # Maksimum veri yaşı (dakika)
    MAX_DATA_AGE = {
        'price': 1,           # 1 dakika
        'orderbook': 1,       # 1 dakika
        'rsi': 5,             # 5 dakika
        'funding_rate': 60,   # 1 saat
        'fear_greed': 1440,   # 24 saat (günlük güncellenir)
        'btc_dominance': 60,  # 1 saat
        'news': 60,           # 1 saat
        'whale': 30,          # 30 dakika
    }
    
    # Maksimum değişim oranları (%/dakika)
    MAX_CHANGE_RATES = {
        'price': 5.0,         # %5/dakika (çok hızlı hareket)
        'rsi': 20.0,          # RSI 20 puan/dakika
        'orderbook_score': 50.0,  # Orderbook hızlı değişebilir
        'fear_greed': 5.0,    # Fear & Greed yavaş değişir
    }
    
    def __init__(self):
        self._previous_values: Dict[str, Dict] = {}
        self._issue_history: List[ValidationIssue] = []
        logger.info("🔍 Data Validator initialized")
    
    def validate_all(self, data: Dict, symbol: str = "UNKNOWN") -> ValidationResult:
        """Tüm verileri validasyon et"""
        issues = []
        
        # 1. Range checks
        range_issues = self._check_ranges(data)
        issues.extend(range_issues)
        
        # 2. Staleness checks
        staleness_issues = self._check_staleness(data)
        issues.extend(staleness_issues)
        
        # 3. Consistency checks
        consistency_issues = self._check_consistency(data)
        issues.extend(consistency_issues)
        
        # 4. Anomaly detection
        anomaly_issues = self._check_anomalies(data, symbol)
        issues.extend(anomaly_issues)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(issues)
        
        # Determine validity
        is_valid = not any(
            i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for i in issues
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues
        )
        
        # Log results
        if issues:
            logger.warning(f"🔍 Validation issues for {symbol}: {len(issues)} issues, quality={quality_score:.0f}%")
            for issue in issues:
                if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                    logger.error(f"  {issue.severity.value}: {issue.field} - {issue.message}")
        else:
            logger.debug(f"✅ Data validation passed for {symbol}")
        
        return result
    
    def _check_ranges(self, data: Dict) -> List[ValidationIssue]:
        """Değer aralıklarını kontrol et"""
        issues = []
        
        for field, (min_val, max_val) in self.VALUE_RANGES.items():
            if field in data and data[field] is not None:
                value = data[field]
                
                if not isinstance(value, (int, float)):
                    issues.append(ValidationIssue(
                        field=field,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid type: expected number, got {type(value).__name__}",
                        value=None
                    ))
                    continue
                
                if value < min_val or value > max_val:
                    issues.append(ValidationIssue(
                        field=field,
                        severity=ValidationSeverity.ERROR if abs(value - min_val) > abs(max_val - min_val) * 0.5 else ValidationSeverity.WARNING,
                        message=f"Out of range: {value:.2f} not in [{min_val}, {max_val}]",
                        value=value,
                        expected_range=(min_val, max_val)
                    ))
        
        return issues
    
    def _check_staleness(self, data: Dict) -> List[ValidationIssue]:
        """Veri yaşını kontrol et"""
        issues = []
        
        now = datetime.now()
        
        for field, max_age_minutes in self.MAX_DATA_AGE.items():
            timestamp_field = f"{field}_timestamp"
            
            if timestamp_field in data:
                ts = data[timestamp_field]
                
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except:
                        continue
                
                if isinstance(ts, datetime):
                    age_minutes = (now - ts).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        severity = ValidationSeverity.WARNING if age_minutes < max_age_minutes * 2 else ValidationSeverity.ERROR
                        issues.append(ValidationIssue(
                            field=field,
                            severity=severity,
                            message=f"Stale data: {age_minutes:.0f} minutes old (max: {max_age_minutes})",
                            value=age_minutes
                        ))
        
        return issues
    
    def _check_consistency(self, data: Dict) -> List[ValidationIssue]:
        """Çelişkileri kontrol et"""
        issues = []
        
        # RSI vs Orderbook consistency
        rsi = data.get('rsi')
        orderbook = data.get('orderbook_score')
        
        if rsi is not None and orderbook is not None:
            # RSI oversold but orderbook heavy selling = contradiction
            if rsi < 30 and orderbook < -60:
                # This is actually consistent - oversold AND selling pressure
                pass
            elif rsi > 70 and orderbook > 60:
                # Overbought with heavy buying - might be a blow-off top
                issues.append(ValidationIssue(
                    field="rsi+orderbook",
                    severity=ValidationSeverity.INFO,
                    message="RSI overbought with heavy buying - possible blow-off top",
                    value=rsi
                ))
        
        # Fear & Greed vs Price trend consistency
        fear_greed = data.get('fear_greed')
        price_change = data.get('price_change_24h', 0)
        
        if fear_greed is not None and price_change is not None:
            # Extreme fear during price rise - unusual
            if fear_greed < 20 and price_change > 5:
                issues.append(ValidationIssue(
                    field="fear_greed+price",
                    severity=ValidationSeverity.WARNING,
                    message=f"Extreme fear ({fear_greed}) despite {price_change:.1f}% price rise",
                    value=fear_greed
                ))
            # Extreme greed during price drop - unusual
            elif fear_greed > 80 and price_change < -5:
                issues.append(ValidationIssue(
                    field="fear_greed+price",
                    severity=ValidationSeverity.WARNING,
                    message=f"Extreme greed ({fear_greed}) despite {price_change:.1f}% price drop",
                    value=fear_greed
                ))
        
        # Funding rate vs L/S ratio consistency
        funding = data.get('funding_rate', 0)
        ls_ratio = data.get('ls_ratio', 1)
        
        if funding is not None and ls_ratio is not None:
            # High funding should correlate with long-heavy L/S ratio
            if funding > 0.05 and ls_ratio < 1:
                issues.append(ValidationIssue(
                    field="funding+ls_ratio",
                    severity=ValidationSeverity.WARNING,
                    message=f"High funding ({funding:.4f}) but short-heavy L/S ratio ({ls_ratio:.2f})",
                    value=funding
                ))
        
        return issues
    
    def _check_anomalies(self, data: Dict, symbol: str) -> List[ValidationIssue]:
        """Ani değişimleri tespit et"""
        issues = []
        
        # Get previous values
        prev = self._previous_values.get(symbol, {})
        
        for field, max_change in self.MAX_CHANGE_RATES.items():
            if field in data and field in prev:
                current = data[field]
                previous = prev[field]
                
                if current is not None and previous is not None and previous != 0:
                    change_pct = abs((current - previous) / previous) * 100
                    
                    if change_pct > max_change:
                        issues.append(ValidationIssue(
                            field=field,
                            severity=ValidationSeverity.WARNING if change_pct < max_change * 2 else ValidationSeverity.ERROR,
                            message=f"Rapid change: {change_pct:.1f}% (max allowed: {max_change}%)",
                            value=change_pct
                        ))
        
        # Store current values for next check
        self._previous_values[symbol] = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        
        return issues
    
    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Veri kalitesi skoru hesapla (0-100)"""
        if not issues:
            return 100.0
        
        # Penalty per issue type
        penalties = {
            ValidationSeverity.INFO: 2,
            ValidationSeverity.WARNING: 10,
            ValidationSeverity.ERROR: 25,
            ValidationSeverity.CRITICAL: 50
        }
        
        total_penalty = sum(penalties.get(issue.severity, 0) for issue in issues)
        
        return max(0, 100 - total_penalty)
    
    def should_proceed(self, result: ValidationResult, min_quality: float = 50.0) -> Tuple[bool, str]:
        """Sinyal üretimine devam edilmeli mi?"""
        if result.has_critical_issues():
            return False, "Critical data issues detected"
        
        if result.quality_score < min_quality:
            return False, f"Data quality too low: {result.quality_score:.0f}% < {min_quality:.0f}%"
        
        return True, "OK"


# Singleton
_validator: Optional[DataValidator] = None

def get_data_validator() -> DataValidator:
    global _validator
    if _validator is None:
        _validator = DataValidator()
    return _validator


def validate_market_data(data: Dict, symbol: str = "UNKNOWN") -> ValidationResult:
    """Quick access function"""
    validator = get_data_validator()
    return validator.validate_all(data, symbol)
