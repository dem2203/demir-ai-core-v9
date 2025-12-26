# -*- coding: utf-8 -*-
"""
DEMIR AI - Structured Logger
=============================
JSON formatında yapılandırılmış loglama sistemi.
ELK Stack (Elasticsearch, Logstash, Kibana) hazır.

Features:
- JSON formatında loglar
- Log rotation (dosya boyutu/sayı limiti)
- Context injection (request_id, symbol, etc.)
- Performance metrics loglama
- Error tracking with stack traces
"""
import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
import traceback
import threading


class JSONFormatter(logging.Formatter):
    """
    JSON formatında log formatter.
    ELK Stack uyumlu.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self.hostname = os.environ.get('HOSTNAME', 'unknown')
        self.service = 'demir-ai-v10'
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            '@timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service,
            'host': self.hostname,
            'thread': record.threadName,
            'file': f"{record.filename}:{record.lineno}"
        }
        
        # Exception bilgisi
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'stacktrace': traceback.format_exception(*record.exc_info)
            }
        
        # Extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename', 
                              'funcName', 'levelname', 'levelno', 'lineno',
                              'module', 'msecs', 'pathname', 'process',
                              'processName', 'relativeCreated', 'stack_info',
                              'exc_info', 'exc_text', 'thread', 'threadName', 'message']:
                    if not key.startswith('_'):
                        log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class ContextLogger(logging.Logger):
    """
    Context injection destekli logger.
    """
    
    def __init__(self, name: str, level: int = logging.NOTSET):
        super().__init__(name, level)
        self._context: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def set_context(self, **kwargs):
        """Set context fields to include in all logs"""
        with self._lock:
            self._context.update(kwargs)
    
    def clear_context(self):
        """Clear all context fields"""
        with self._lock:
            self._context.clear()
    
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, **kwargs):
        """Override _log to inject context"""
        if extra is None:
            extra = {}
        
        with self._lock:
            extra.update(self._context)
        
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, 
                    stack_info=stack_info, **kwargs)


class StructuredLogger:
    """
    Structured Logging System
    
    Merkezi loglama yönetimi.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "demir_ai.log",
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        log_level: str = "INFO",
        enable_json: bool = True,
        enable_console: bool = True
    ):
        """
        Args:
            log_dir: Log dosyaları dizini
            log_file: Ana log dosyası adı
            max_file_size_mb: Max dosya boyutu (MB)
            backup_count: Backup dosya sayısı
            log_level: Log seviyesi
            enable_json: JSON formatı aktif
            enable_console: Console output aktif
        """
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.log_level = getattr(logging, log_level.upper())
        self.enable_json = enable_json
        self.enable_console = enable_console
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Register custom logger class
        logging.setLoggerClass(ContextLogger)
        
        # Setup handlers
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # File handler with rotation
        file_path = self.log_dir / self.log_file
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        
        if self.enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
            ))
        
        root_logger.addHandler(file_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            # Human readable for console
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
                datefmt='%H:%M:%S'
            ))
            root_logger.addHandler(console_handler)
        
        # Error log (separate file for errors only)
        error_file = self.log_dir / "errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> ContextLogger:
        """Get logger with context support"""
        return logging.getLogger(name)


# =========================================
# Performance Logging Helpers
# =========================================

class PerformanceLogger:
    """Helper for logging performance metrics"""
    
    def __init__(self, logger_name: str = "PERFORMANCE"):
        self.logger = logging.getLogger(logger_name)
        self._timers: Dict[str, datetime] = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self._timers[operation] = datetime.now()
    
    def end_timer(self, operation: str, extra: Optional[Dict] = None):
        """End timing and log duration"""
        if operation not in self._timers:
            return
        
        duration = (datetime.now() - self._timers.pop(operation)).total_seconds() * 1000
        
        log_extra = {
            'operation': operation,
            'duration_ms': round(duration, 2),
            'metric_type': 'timing'
        }
        if extra:
            log_extra.update(extra)
        
        self.logger.info(f"⏱️ {operation}: {duration:.2f}ms", extra=log_extra)
    
    def log_metric(self, name: str, value: float, unit: str = "", extra: Optional[Dict] = None):
        """Log a numeric metric"""
        log_extra = {
            'metric_name': name,
            'metric_value': value,
            'metric_unit': unit,
            'metric_type': 'gauge'
        }
        if extra:
            log_extra.update(extra)
        
        self.logger.info(f"📊 {name}: {value}{unit}", extra=log_extra)


class TradingLogger:
    """Helper for logging trading events"""
    
    def __init__(self, logger_name: str = "TRADING"):
        self.logger = logging.getLogger(logger_name)
    
    def log_signal(self, symbol: str, direction: str, confidence: float, 
                   reason: str, extra: Optional[Dict] = None):
        """Log trading signal"""
        log_extra = {
            'event_type': 'signal',
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'reason': reason
        }
        if extra:
            log_extra.update(extra)
        
        emoji = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"
        self.logger.info(
            f"{emoji} SIGNAL: {symbol} {direction} ({confidence:.1f}%) - {reason}",
            extra=log_extra
        )
    
    def log_trade(self, symbol: str, action: str, price: float, 
                  quantity: float, extra: Optional[Dict] = None):
        """Log executed trade"""
        log_extra = {
            'event_type': 'trade',
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'value_usd': price * quantity
        }
        if extra:
            log_extra.update(extra)
        
        emoji = "🛒" if action == "BUY" else "💰"
        self.logger.info(
            f"{emoji} TRADE: {action} {quantity} {symbol} @ ${price:.2f}",
            extra=log_extra
        )
    
    def log_pnl(self, symbol: str, realized_pnl: float, unrealized_pnl: float,
                extra: Optional[Dict] = None):
        """Log P&L update"""
        log_extra = {
            'event_type': 'pnl',
            'symbol': symbol,
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': realized_pnl + unrealized_pnl
        }
        if extra:
            log_extra.update(extra)
        
        emoji = "📈" if realized_pnl >= 0 else "📉"
        self.logger.info(
            f"{emoji} PNL: {symbol} Realized=${realized_pnl:.2f}, Unrealized=${unrealized_pnl:.2f}",
            extra=log_extra
        )


# =========================================
# Global Setup
# =========================================

_structured_logger: Optional[StructuredLogger] = None
_perf_logger: Optional[PerformanceLogger] = None
_trading_logger: Optional[TradingLogger] = None


def setup_logging(**kwargs) -> StructuredLogger:
    """Initialize structured logging system"""
    global _structured_logger, _perf_logger, _trading_logger
    
    _structured_logger = StructuredLogger(**kwargs)
    _perf_logger = PerformanceLogger()
    _trading_logger = TradingLogger()
    
    logging.getLogger("STRUCTURED_LOGGER").info(
        "📋 Structured logging initialized",
        extra={'config': kwargs}
    )
    
    return _structured_logger


def get_perf_logger() -> PerformanceLogger:
    """Get performance logger"""
    global _perf_logger
    if _perf_logger is None:
        _perf_logger = PerformanceLogger()
    return _perf_logger


def get_trading_logger() -> TradingLogger:
    """Get trading logger"""
    global _trading_logger
    if _trading_logger is None:
        _trading_logger = TradingLogger()
    return _trading_logger
