"""
Circuit Breaker Pattern Implementation.
FIX 2.2: Part of HIGH PRIORITY fixes.

Protects system from cascading failures when external services are down.
"""
import time
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, TypeVar

logger = logging.getLogger("CIRCUIT_BREAKER")

T = TypeVar('T')

class CircuitBreakerOpenException(Exception):
    """Raised when calls are attempted while circuit is open"""
    pass

class CircuitBreaker:
    """
    State machine:
    - CLOSED: Normal operation. Calls go through.
    - OPEN: Service failed. Calls fail fast.
    - HALF-OPEN: Testing if service recovered. One call allowed.
    """
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0
        
    def _set_state(self, new_state: str):
        if self.state != new_state:
            logger.info(f"🔌 Circuit Breaker '{self.name}': {self.state} → {new_state}")
            self.state = new_state
            
    def record_success(self):
        """Call succeeded, reset failures"""
        if self.state == "HALF-OPEN":
            self._set_state("CLOSED")
            self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
            
    def record_failure(self):
        """Call failed, increment counter and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF-OPEN":
            self._set_state("OPEN")
        
        elif self.state == "CLOSED":
            if self.failure_count >= self.failure_threshold:
                self._set_state("OPEN")
                logger.error(f"🚨 Circuit Breaker '{self.name}' OPENED after {self.failure_count} failures")

    def allow_request(self) -> bool:
        """Check if request allowed"""
        if self.state == "CLOSED":
            return True
            
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed > self.recovery_timeout:
                self._set_state("HALF-OPEN")
                return True
            return False
            
        if self.state == "HALF-OPEN":
            # Only allow one request (simplified: strictly sequential logic would need locking)
            return True
        return True

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator usage"""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not self.allow_request():
                logger.warning(f"⛔ Circuit '{self.name}' is OPEN. Blocking request.")
                # Return None or raise exception? For safety, we raise custom exception
                raise CircuitBreakerOpenException(f"Circuit '{self.name}' is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                # Don't count CircuitBreakerOpenException as failure (recursive safety)
                if not isinstance(e, CircuitBreakerOpenException):
                    self.record_failure()
                raise e
                
        return wrapper
