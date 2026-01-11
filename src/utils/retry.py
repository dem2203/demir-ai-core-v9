"""
Retry utility for external API calls with exponential backoff.
FIX 2.1: Part of HIGH PRIORITY fixes
"""
import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Any

logger = logging.getLogger("RETRY_UTIL")

T = TypeVar('T')

def async_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
    backoff_multiplier: float = 2.0
):
    """
    Async retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        backoff_multiplier: Multiplier for exponential backoff
    
    Usage:
        @async_retry(max_attempts=3, base_delay=2)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(


*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    wait_time = base_delay * (backoff_multiplier ** (attempt - 1))
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def sync_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple = (Exception,),
    backoff_multiplier: float = 2.0
):
    """
    Synchronous retry decorator with exponential backoff.
    
    Same as async_retry but for synchronous functions.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import time
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"❌ {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise
                    
                    wait_time = base_delay * (backoff_multiplier ** (attempt - 1))
                    logger.warning(
                        f"⚠️ {func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator
