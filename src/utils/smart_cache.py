# -*- coding: utf-8 -*-
"""
DEMIR AI - Smart Cache System
==============================
File-based persistent cache with TTL support.
Redis-ready architecture for future scaling.

Features:
- File-based persistence (JSON)
- TTL (Time To Live) support
- Memory + disk hybrid caching
- Redis-compatible interface
"""
import logging
import json
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import threading

logger = logging.getLogger("SMART_CACHE")


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'hits': self.hits
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        return cls(
            key=data['key'],
            value=data['value'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            hits=data.get('hits', 0)
        )


class SmartCache:
    """
    Smart Cache System
    
    File-based persistence with in-memory caching.
    Redis-compatible interface for easy migration.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        default_ttl_seconds: int = 300,
        max_memory_items: int = 1000,
        enable_persistence: bool = True
    ):
        """
        Args:
            cache_dir: Directory for persistent cache files
            default_ttl_seconds: Default TTL in seconds (0 = infinite)
            max_memory_items: Max items in memory cache
            enable_persistence: Enable disk persistence
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl_seconds
        self.max_memory_items = max_memory_items
        self.enable_persistence = enable_persistence
        
        # In-memory cache
        self._memory: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Stats
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
        # Create cache directory
        if enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
        
        logger.info(f"💾 SmartCache initialized: Dir={cache_dir}, TTL={default_ttl_seconds}s")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        Redis-compatible: GET key
        """
        with self._lock:
            # Check memory first
            if key in self._memory:
                entry = self._memory[key]
                
                if entry.is_expired:
                    self._delete_entry(key)
                    self._stats['misses'] += 1
                    return default
                
                entry.hits += 1
                self._stats['hits'] += 1
                return entry.value
            
            # Check disk if persistence enabled
            if self.enable_persistence:
                entry = self._load_from_disk(key)
                if entry and not entry.is_expired:
                    self._memory[key] = entry
                    entry.hits += 1
                    self._stats['hits'] += 1
                    return entry.value
            
            self._stats['misses'] += 1
            return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        persist: bool = True
    ) -> bool:
        """
        Set value in cache.
        Redis-compatible: SET key value EX ttl
        """
        with self._lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            
            # Check memory limit
            if len(self._memory) >= self.max_memory_items:
                self._evict_oldest()
            
            self._memory[key] = entry
            self._stats['sets'] += 1
            
            # Persist to disk
            if self.enable_persistence and persist:
                self._save_to_disk(entry)
            
            return True
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        Redis-compatible: DEL key
        """
        with self._lock:
            return self._delete_entry(key)
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists.
        Redis-compatible: EXISTS key
        """
        with self._lock:
            if key in self._memory:
                if not self._memory[key].is_expired:
                    return True
                self._delete_entry(key)
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get remaining TTL in seconds.
        Redis-compatible: TTL key
        Returns: -1 if no TTL, -2 if not exists, else seconds remaining
        """
        with self._lock:
            if key not in self._memory:
                return -2
            
            entry = self._memory[key]
            if entry.is_expired:
                return -2
            
            if entry.expires_at is None:
                return -1
            
            remaining = (entry.expires_at - datetime.now()).total_seconds()
            return max(0, int(remaining))
    
    def keys(self, pattern: str = "*") -> list:
        """
        Get all keys matching pattern.
        Redis-compatible: KEYS pattern
        """
        with self._lock:
            # Simple pattern matching (only * supported)
            if pattern == "*":
                return [k for k, v in self._memory.items() if not v.is_expired]
            
            # Prefix matching
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                return [k for k, v in self._memory.items() 
                       if k.startswith(prefix) and not v.is_expired]
            
            return []
    
    def clear(self) -> int:
        """
        Clear all cache.
        Redis-compatible: FLUSHALL
        """
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
            
            if self.enable_persistence:
                for f in self.cache_dir.glob("*.cache"):
                    f.unlink()
            
            return count
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': round(hit_rate, 2),
                'memory_items': len(self._memory),
                'max_memory': self.max_memory_items
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            expired_keys = [k for k, v in self._memory.items() if v.is_expired]
            for key in expired_keys:
                self._delete_entry(key)
            return len(expired_keys)
    
    # =========================================
    # Private Methods
    # =========================================
    
    def _delete_entry(self, key: str) -> bool:
        """Delete entry from memory and disk"""
        deleted = False
        
        if key in self._memory:
            del self._memory[key]
            deleted = True
        
        if self.enable_persistence:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
        
        if deleted:
            self._stats['deletes'] += 1
        
        return deleted
    
    def _evict_oldest(self):
        """Evict oldest entry when memory is full"""
        if not self._memory:
            return
        
        # Find oldest entry
        oldest_key = min(self._memory.keys(), 
                        key=lambda k: self._memory[k].created_at)
        self._delete_entry(oldest_key)
        logger.debug(f"🗑️ Evicted oldest cache entry: {oldest_key}")
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        # Hash key for safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        safe_key = ''.join(c if c.isalnum() else '_' for c in key[:32])
        return self.cache_dir / f"{safe_key}_{key_hash}.cache"
    
    def _save_to_disk(self, entry: CacheEntry):
        """Save entry to disk"""
        try:
            cache_file = self._get_cache_file(entry.key)
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to persist cache: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk"""
        try:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                return CacheEntry.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _load_persistent_cache(self):
        """Load all cache files from disk"""
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    entry = CacheEntry.from_dict(data)
                    
                    if not entry.is_expired:
                        self._memory[entry.key] = entry
                        count += 1
                    else:
                        cache_file.unlink()  # Remove expired
                except:
                    continue
            
            if count > 0:
                logger.info(f"📂 Loaded {count} cache entries from disk")
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")


# =========================================
# Redis Wrapper (Future Use)
# =========================================

class RedisCache:
    """
    Redis Cache Wrapper
    
    Drop-in replacement for SmartCache when Redis is available.
    Uncomment and configure when scaling.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis connection.
        
        Requires: pip install redis
        """
        try:
            import redis
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            logger.info(f"🔴 Redis connected: {host}:{port}")
        except ImportError:
            logger.error("Redis not installed: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        value = self.client.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except:
            return value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300, **kwargs) -> bool:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        return self.client.setex(key, ttl_seconds, serialized)
    
    def delete(self, key: str) -> bool:
        return self.client.delete(key) > 0
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key) > 0
    
    def ttl(self, key: str) -> int:
        return self.client.ttl(key)
    
    def keys(self, pattern: str = "*") -> list:
        return self.client.keys(pattern)
    
    def clear(self) -> int:
        return self.client.flushdb()


# =========================================
# Factory Function
# =========================================

_cache_instance: Optional[SmartCache] = None


def get_cache(use_redis: bool = False, **kwargs) -> Union[SmartCache, 'RedisCache']:
    """
    Get or create cache instance.
    
    Args:
        use_redis: Use Redis instead of file cache
        **kwargs: Cache configuration
    """
    global _cache_instance
    
    if use_redis:
        return RedisCache(**kwargs)
    
    if _cache_instance is None:
        _cache_instance = SmartCache(**kwargs)
    
    return _cache_instance
