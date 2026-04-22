"""
EvoDoc Clinical Drug Safety Engine — Cache Layer
Deterministic hash-based caching with TTL support.
Same drugs in any order = same cache key.
"""

import hashlib
import json
import time
import logging
from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.time() - self.created_at) > ttl_seconds

    def record_hit(self):
        self.hits += 1


class DrugSafetyCache:
    """
    In-memory LRU-like cache with TTL.
    Key: SHA-256 hash of sorted(proposed_medicines) + sorted(current_medications)
    TTL: configurable, default 3600s (1 hour)
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._store: Dict[str, CacheEntry] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._total_requests = 0
        self._total_hits = 0

    @staticmethod
    def build_cache_key(proposed_medicines: list[str], current_medications: list[str]) -> str:
        """
        Deterministic hash regardless of input order.
        sorted() ensures 'Aspirin + Warfarin' == 'Warfarin + Aspirin'
        Lowercased to prevent case-sensitive duplicates.
        """
        sorted_proposed = sorted(m.strip().lower() for m in proposed_medicines)
        sorted_current = sorted(m.strip().lower() for m in current_medications)

        key_payload = {
            "proposed": sorted_proposed,
            "current": sorted_current,
        }

        key_string = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(key_string.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """Returns (cache_hit, value). Handles expiry transparently."""
        self._total_requests += 1

        if key not in self._store:
            logger.debug(f"Cache MISS for key {key[:16]}...")
            return False, None

        entry = self._store[key]

        if entry.is_expired(self.ttl_seconds):
            logger.debug(f"Cache EXPIRED for key {key[:16]}...")
            del self._store[key]
            return False, None

        entry.record_hit()
        self._total_hits += 1
        logger.debug(f"Cache HIT for key {key[:16]}... (hit #{entry.hits})")
        return True, entry.value

    def set(self, key: str, value: Any) -> None:
        """Store value with current timestamp. Evict oldest if over capacity."""
        if len(self._store) >= self.max_size:
            self._evict_oldest()

        self._store[key] = CacheEntry(value=value)
        logger.debug(f"Cache SET for key {key[:16]}...")

    def _evict_oldest(self) -> None:
        """Evict the entry with the oldest creation time."""
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].created_at)
        del self._store[oldest_key]
        logger.debug(f"Cache evicted oldest entry: {oldest_key[:16]}...")

    def invalidate(self, key: str) -> bool:
        """Manually invalidate a cache entry."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns count cleared."""
        count = len(self._store)
        self._store.clear()
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        expired_keys = [
            k for k, v in self._store.items()
            if (now - v.created_at) > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._store[k]

        if expired_keys:
            logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
        return len(expired_keys)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return round(self._total_hits / self._total_requests * 100, 2)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "total_requests": self._total_requests,
            "total_hits": self._total_hits,
            "hit_rate_percent": self.hit_rate,
        }


# Singleton cache instance
_cache_instance: Optional[DrugSafetyCache] = None


def get_cache(ttl_seconds: int = 3600) -> DrugSafetyCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DrugSafetyCache(ttl_seconds=ttl_seconds)
        logger.info(f"Cache initialized: TTL={ttl_seconds}s, max_size=1000")
    return _cache_instance
