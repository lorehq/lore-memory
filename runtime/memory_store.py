import time
import math

import redis


class MemoryStore:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._r = redis.Redis.from_url(redis_url, decode_responses=True)
        self._index_key = "lore:activity:index"

    def _activity_key(self, path: str) -> str:
        return f"lore:activity:{path}"

    def record_access(self, path: str):
        now = int(time.time())
        key = self._activity_key(path)
        pipe = self._r.pipeline()
        pipe.hincrby(key, "hits", 1)
        pipe.hset(key, "last_accessed", now)
        pipe.sadd(self._index_key, path)
        pipe.execute()

    def get_hot_primitives(self, limit: int = 10, decay_half_life_days: float = 7.0):
        decay_constant = math.log(2) / (decay_half_life_days * 24 * 3600)
        now = int(time.time())
        paths = self._r.smembers(self._index_key)
        if not paths:
            return []

        pipe = self._r.pipeline()
        for p in paths:
            pipe.hgetall(self._activity_key(p))
        results = pipe.execute()

        scored = []
        for p, data in zip(paths, results):
            if not data:
                continue
            hits = int(data.get("hits", 0))
            last_accessed = int(data.get("last_accessed", 0))
            current_score = hits * math.exp(-decay_constant * (now - last_accessed))
            scored.append({
                "path": p,
                "hits": hits,
                "last_accessed": last_accessed,
                "current_score": current_score,
            })

        scored.sort(key=lambda x: x["current_score"], reverse=True)
        return scored[:limit]

    def get_faded_primitives(self, threshold: float = 0.1, decay_half_life_days: float = 7.0):
        decay_constant = math.log(2) / (decay_half_life_days * 24 * 3600)
        now = int(time.time())
        paths = self._r.smembers(self._index_key)
        if not paths:
            return []

        pipe = self._r.pipeline()
        for p in paths:
            pipe.hgetall(self._activity_key(p))
        results = pipe.execute()

        faded = []
        for p, data in zip(paths, results):
            if not data:
                continue
            hits = int(data.get("hits", 0))
            last_accessed = int(data.get("last_accessed", 0))
            current_score = hits * math.exp(-decay_constant * (now - last_accessed))
            if current_score < threshold:
                faded.append({
                    "path": p,
                    "hits": hits,
                    "last_accessed": last_accessed,
                    "current_score": current_score,
                })

        return faded
