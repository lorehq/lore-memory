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

    # ── Scoped hot memory ──────────────────────────────────────────────────

    def _scoped_key(self, scope: str, project: str, key: str) -> str:
        if scope == "global":
            return f"lore:hot:global:{key}"
        return f"lore:hot:project:{project}:{key}"

    def _index_add(self, scope: str, project: str, full_key: str):
        pipe = self._r.pipeline()
        pipe.sadd("lore:hot:idx:all", full_key)
        if scope == "global":
            pipe.sadd("lore:hot:idx:global", full_key)
        else:
            pipe.sadd(f"lore:hot:idx:project:{project}", full_key)
        pipe.execute()

    def _get_index_keys(self, scope: str, project: str) -> set:
        if scope == "global":
            return self._r.smembers("lore:hot:idx:global")
        if scope == "all":
            return self._r.smembers("lore:hot:idx:all")
        # Default "project" scope: global + this project
        g = self._r.smembers("lore:hot:idx:global") or set()
        p = self._r.smembers(f"lore:hot:idx:project:{project}") or set()
        return g | p

    def hot_write(self, key: str, scope: str, project: str,
                  content: str = "", type_: str = "fact",
                  name: str = "", description: str = "", body: str = ""):
        full_key = self._scoped_key(scope, project, key)
        now = str(int(time.time()))
        fields = {
            "content": content, "type": type_,
            "created_at": now, "last_accessed": now, "hits": "1",
        }
        if name:
            fields["name"] = name
        if description:
            fields["description"] = description
        if body:
            fields["body"] = body
        self._r.hset(full_key, mapping=fields)
        self._index_add(scope, project, full_key)
        return full_key

    def hot_recall(self, limit: int = 10, scope: str = "project",
                   project: str = "", decay_half_life_days: float = 7.0):
        decay_constant = math.log(2) / (decay_half_life_days * 24 * 3600)
        now = int(time.time())
        keys = self._get_index_keys(scope, project)
        if not keys:
            return []

        pipe = self._r.pipeline()
        key_list = list(keys)
        for k in key_list:
            pipe.hgetall(k)
        results = pipe.execute()

        facts = []
        boost_pipe = self._r.pipeline()
        for full_key, data in zip(key_list, results):
            if not data or not data.get("created_at"):
                continue
            hits = int(data.get("hits", 1))
            last_accessed = int(data.get("last_accessed", data.get("created_at", 0)))
            score = hits * math.exp(-decay_constant * max(0, now - last_accessed))

            # Boost on recall
            boost_pipe.hincrby(full_key, "hits", 1)
            boost_pipe.hset(full_key, "last_accessed", str(now))

            # Extract display key (strip prefix)
            display_key = full_key
            for prefix in ("lore:hot:global:", f"lore:hot:project:{project}:"):
                if full_key.startswith(prefix):
                    display_key = full_key[len(prefix):]
                    break
            tier = "global" if full_key.startswith("lore:hot:global:") else project

            facts.append({
                "key": display_key,
                "tier": tier,
                "score": score,
                "type": data.get("type", "fact"),
                "content": data.get("content", ""),
                "description": data.get("description", ""),
                "body": data.get("body", ""),
            })

        boost_pipe.execute()
        facts.sort(key=lambda x: x["score"], reverse=True)
        return facts[:limit]

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
