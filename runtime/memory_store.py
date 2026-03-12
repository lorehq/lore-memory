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

    VALID_TYPES = {"session-memory", "project-memory", "fieldnote"}
    VALID_SCOPES = {"session", "project", "global"}

    def _scoped_key(self, scope: str, project: str, topic: str,
                    session_id: str = "") -> str:
        if scope == "session":
            return f"lore:hot:session:{session_id}:{topic}"
        if scope == "global":
            return f"lore:hot:global:{topic}"
        return f"lore:hot:project:{project}:{topic}"

    def _index_add(self, scope: str, project: str, session_id: str,
                   full_key: str):
        pipe = self._r.pipeline()
        pipe.sadd("lore:hot:idx:all", full_key)
        if scope == "session":
            pipe.sadd(f"lore:hot:idx:session:{session_id}", full_key)
        elif scope == "global":
            pipe.sadd("lore:hot:idx:global", full_key)
        else:
            pipe.sadd(f"lore:hot:idx:project:{project}", full_key)
        pipe.execute()

    def _get_index_keys(self, scope: str, project: str,
                        session_id: str = "") -> set:
        if scope == "session":
            return self._r.smembers(f"lore:hot:idx:session:{session_id}")
        if scope == "global":
            return self._r.smembers("lore:hot:idx:global")
        if scope == "all":
            return self._r.smembers("lore:hot:idx:all")
        # Default scope: session + project + global
        s = self._r.smembers(f"lore:hot:idx:session:{session_id}") or set() if session_id else set()
        p = self._r.smembers(f"lore:hot:idx:project:{project}") or set()
        g = self._r.smembers("lore:hot:idx:global") or set()
        return s | p | g

    def hot_write(self, topic: str, scope: str, project: str,
                  type_: str, content: str = "",
                  session_ref: str = "",
                  name: str = "", description: str = "", body: str = ""):
        if type_ not in self.VALID_TYPES:
            raise ValueError(f"invalid type: {type_!r}, must be one of {self.VALID_TYPES}")
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"invalid scope: {scope!r}, must be one of {self.VALID_SCOPES}")
        full_key = self._scoped_key(scope, project, topic, session_id=session_ref)
        now = str(int(time.time()))
        fields = {
            "content": content,
            "type": type_,
            "scope": scope,
            "project": project,
            "created_at": now,
            "updated_at": now,
            "last_accessed": now,
            "hits": "1",
        }
        if session_ref:
            fields["session_ref"] = session_ref
        if name:
            fields["name"] = name
        if description:
            fields["description"] = description
        if body:
            fields["body"] = body
        self._r.hset(full_key, mapping=fields)
        self._index_add(scope, project, session_ref, full_key)
        return full_key

    def hot_recall(self, limit: int = 10, scope: str = "project",
                   project: str = "", session_id: str = "",
                   decay_half_life_days: float = 7.0):
        decay_constant = math.log(2) / (decay_half_life_days * 24 * 3600)
        now = int(time.time())
        keys = self._get_index_keys(scope, project, session_id=session_id)
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
            if not data or "created_at" not in data:
                continue
            hits = int(data["hits"])
            last_accessed = int(data["last_accessed"])
            score = hits * math.exp(-decay_constant * max(0, now - last_accessed))

            # Boost on recall
            boost_pipe.hincrby(full_key, "hits", 1)
            boost_pipe.hset(full_key, "last_accessed", str(now))

            # Extract display topic (strip prefix)
            display_topic = full_key
            for prefix in (
                f"lore:hot:session:{session_id}:",
                f"lore:hot:project:{project}:",
                "lore:hot:global:",
            ):
                if full_key.startswith(prefix):
                    display_topic = full_key[len(prefix):]
                    break

            entry = {
                "topic": display_topic,
                "scope": data["scope"],
                "score": score,
                "type": data["type"],
                "content": data["content"],
            }
            if "description" in data:
                entry["description"] = data["description"]
            if "body" in data:
                entry["body"] = data["body"]
            if "session_ref" in data:
                entry["session_ref"] = data["session_ref"]

            facts.append(entry)

        boost_pipe.execute()
        facts.sort(key=lambda x: x["score"], reverse=True)
        return facts[:limit]

    def hot_stats(self, project: str = "", session_id: str = ""):
        total = self._r.scard("lore:hot:idx:all")
        global_count = self._r.scard("lore:hot:idx:global")
        project_count = self._r.scard(f"lore:hot:idx:project:{project}") if project else 0
        session_count = self._r.scard(f"lore:hot:idx:session:{session_id}") if session_id else 0
        return {
            "total": total,
            "global": global_count,
            "project": project_count,
            "session": session_count,
        }

    def hot_delete(self, scope: str, project: str, topic: str,
                   session_id: str = "") -> bool:
        """Delete a hot memory entry by scope + topic. Returns True if found."""
        full_key = self._scoped_key(scope, project, topic, session_id=session_id)
        existed = self._r.delete(full_key) > 0
        if existed:
            pipe = self._r.pipeline()
            pipe.srem("lore:hot:idx:all", full_key)
            if scope == "session":
                pipe.srem(f"lore:hot:idx:session:{session_id}", full_key)
            elif scope == "global":
                pipe.srem("lore:hot:idx:global", full_key)
            else:
                pipe.srem(f"lore:hot:idx:project:{project}", full_key)
            pipe.execute()
        return existed

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
