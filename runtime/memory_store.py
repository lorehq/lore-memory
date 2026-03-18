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
            scored.append(
                {
                    "path": p,
                    "hits": hits,
                    "last_accessed": last_accessed,
                    "current_score": current_score,
                }
            )

        scored.sort(key=lambda x: x["current_score"], reverse=True)
        return scored[:limit]

    # ── Scoped hot memory ──────────────────────────────────────────────────

    VALID_TYPES = {"session-memory", "project-memory", "fieldnote"}
    VALID_SCOPES = {"session", "project", "global"}

    def _scoped_key(
        self, scope: str, project: str, topic: str, session_id: str = ""
    ) -> str:
        if scope == "session":
            return f"lore:hot:session:{session_id}:{topic}"
        if scope == "global":
            return f"lore:hot:global:{topic}"
        return f"lore:hot:project:{project}:{topic}"

    def _index_add(self, scope: str, project: str, session_id: str, full_key: str):
        pipe = self._r.pipeline()
        pipe.sadd("lore:hot:idx:all", full_key)
        if scope == "session":
            pipe.sadd(f"lore:hot:idx:session:{session_id}", full_key)
            pipe.sadd(f"lore:hot:idx:project-sessions:{project}", full_key)
        elif scope == "global":
            pipe.sadd("lore:hot:idx:global", full_key)
        else:
            pipe.sadd(f"lore:hot:idx:project:{project}", full_key)
        pipe.execute()

    def _get_working_keys(
        self, project: str, session_id: str = ""
    ) -> list[tuple[str, str]]:
        tiers = []

        if session_id:
            session_keys = (
                self._r.smembers(f"lore:hot:idx:session:{session_id}") or set()
            )
            tiers.append(("session", session_keys))

        project_keys = self._r.smembers(f"lore:hot:idx:project:{project}") or set()
        tiers.append(("project", project_keys))

        same_project_session_keys = self._get_project_session_keys(project)
        if session_id:
            current_session_keys = (
                self._r.smembers(f"lore:hot:idx:session:{session_id}") or set()
            )
            same_project_session_keys -= current_session_keys
        tiers.append(("project-session", same_project_session_keys))

        global_keys = self._r.smembers("lore:hot:idx:global") or set()
        tiers.append(("global", global_keys))

        seen = set()
        ordered = []
        for tier, keys in tiers:
            for key in keys:
                if key in seen:
                    continue
                seen.add(key)
                ordered.append((key, tier))
        return ordered

    def _get_project_session_keys(self, project: str) -> set:
        keys = self._r.smembers(f"lore:hot:idx:project-sessions:{project}") or set()
        recovered = set()
        for key in self._r.scan_iter(match="lore:hot:session:*"):
            data = self._r.hgetall(key)
            if data.get("project") == project:
                recovered.add(key)
        return keys | recovered

    def _get_index_keys(self, scope: str, project: str, session_id: str = ""):
        if scope == "working":
            return self._get_working_keys(project, session_id=session_id)
        if scope == "session":
            return [
                (k, "session")
                for k in (
                    self._r.smembers(f"lore:hot:idx:session:{session_id}") or set()
                )
            ]
        if scope == "global":
            return [
                (k, "global")
                for k in (self._r.smembers("lore:hot:idx:global") or set())
            ]
        if scope == "all":
            return [(k, "all") for k in (self._r.smembers("lore:hot:idx:all") or set())]
        if scope == "project":
            return [
                (k, "project")
                for k in (self._r.smembers(f"lore:hot:idx:project:{project}") or set())
            ]
        return self._get_working_keys(project, session_id=session_id)

    def _display_topic(self, full_key: str) -> str:
        parts = full_key.split(":")
        if len(parts) >= 5 and parts[:3] == ["lore", "hot", "session"]:
            return ":".join(parts[4:])
        if len(parts) >= 4 and parts[:3] == ["lore", "hot", "project"]:
            return ":".join(parts[4:])
        if len(parts) >= 3 and parts[:3] == ["lore", "hot", "global"]:
            return ":".join(parts[3:])
        return full_key

    def _tier_bonus(self, tier: str) -> float:
        return {
            "session": 40.0,
            "project": 30.0,
            "project-session": 20.0,
            "global": 10.0,
            "all": 0.0,
        }.get(tier, 0.0)

    def hot_write(
        self,
        topic: str,
        scope: str,
        project: str,
        type_: str,
        content: str = "",
        session_ref: str = "",
        name: str = "",
        description: str = "",
        body: str = "",
    ):
        if type_ not in self.VALID_TYPES:
            raise ValueError(
                f"invalid type: {type_!r}, must be one of {self.VALID_TYPES}"
            )
        if scope not in self.VALID_SCOPES:
            raise ValueError(
                f"invalid scope: {scope!r}, must be one of {self.VALID_SCOPES}"
            )
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

    def hot_recall(
        self,
        limit: int = 10,
        scope: str = "working",
        project: str = "",
        session_id: str = "",
        decay_half_life_days: float = 7.0,
    ):
        decay_constant = math.log(2) / (decay_half_life_days * 24 * 3600)
        now = int(time.time())
        keyed_entries = self._get_index_keys(scope, project, session_id=session_id)
        if not keyed_entries:
            return []

        pipe = self._r.pipeline()
        key_list = [full_key for full_key, _ in keyed_entries]
        for k in key_list:
            pipe.hgetall(k)
        results = pipe.execute()

        facts = []
        boost_pipe = self._r.pipeline()
        for (full_key, tier), data in zip(keyed_entries, results):
            if not data or "created_at" not in data:
                continue
            hits = int(data["hits"])
            last_accessed = int(data["last_accessed"])
            score = hits * math.exp(-decay_constant * max(0, now - last_accessed))
            rank_score = self._tier_bonus(tier) + score

            # Boost on recall
            boost_pipe.hincrby(full_key, "hits", 1)
            boost_pipe.hset(full_key, "last_accessed", str(now))

            entry = {
                "topic": self._display_topic(full_key),
                "scope": data.get("scope", "global"),
                "score": rank_score,
                "type": data.get("type", "memory"),
                "content": data.get("content", ""),
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
        project_count = (
            self._r.scard(f"lore:hot:idx:project:{project}") if project else 0
        )
        session_count = (
            self._r.scard(f"lore:hot:idx:session:{session_id}") if session_id else 0
        )
        return {
            "total": total,
            "global": global_count,
            "project": project_count,
            "session": session_count,
        }

    def hot_delete(
        self, scope: str, project: str, topic: str, session_id: str = ""
    ) -> bool:
        """Delete a hot memory entry by scope + topic. Returns True if found."""
        full_key = self._scoped_key(scope, project, topic, session_id=session_id)
        existed = self._r.delete(full_key) > 0
        if existed:
            pipe = self._r.pipeline()
            pipe.srem("lore:hot:idx:all", full_key)
            if scope == "session":
                pipe.srem(f"lore:hot:idx:session:{session_id}", full_key)
                pipe.srem(f"lore:hot:idx:project-sessions:{project}", full_key)
            elif scope == "global":
                pipe.srem("lore:hot:idx:global", full_key)
            else:
                pipe.srem(f"lore:hot:idx:project:{project}", full_key)
            pipe.execute()
        return existed

    def get_faded_primitives(
        self, threshold: float = 0.1, decay_half_life_days: float = 7.0
    ):
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
                faded.append(
                    {
                        "path": p,
                        "hits": hits,
                        "last_accessed": last_accessed,
                        "current_score": current_score,
                    }
                )

        return faded
