import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastembed import TextEmbedding
from pydantic import BaseModel
from memory_store import MemoryStore


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DOCS_ROOT = Path(os.getenv("DOCS_SOURCE", "/data/DATABANK"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_K = int(os.getenv("DEFAULT_K", "10"))
MAX_K = int(os.getenv("MAX_K", "10"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "700"))
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "80"))
RESULT_MODE = os.getenv("RESULT_MODE", "paths_min")


@dataclass
class Chunk:
    path: str
    chunk_id: str
    text: str


class ActivityRequest(BaseModel):
    path: str

class ReindexPathRequest(BaseModel):
    path: str

class HotDeleteRequest(BaseModel):
    topic: str
    scope: str
    project: str
    session_ref: str = ""

class HotWriteRequest(BaseModel):
    topic: str
    content: str = ""
    scope: str
    project: str
    type: str
    session_ref: str = ""
    name: str = ""
    description: str = ""
    body: str = ""


HOT_PATH_PREFIX = "hot:"


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = TextEmbedding(model_name=EMBED_MODEL, local_files_only=True)
        # DATABANK file index
        self.chunks_by_path: Dict[str, List[Chunk]] = {}
        self.vectors_by_path: Dict[str, np.ndarray] = {}
        self.chunks: List[Chunk] = []
        self.vectors = np.zeros((0, 1), dtype=np.float32)
        self.file_count = 0
        self.chunk_count = 0
        # Hot memory index (separate from DATABANK)
        self.hot_chunks: List[Chunk] = []
        self.hot_vectors = np.zeros((0, 1), dtype=np.float32)
        self.hot_entries: Dict[str, Chunk] = {}
        self.hot_entry_vectors: Dict[str, np.ndarray] = {}
        self.hot_count = 0
        self.last_indexed_at = 0.0
        self.memory = MemoryStore(redis_url=REDIS_URL)

    def rebuild(self) -> None:
        chunks_by_path: Dict[str, List[Chunk]] = {}
        vectors_by_path: Dict[str, np.ndarray] = {}
        for md in markdown_files():
            rel = rel_path(md)
            chunks = chunk_doc(md, md.read_text(encoding="utf-8"))
            chunks_by_path[rel] = chunks
            vectors_by_path[rel] = embed_chunks(self.model, chunks)
        with self.lock:
            self.chunks_by_path = chunks_by_path
            self.vectors_by_path = vectors_by_path
            self.refresh_locked()

    def rebuild_hot(self) -> None:
        """Load all hot memory entries from Redis and embed them."""
        all_keys = self.memory._r.smembers("lore:hot:idx:all")
        if not all_keys:
            with self.lock:
                self.hot_entries.clear()
                self.hot_entry_vectors.clear()
                self._refresh_hot_locked()
            return

        pipe = self.memory._r.pipeline()
        key_list = list(all_keys)
        for k in key_list:
            pipe.hgetall(k)
        results = pipe.execute()

        entries: Dict[str, Chunk] = {}
        entry_vectors: Dict[str, np.ndarray] = {}
        for full_key, data in zip(key_list, results):
            if not data:
                continue
            text = _hot_entry_text(full_key, data)
            if not text:
                continue
            norm = normalize(text)
            if not norm:
                continue
            path = f"{HOT_PATH_PREFIX}{full_key}"
            chunk = Chunk(path=path, chunk_id=path, text=norm)
            entries[full_key] = chunk
            vec = embed_chunks(self.model, [chunk])
            entry_vectors[full_key] = vec

        with self.lock:
            self.hot_entries = entries
            self.hot_entry_vectors = entry_vectors
            self._refresh_hot_locked()

    def index_hot_entry(self, full_key: str, data: Dict[str, str]) -> None:
        """Embed a single hot memory entry and add to the hot index."""
        text = _hot_entry_text(full_key, data)
        if not text:
            return
        norm = normalize(text)
        if not norm:
            return
        path = f"{HOT_PATH_PREFIX}{full_key}"
        chunk = Chunk(path=path, chunk_id=path, text=norm)
        vec = embed_chunks(self.model, [chunk])
        with self.lock:
            self.hot_entries[full_key] = chunk
            self.hot_entry_vectors[full_key] = vec
            self._refresh_hot_locked()

    def remove_hot_entry(self, full_key: str) -> None:
        """Remove a hot memory entry from the hot index."""
        with self.lock:
            removed = self.hot_entries.pop(full_key, None) is not None
            self.hot_entry_vectors.pop(full_key, None)
            if removed:
                self._refresh_hot_locked()

    def _refresh_hot_locked(self) -> None:
        chunks: List[Chunk] = []
        mats: List[np.ndarray] = []
        for key in sorted(self.hot_entries.keys()):
            chunk = self.hot_entries[key]
            vec = self.hot_entry_vectors.get(key)
            if vec is None or vec.shape[0] == 0:
                continue
            chunks.append(chunk)
            mats.append(vec)
        self.hot_chunks = chunks
        self.hot_vectors = np.vstack(mats) if mats else np.zeros((0, 1), dtype=np.float32)
        self.hot_count = len(chunks)

    def reindex_file(self, rel: str) -> bool:
        rel = rel.replace("\\", "/").lstrip("/")
        fp = DOCS_ROOT / rel
        with self.lock:
            if fp.exists() and fp.suffix == ".md":
                chunks = chunk_doc(fp, fp.read_text(encoding="utf-8"))
                self.chunks_by_path[rel] = chunks
                self.vectors_by_path[rel] = embed_chunks(self.model, chunks)
                changed = True
            else:
                changed = self.chunks_by_path.pop(rel, None) is not None
                self.vectors_by_path.pop(rel, None)
            if changed:
                self.refresh_locked()
            return changed

    def refresh_locked(self) -> None:
        all_chunks: List[Chunk] = []
        mats: List[np.ndarray] = []
        for key in sorted(self.chunks_by_path.keys()):
            chunks = self.chunks_by_path[key]
            vectors = self.vectors_by_path.get(key)
            if vectors is None or vectors.shape[0] == 0:
                continue
            all_chunks.extend(chunks)
            mats.append(vectors)

        self.chunks = all_chunks
        self.vectors = np.vstack(mats) if mats else np.zeros((0, 1), dtype=np.float32)
        self.file_count = len(self.chunks_by_path)
        self.chunk_count = len(self.chunks)
        self.last_indexed_at = time.time()


def _hot_entry_text(full_key: str, data: Dict[str, str]) -> str:
    """Build searchable text from a hot memory Redis hash."""
    parts = []
    topic = full_key.rsplit(":", 1)[-1] if ":" in full_key else full_key
    parts.append(topic)
    for field in ("name", "description", "content", "body"):
        val = data.get(field, "").strip()
        if val:
            parts.append(val)
    return " ".join(parts)


def markdown_files() -> List[Path]:
    if DOCS_ROOT.exists():
        return sorted(DOCS_ROOT.rglob("*.md"))
    return []


def rel_path(path: Path) -> str:
    return path.relative_to(DOCS_ROOT).as_posix()


def normalize(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_doc(path: Path, text: str) -> List[Chunk]:
    rel = rel_path(path)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    buf = ""
    idx = 0
    for paragraph in paragraphs:
        candidate = f"{buf}\n\n{paragraph}".strip() if buf else paragraph
        if len(candidate) <= MAX_CHUNK_CHARS:
            buf = candidate
            continue
        if buf:
            norm = normalize(buf)
            if norm:
                chunks.append(Chunk(path=rel, chunk_id=f"{rel}#c{idx}", text=norm))
                idx += 1
        buf = paragraph
    if buf:
        norm = normalize(buf)
        if norm:
            chunks.append(Chunk(path=rel, chunk_id=f"{rel}#c{idx}", text=norm))
    return chunks


def embed_chunks(model: TextEmbedding, chunks: List[Chunk]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    texts = [c.text for c in chunks]
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def embed_query(model: TextEmbedding, query: str) -> np.ndarray:
    vec = np.array(list(model.embed([query]))[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        norm = 1e-12
    return vec / norm


def compact(data: Dict[str, object]) -> Response:
    return Response(
        content=json.dumps(data, separators=(",", ":"), ensure_ascii=False),
        media_type="application/json",
    )


app = FastAPI(title="Lore Memory", version="0.1.5")
LORE_TOKEN = os.getenv("LORE_TOKEN")

@app.middleware("http")
async def authorize_request(request, call_next):
    if LORE_TOKEN and request.headers.get("Authorization") != f"Bearer {LORE_TOKEN}":
        return Response(status_code=401, content="Unauthorized")
    return await call_next(request)
state = State()


@app.on_event("startup")
def startup() -> None:
    state.rebuild()
    state.rebuild_hot()


@app.post("/activity")
def record_activity(req: ActivityRequest):
    state.memory.record_access(req.path)
    return {"ok": True}

@app.get("/memory/hot")
def get_hot(limit: int = 5):
    return state.memory.get_hot_primitives(limit=limit)

@app.post("/memory/hot/write")
def hot_write(req: HotWriteRequest):
    full_key = state.memory.hot_write(
        topic=req.topic, scope=req.scope, project=req.project,
        type_=req.type, content=req.content,
        session_ref=req.session_ref,
        name=req.name, description=req.description, body=req.body,
    )
    # Index into hot search index
    data = {
        "name": req.name, "description": req.description,
        "content": req.content, "body": req.body,
    }
    state.index_hot_entry(full_key, data)
    return {"ok": True, "key": full_key}

@app.post("/memory/hot/delete")
def hot_delete(req: HotDeleteRequest):
    full_key = state.memory._scoped_key(
        req.scope, req.project, req.topic, session_id=req.session_ref,
    )
    existed = state.memory.hot_delete(
        scope=req.scope, project=req.project, topic=req.topic,
        session_id=req.session_ref,
    )
    if existed:
        state.remove_hot_entry(full_key)
    return {"ok": True, "deleted": existed, "key": full_key}

@app.get("/memory/hot/recall")
def hot_recall(limit: int = 10, scope: str = "project", project: str = "",
               session_id: str = ""):
    facts = state.memory.hot_recall(
        limit=limit, scope=scope, project=project, session_id=session_id,
    )
    return {"ok": True, "facts": facts}

@app.get("/memory/hot/stats")
def hot_stats(project: str = "", session_id: str = ""):
    stats = state.memory.hot_stats(project=project, session_id=session_id)
    return {"ok": True, "stats": stats}

@app.get("/health")
def health() -> Dict[str, object]:
    with state.lock:
        return {
            "ok": True,
            "docs_root": str(DOCS_ROOT),
            "model": EMBED_MODEL,
            "result_mode": RESULT_MODE,
            "max_k": MAX_K,
            "file_count": state.file_count,
            "chunk_count": state.chunk_count,
            "hot_count": state.hot_count,
            "last_indexed_at": state.last_indexed_at,
        }


@app.post("/reindex")
def reindex() -> Dict[str, object]:
    state.rebuild()
    return {
        "ok": True,
        "file_count": state.file_count,
        "chunk_count": state.chunk_count,
    }


@app.post("/reindex-file")
def reindex_file(req: ReindexPathRequest) -> Dict[str, object]:
    rel = req.path.replace("\\", "/").lstrip("/")
    fp = DOCS_ROOT / rel
    if not fp.resolve().is_relative_to(DOCS_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Path outside docs root")
    changed = state.reindex_file(req.path)
    return {
        "ok": True,
        "path": req.path,
        "changed": changed,
        "file_count": state.file_count,
        "chunk_count": state.chunk_count,
    }


@app.get("/search")
def search(
    q: str = Query(..., min_length=2),
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K),
    mode: str = Query(RESULT_MODE),
) -> Response:
    with state.lock:
        db_chunks = state.chunks
        db_vectors = state.vectors
        hot_chunks = state.hot_chunks
        hot_vectors = state.hot_vectors
        model = state.model

    all_chunks = db_chunks + hot_chunks
    if not all_chunks:
        return compact({"query": q, "results": []})

    # Build combined vector matrix
    mats = []
    if db_vectors.shape[0] > 0:
        mats.append(db_vectors)
    if hot_vectors.shape[0] > 0:
        mats.append(hot_vectors)
    if not mats:
        return compact({"query": q, "results": []})
    combined = np.vstack(mats)

    qv = embed_query(model, q)
    scores = combined @ qv
    top = np.argsort(scores)[::-1][:k]

    if mode == "paths_min":
        best: Dict[str, float] = {}
        for idx in top:
            chunk = all_chunks[int(idx)]
            score = float(scores[int(idx)])
            prev = best.get(chunk.path)
            if prev is None or score > prev:
                best[chunk.path] = score
        paths = [
            p
            for p, _ in sorted(best.items(), key=lambda item: item[1], reverse=True)[:k]
        ]
        return compact({"r": paths})

    results = []
    for idx in top:
        chunk = all_chunks[int(idx)]
        score = float(scores[int(idx)])
        results.append(
            {
                "path": chunk.path,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 6),
                "snippet": chunk.text[:SNIPPET_CHARS],
                "source": "hot" if chunk.path.startswith(HOT_PATH_PREFIX) else "databank",
            }
        )
    return compact({"query": q, "results": results})
