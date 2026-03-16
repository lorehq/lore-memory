# lore-os-services

Docker services stack for [Lore OS](https://github.com/lorehq/lore-os). Provides semantic search over the DATABANK and scoped hot memory backed by Redis.

## What's Inside

A single container runs three processes:

- **Redis** -- persistent hot cache for session, project, and global memory
- **FastAPI** -- semantic search and memory API (port 8080)
- **Watcher** -- monitors the DATABANK for file changes and triggers incremental reindexing

The embedding model (`BAAI/bge-small-en-v1.5`) is baked into the image at build time. No network access is required at runtime.

## Usage

The container is managed by the Lore OS bundle. The `docker-compose.yml` lives in the lore-os repo (`~/.lore-os/docker-compose.yml`) and is started/stopped from there.

Pull and run manually:

```
docker pull lorehq/lore-os-services:latest
docker run -d \
  -p 9184:8080 \
  -v ~/LORE-OS/DATABANK:/data/DATABANK:ro \
  -v ~/LORE-OS/HOT:/data/redis \
  lorehq/lore-os-services:latest
```

## Volumes

| Mount | Container path | Mode | Purpose |
|-------|---------------|------|---------|
| `~/LORE-OS/DATABANK` | `/data/DATABANK` | ro | Markdown files indexed for semantic search |
| `~/LORE-OS/HOT` | `/data/redis` | rw | Redis AOF persistence |

## API

All endpoints are served on port 8080 inside the container.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status, index stats |
| `GET` | `/search?q=...&k=10` | Semantic search over DATABANK |
| `POST` | `/reindex` | Full reindex of all markdown files |
| `POST` | `/reindex-file` | Incremental reindex of a single file |
| `POST` | `/activity` | Record file access for hot-path tracking |
| `POST` | `/memory/hot/write` | Write scoped memory (session/project/global) |
| `GET` | `/memory/hot/recall` | Recall scoped memory entries |
| `GET` | `/memory/hot/stats` | Memory entry counts by scope |

Set `LORE_TOKEN` to require Bearer token auth on all requests.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCS_SOURCE` | `/data/DATABANK` | Root directory for markdown indexing |
| `SEMANTIC_PORT` | `8080` | FastAPI listen port |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (must be baked in) |
| `DEFAULT_K` | `10` | Default number of search results |
| `MAX_K` | `10` | Maximum allowed k |
| `RESULT_MODE` | `paths_min` | Default search output format |
| `LORE_TOKEN` | *(none)* | Bearer token for API auth |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

## Building

```
docker build -t lorehq/lore-os-services:latest .
```

## License

Apache 2.0
