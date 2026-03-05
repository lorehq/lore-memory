import json
import os
import threading
import urllib.request
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver


DATABANK = Path(os.getenv("DOCS_SOURCE", "/data/DATABANK"))
SEMANTIC_PORT = int(os.getenv("SEMANTIC_PORT", "8080"))
REINDEX_DEBOUNCE = float(os.getenv("WATCH_REINDEX_DEBOUNCE_SECONDS", "1.0"))
POLL_INTERVAL = float(os.getenv("WATCH_POLL_INTERVAL_SECONDS", "1.0"))


def _is_actually_dir(path: Path) -> bool:
    """Check if a path is a directory on the source filesystem.

    Docker Desktop VirtioFS can report directory events with is_directory=False.
    Always verify against the actual filesystem.
    """
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def rel_from_source(path: Path):
    try:
        return path.resolve().relative_to(DATABANK.resolve()).as_posix()
    except Exception:
        return None


class RuntimeWatcher(FileSystemEventHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.reindex_timer = None
        self.pending_md = set()

    def on_any_event(self, event):
        event_type = event.event_type
        if event_type not in ("created", "modified", "deleted", "moved"):
            return

        src = Path(getattr(event, "src_path", ""))
        dest = (
            Path(getattr(event, "dest_path", ""))
            if hasattr(event, "dest_path")
            else None
        )

        is_dir = event.is_directory
        if not is_dir and event_type != "deleted":
            is_dir = _is_actually_dir(src)

        src_rel = rel_from_source(src)
        dst_rel = rel_from_source(dest) if dest else None

        print(
            f"[watcher] {event_type}: {src_rel or src} (dir={is_dir})",
            flush=True,
        )

        if is_dir:
            self._handle_dir_event(event_type, src_rel, dst_rel, src)
            return

        self._handle_file_event(event_type, src_rel, dst_rel)

    def _handle_dir_event(self, event_type: str, src_rel, dst_rel, src_path: Path = None):
        if event_type == "created" and src_rel:
            src = DATABANK / src_rel
            if src.exists() and src.is_dir():
                for md in src.rglob("*.md"):
                    try:
                        md_rel = md.relative_to(DATABANK).as_posix()
                        with self.lock:
                            self.pending_md.add(md_rel)
                    except ValueError:
                        pass
                self._schedule_reindex()
        elif event_type == "deleted" and src_rel:
            with self.lock:
                self.pending_md.add(src_rel)
            self._schedule_reindex()
        elif event_type == "moved":
            if src_rel:
                with self.lock:
                    self.pending_md.add(src_rel)
            if dst_rel:
                dst_src = DATABANK / dst_rel
                if dst_src.exists() and dst_src.is_dir():
                    for md in dst_src.rglob("*.md"):
                        try:
                            md_rel = md.relative_to(DATABANK).as_posix()
                            with self.lock:
                                self.pending_md.add(md_rel)
                        except ValueError:
                            pass
            self._schedule_reindex()

    def _handle_file_event(self, event_type: str, src_rel, dst_rel):
        if event_type in ("created", "modified") and src_rel:
            if src_rel.endswith(".md"):
                with self.lock:
                    self.pending_md.add(src_rel)
                self._schedule_reindex()
        elif event_type == "deleted" and src_rel:
            if src_rel.endswith(".md"):
                with self.lock:
                    self.pending_md.add(src_rel)
                self._schedule_reindex()
        elif event_type == "moved":
            if src_rel and src_rel.endswith(".md"):
                with self.lock:
                    self.pending_md.add(src_rel)
            if dst_rel and dst_rel.endswith(".md"):
                with self.lock:
                    self.pending_md.add(dst_rel)
            if (src_rel and src_rel.endswith(".md")) or (
                dst_rel and dst_rel.endswith(".md")
            ):
                self._schedule_reindex()

    def _schedule_reindex(self):
        with self.lock:
            if self.reindex_timer:
                self.reindex_timer.cancel()
            self.reindex_timer = threading.Timer(REINDEX_DEBOUNCE, self._flush_reindex)
            self.reindex_timer.daemon = True
            self.reindex_timer.start()

    def _flush_reindex(self):
        with self.lock:
            paths = sorted(self.pending_md)
            self.pending_md.clear()

        for rel in paths:
            payload = json.dumps({"path": rel}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{SEMANTIC_PORT}/reindex-file",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5):
                    pass
            except Exception as exc:
                print(f"[watcher] reindex-file failed for {rel}: {exc}", flush=True)


def run():
    DATABANK.mkdir(parents=True, exist_ok=True)

    handler = RuntimeWatcher()
    observer = PollingObserver(timeout=POLL_INTERVAL)
    observer.schedule(handler, str(DATABANK), recursive=True)
    observer.daemon = True
    observer.start()

    print(f"[watcher] started (polling, interval={POLL_INTERVAL}s)", flush=True)

    try:
        observer.join()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join(timeout=2)


if __name__ == "__main__":
    run()
