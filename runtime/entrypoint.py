import os
import signal
import subprocess
import sys
import time


SEMANTIC_PORT = int(os.getenv("SEMANTIC_PORT", "8080"))


def spawn(name: str, cmd):
    print(f"[runtime] starting {name}: {' '.join(cmd)}", flush=True)
    return name, subprocess.Popen(cmd)


def stop_all(procs):
    for _name, proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 8
    for _name, proc in procs:
        if proc.poll() is not None:
            continue
        timeout = max(0, deadline - time.time())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    procs = []
    procs.append(
        spawn("redis", [
            "redis-server",
            "--dir", "/data/redis",
            "--appendonly", "yes",
            "--protected-mode", "no",
            "--bind", "0.0.0.0",
        ])
    )
    procs.append(
        spawn(
            "semantic",
            [
                "uvicorn",
                "semantic_server:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(SEMANTIC_PORT),
                "--app-dir",
                "/runtime",
            ],
        )
    )
    procs.append(spawn("watcher", ["python", "/runtime/watcher.py"]))

    def on_signal(_sig, _frame):
        stop_all(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    while True:
        for name, proc in procs:
            code = proc.poll()
            if code is not None:
                print(f"[runtime] {name} exited with code {code}", flush=True)
                stop_all(procs)
                return code
        time.sleep(0.5)


if __name__ == "__main__":
    raise SystemExit(main())
