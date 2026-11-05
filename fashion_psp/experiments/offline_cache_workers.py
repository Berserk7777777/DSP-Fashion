"""Process dispatch helpers without model or framework dependencies."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time


def task_indices(total, worker_index, worker_count):
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid task worker index/count")
    return range(worker_index, total, worker_count)


def run_workers(commands, output_root, cwd):
    """Run disjoint worker commands, with logs and cleanup on error/interruption."""
    log_root = Path(output_root) / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    processes, handles = [], []
    try:
        for index, command in commands:
            path = log_root / f"{run_id}_worker_{index:02d}.log"
            handle = path.open("w", encoding="utf-8")
            handles.append(handle)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            process = subprocess.Popen(
                command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((index, process, path))
            print(f"worker={index} pid={process.pid} log={path}", flush=True)
        pending = {index for index, _, _ in processes}
        while pending:
            for index, process, path in processes:
                if index not in pending:
                    continue
                code = process.poll()
                if code is None:
                    continue
                pending.remove(index)
                if code:
                    raise RuntimeError(f"Worker {index} failed (exit {code}); see {path}")
                print(f"worker={index} completed", flush=True)
            if pending:
                time.sleep(.5)
    finally:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
