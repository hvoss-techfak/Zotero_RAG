#!/usr/bin/env python3
"""Smoke test for embedded_docs.json atomic/thread-safe writes.

This doesn't hit Zotero/Ollama. It just stresses VectorStore metadata IO
from multiple threads to ensure we never corrupt JSON.

Usage:
    PYTHONPATH=src python scripts/concurrency_smoke_test.py
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from semtero.vector_store import VectorStore


def main() -> None:
    store_dir = Path("/tmp/zoterorag_vector_store_smoke")
    store_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort cleanup of old metadata
    meta = store_dir / "embedded_docs.json"
    for p in [meta, store_dir / "embedded_docs.json.tmp", store_dir / "embedded_docs.json.bak"]:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    vs = VectorStore(persist_directory=str(store_dir))

    n_threads = 16
    n_updates_per_thread = 200

    def worker(tid: int) -> None:
        for i in range(n_updates_per_thread):
            vs.update_embedded_document(f"T{tid}", i)

    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(n_threads)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    meta_path = store_dir / "embedded_docs.json"
    assert meta_path.exists(), "embedded_docs.json missing"

    raw = meta_path.read_text().strip()
    parsed = json.loads(raw)

    missing = [k for k in (f"T{i}" for i in range(n_threads)) if k not in parsed]
    assert not missing, f"Missing keys: {missing}"

    dur = time.time() - start
    print(f"OK: {n_threads} threads x {n_updates_per_thread} updates, JSON valid, {dur:.2f}s")


if __name__ == "__main__":
    main()

