"""Per-claim orchestration: load images, call the model, apply guardrails, and
write output.csv. Includes a verdict cache and bounded concurrency.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config, data, images, postprocess, schema
from .data import Claim, UserHistory
from .gemini_client import GeminiClient, Usage
from .schema import VisionVerdict


class VerdictCache:
    """Disk-backed cache keyed by (model, claim content) to avoid repeat calls."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        self._store: Dict[str, dict] = {}
        if path and path.exists():
            try:
                self._store = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._store = {}

    @staticmethod
    def key(model: str, claim: Claim) -> str:
        h = hashlib.sha256()
        h.update(f"{model}\0{claim.user_id}\0{claim.image_paths}\0{claim.user_claim}".encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> Optional[VisionVerdict]:
        with self._lock:
            raw = self._store.get(key)
        return VisionVerdict(**raw) if raw else None

    def put(self, key: str, verdict: VisionVerdict) -> None:
        with self._lock:
            self._store[key] = verdict.model_dump()

    def flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.path.write_text(json.dumps(self._store, ensure_ascii=False, indent=0), encoding="utf-8")


def process_claim(
    claim: Claim,
    history_map: Dict[str, UserHistory],
    reqs,
    client: GeminiClient,
    cache: Optional[VerdictCache] = None,
) -> Tuple[Dict[str, str], Usage]:
    history = history_map.get(claim.user_id)
    loaded = images.load_claim_images(claim.image_path_list)

    if not any(im.ok for im in loaded):
        return postprocess.unreadable_row(claim, history), Usage()

    object_reqs = data.requirements_for_object(reqs, claim.claim_object)
    from . import prompts
    text_context = prompts.build_text_context(claim, history, object_reqs, loaded)

    usage = Usage()
    verdict: Optional[VisionVerdict] = None
    cache_id = f"{client.model}@{client.prompt_fingerprint}"
    if cache is not None:
        ck = cache.key(cache_id, claim)
        verdict = cache.get(ck)
    if verdict is None:
        verdict, usage = client.verdict(text_context, loaded)
        if cache is not None and not client.mock:
            cache.put(cache.key(cache_id, claim), verdict)

    row = postprocess.finalize(claim, verdict, history, loaded)
    return row, usage


def run(
    claims: List[Claim],
    output_path: Path,
    client: GeminiClient,
    cache: Optional[VerdictCache] = None,
    max_workers: int = config.MAX_WORKERS,
    progress: bool = True,
) -> Tuple[List[Dict[str, str]], Usage]:
    history_map = data.load_user_history()
    reqs = data.load_evidence_requirements()

    results: List[Optional[Dict[str, str]]] = [None] * len(claims)
    total_usage = Usage()
    done = 0
    lock = threading.Lock()

    def work(idx: int, claim: Claim):
        return idx, process_claim(claim, history_map, reqs, client, cache)

    workers = 1 if client.mock else max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, i, c) for i, c in enumerate(claims)]
        for fut in as_completed(futures):
            idx, (row, usage) = fut.result()
            results[idx] = row
            with lock:
                total_usage.add(usage)
                done += 1
                if progress:
                    print(f"  [{done}/{len(claims)}] {claims[idx].case_id or claims[idx].user_id} "
                          f"-> {row['claim_status']}", flush=True)

    rows = [r for r in results if r is not None]
    write_output(rows, output_path)
    if cache is not None:
        cache.flush()
    return rows, total_usage


def write_output(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=schema.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in schema.OUTPUT_COLUMNS})
