"""Entry point: run the evidence-review system over a claims CSV and write output.csv.

Usage:
  python code/main.py                      # run on dataset/claims.csv -> ./output.csv
  python code/main.py --input dataset/sample_claims.csv --output sample_output.csv
  python code/main.py --model gemini-2.5-flash-lite
  python code/main.py --dry-run            # keyless mock (no API calls)

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment unless --dry-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))  # allow `python code/main.py` from anywhere


def _load_dotenv() -> None:
    """Load .env BEFORE importing config so EVIDENCE_*/keys take effect."""
    try:
        from dotenv import load_dotenv

        load_dotenv(_CODE_DIR.parent / ".env")
        load_dotenv(_CODE_DIR / ".env")
    except ImportError:
        pass


_load_dotenv()

from evidence_review import config, data  # noqa: E402
from evidence_review.gemini_client import GeminiClient  # noqa: E402
from evidence_review.pipeline import VerdictCache, run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-Modal Evidence Review runner")
    p.add_argument("--input", default=str(config.CLAIMS_CSV), help="claims CSV to process")
    p.add_argument("--output", default=str(config.DEFAULT_OUTPUT_CSV), help="output CSV path")
    p.add_argument("--model", default=config.DEFAULT_MODEL,
                   help="Gemini model id, or comma-separated failover chain "
                        "(e.g. gemini-flash-latest,gemini-flash-lite-latest)")
    p.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    p.add_argument("--min-interval", type=float, default=config.MIN_REQUEST_INTERVAL_S,
                   help="min seconds between request starts (RPM pacing; ~5 for free tier)")
    p.add_argument("--limit", type=int, default=0, help="process only the first N rows (0=all)")
    p.add_argument("--dry-run", action="store_true", help="keyless deterministic mock (no API calls)")
    p.add_argument("--no-cache", action="store_true", help="disable the verdict cache")
    args = p.parse_args()
    models = [m.strip() for m in args.model.split(",") if m.strip()]

    has_key = bool(config.get_api_key())
    mock = args.dry_run or not has_key
    if mock and not args.dry_run:
        print("WARNING: no GEMINI_API_KEY/GOOGLE_API_KEY found -> falling back to keyless MOCK mode.")
        print("         Outputs are placeholders, NOT real model judgments. Set a key for a real run.\n")

    claims = data.load_claims(Path(args.input))
    if args.limit:
        claims = claims[: args.limit]

    print(f"Model chain: {models}{'  [MOCK]' if mock else ''}")
    print(f"Input: {args.input}  ({len(claims)} claims)")
    print(f"Output: {args.output}\n")

    client = GeminiClient(models, mock=mock, min_interval_s=args.min_interval)
    cache_path = None if args.no_cache else config.RUNS_DIR / f"cache_{models[0]}.json"
    cache = VerdictCache(cache_path)

    t0 = time.time()
    rows, usage = run(claims, Path(args.output), client, cache=cache, max_workers=args.workers)
    elapsed = time.time() - t0

    stats = usage.as_dict()
    stats.update({
        "model_chain": models, "mock": mock, "rows": len(rows),
        "wall_clock_s": round(elapsed, 1),
    })
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = config.RUNS_DIR / f"run_stats_{Path(args.input).stem}_{models[0]}.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nDone in {elapsed:.1f}s. Wrote {len(rows)} rows to {args.output}")
    print(f"Usage: {json.dumps(stats)}")
    print(f"Run stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
