"""Evaluation entry point: run the system on dataset/sample_claims.csv and score
predictions against the gold labels. Supports comparing multiple model configs.

Usage:
  python code/evaluation/main.py                       # eval default model on sample
  python code/evaluation/main.py --models gemini-2.5-flash,gemini-2.5-flash-lite
  python code/evaluation/main.py --dry-run             # keyless mock plumbing test
  python code/evaluation/main.py --predictions out.csv # score an existing CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                 # for metrics.py
sys.path.insert(0, str(_HERE.parent))          # for evidence_review package


def _load_dotenv() -> None:
    """Load .env BEFORE importing config so EVIDENCE_*/keys take effect."""
    try:
        from dotenv import load_dotenv

        load_dotenv(_HERE.parent.parent / ".env")
        load_dotenv(_HERE.parent / ".env")
    except ImportError:
        pass


_load_dotenv()

import metrics  # noqa: E402
from evidence_review import config, data  # noqa: E402
from evidence_review.gemini_client import GeminiClient  # noqa: E402
from evidence_review.pipeline import VerdictCache, run  # noqa: E402


def _gold_rows(claims) -> list:
    rows = []
    for c in claims:
        row = {
            "user_id": c.user_id, "image_paths": c.image_paths,
            "user_claim": c.user_claim, "claim_object": c.claim_object,
        }
        row.update(c.expected)
        rows.append(row)
    return rows


def _read_predictions(path: Path) -> list:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate evidence-review predictions on the gold sample")
    p.add_argument("--input", default=str(config.SAMPLE_CLAIMS_CSV))
    p.add_argument("--models", default=config.DEFAULT_MODEL,
                   help="comma-separated model ids to run + compare")
    p.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    p.add_argument("--dry-run", action="store_true", help="keyless mock")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--predictions", default="", help="score an existing predictions CSV (skip running)")
    args = p.parse_args()

    claims = data.load_claims(Path(args.input))
    golds = _gold_rows(claims)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Score-only mode
    if args.predictions:
        preds = _read_predictions(Path(args.predictions))
        report = metrics.score(preds, golds)
        print(metrics.format_report(report))
        return 0

    has_key = bool(config.get_api_key())
    mock = args.dry_run or not has_key
    if mock and not args.dry_run:
        print("WARNING: no API key found -> MOCK mode (placeholder outputs, not real judgments).\n")

    summary = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== Evaluating model: {model}{'  [MOCK]' if mock else ''} =====")
        client = GeminiClient(model, mock=mock)
        cache = None if args.no_cache else VerdictCache(config.RUNS_DIR / f"cache_{model}.json")
        out_csv = config.RUNS_DIR / f"sample_pred_{model}.csv"
        preds, usage = run(claims, out_csv, client, cache=cache, max_workers=args.workers, progress=True)
        report = metrics.score(preds, golds)
        print("\n" + metrics.format_report(report))
        stats = usage.as_dict()
        (config.RUNS_DIR / f"eval_report_{model}.json").write_text(
            json.dumps({"model": model, "mock": mock, "usage": stats, "metrics": report}, indent=2),
            encoding="utf-8",
        )
        summary[model] = {
            "headline_claim_status_accuracy": report["headline_claim_status_accuracy"],
            "row_exact_accuracy": report["row_exact_accuracy"],
            "usage": stats,
        }

    if len(summary) > 1:
        print("\n===== MODEL COMPARISON =====")
        print(f"{'model':<26}{'claim_status_acc':<18}{'row_exact':<12}{'input_tok':<12}{'output_tok':<12}")
        for m, s in summary.items():
            print(f"{m:<26}{s['headline_claim_status_accuracy']:<18.1%}"
                  f"{s['row_exact_accuracy']:<12.1%}"
                  f"{s['usage']['input_tokens']:<12}{s['usage']['output_tokens']:<12}")
    (config.RUNS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote eval artifacts to {config.RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
