"""Scoring helpers for evaluating predictions against gold sample labels."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

# Fields scored by exact categorical match.
CATEGORICAL_FIELDS = [
    "evidence_standard_met",
    "claim_status",
    "issue_type",
    "object_part",
    "valid_image",
    "severity",
]
# Fields scored as semicolon-separated sets (order-insensitive).
SET_FIELDS = ["risk_flags", "supporting_image_ids"]


def _norm(v: str) -> str:
    return (v or "").strip().lower()


def _as_set(v: str) -> set:
    v = _norm(v)
    if not v or v == "none":
        return set()
    return {x.strip() for x in v.split(";") if x.strip()}


def set_prf(pred: str, gold: str) -> Tuple[float, float, float]:
    p, g = _as_set(pred), _as_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    tp = len(p & g)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def score(preds: List[Dict[str, str]], golds: List[Dict[str, str]]) -> Dict:
    """Compare aligned prediction/gold rows. Returns a metrics report dict."""
    n = len(golds)
    cat_correct = {f: 0 for f in CATEGORICAL_FIELDS}
    set_f1_sum = {f: 0.0 for f in SET_FIELDS}
    set_prec_sum = {f: 0.0 for f in SET_FIELDS}
    set_rec_sum = {f: 0.0 for f in SET_FIELDS}
    confusion = defaultdict(int)  # (gold, pred) -> count for claim_status
    row_exact = 0
    per_row = []

    for pred, gold in zip(preds, golds):
        row_ok = True
        diffs = {}
        for f in CATEGORICAL_FIELDS:
            ok = _norm(pred.get(f)) == _norm(gold.get(f))
            cat_correct[f] += int(ok)
            if not ok:
                row_ok = False
                diffs[f] = (gold.get(f), pred.get(f))
        for f in SET_FIELDS:
            pr, rc, f1 = set_prf(pred.get(f, ""), gold.get(f, ""))
            set_prec_sum[f] += pr
            set_rec_sum[f] += rc
            set_f1_sum[f] += f1
            if f1 < 1.0:
                row_ok = False
                diffs[f] = (gold.get(f), pred.get(f))
        confusion[(_norm(gold.get("claim_status")), _norm(pred.get("claim_status")))] += 1
        row_exact += int(row_ok)
        per_row.append({
            "user_id": gold.get("user_id"),
            "case": _case(gold.get("image_paths", "")),
            "claim_status_gold": gold.get("claim_status"),
            "claim_status_pred": pred.get("claim_status"),
            "ok": row_ok,
            "diffs": diffs,
        })

    report = {
        "n": n,
        "categorical_accuracy": {f: round(cat_correct[f] / n, 3) for f in CATEGORICAL_FIELDS},
        "set_f1": {f: round(set_f1_sum[f] / n, 3) for f in SET_FIELDS},
        "set_precision": {f: round(set_prec_sum[f] / n, 3) for f in SET_FIELDS},
        "set_recall": {f: round(set_rec_sum[f] / n, 3) for f in SET_FIELDS},
        "row_exact_accuracy": round(row_exact / n, 3),
        "claim_status_confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
        "headline_claim_status_accuracy": round(cat_correct["claim_status"] / n, 3),
        "per_row": per_row,
    }
    return report


def _case(image_paths: str) -> str:
    for token in image_paths.split("/"):
        if token.startswith("case_"):
            return token
    return ""


def format_report(report: Dict) -> str:
    lines = []
    lines.append(f"n = {report['n']}")
    lines.append(f"HEADLINE claim_status accuracy: {report['headline_claim_status_accuracy']:.1%}")
    lines.append(f"row-exact accuracy (all categorical+set fields): {report['row_exact_accuracy']:.1%}")
    lines.append("\nPer-field categorical accuracy:")
    for f, v in report["categorical_accuracy"].items():
        lines.append(f"  {f:<24} {v:.1%}")
    lines.append("\nSet-field F1 (precision / recall):")
    for f in SET_FIELDS:
        lines.append(f"  {f:<24} F1={report['set_f1'][f]:.3f}  "
                     f"(P={report['set_precision'][f]:.3f} / R={report['set_recall'][f]:.3f})")
    lines.append("\nclaim_status confusion (gold->pred):")
    for k, c in report["claim_status_confusion"].items():
        lines.append(f"  {k:<48} {c}")
    wrong = [r for r in report["per_row"] if not r["ok"]]
    if wrong:
        lines.append(f"\nRows with diffs ({len(wrong)}):")
        for r in wrong:
            lines.append(f"  {r['case']} ({r['user_id']}): "
                         f"claim_status {r['claim_status_gold']}->{r['claim_status_pred']}")
            for f, (g, p) in r["diffs"].items():
                if f != "claim_status":
                    lines.append(f"      {f}: gold={g!r} pred={p!r}")
    return "\n".join(lines)
