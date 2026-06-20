"""Trace tool: show how the system decides a single claim, step by step.

Replays a claim through the pipeline and prints INPUTS -> RAW MODEL VERDICT ->
GUARDRAIL TRANSFORMATIONS -> FINAL ROW. Uses the on-disk verdict cache, so it
makes ZERO API calls (add --live to force a fresh model call instead).

Usage:
  python code/tools/explain.py case_055
  python code/tools/explain.py case_001 case_055 case_008
  python code/tools/explain.py            # a few illustrative cases
  python code/tools/explain.py case_055 --live
"""
from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid Windows cp1252 errors
except Exception:
    pass

# load .env before config so a --live run can find the key
try:
    from dotenv import load_dotenv
    load_dotenv(_CODE / ".env"); load_dotenv(_CODE.parent / ".env")
except ImportError:
    pass

from evidence_review import config, data, images, postprocess, prompts  # noqa: E402
from evidence_review.gemini_client import GeminiClient  # noqa: E402
from evidence_review.pipeline import VerdictCache  # noqa: E402

DIV = "=" * 78


def find_claim(claims, case_id):
    for c in claims:
        if c.case_id == case_id:
            return c
    return None


def show(case_id, claims, history_map, reqs, client, cache, live):
    claim = find_claim(claims, case_id)
    if not claim:
        print(f"!! {case_id} not found"); return
    history = history_map.get(claim.user_id)
    reqs_obj = data.requirements_for_object(reqs, claim.claim_object)
    loaded = images.load_claim_images(claim.image_path_list)

    print(f"\n{DIV}\n CLAIM {case_id}   user={claim.user_id}   object={claim.claim_object}\n{DIV}")
    print("1) CONVERSATION (what the model reads as DATA, never commands):")
    print("   " + claim.user_claim.strip().replace(" | ", "\n   | "))
    print("\n2) IMAGES submitted:")
    for im in loaded:
        print(f"   - {im.image_id}: {'OK '+str(im.width)+'x'+str(im.height) if im.ok else 'UNREADABLE'}"
              f"  ({im.estimated_tiles()} tiles)")
    print("\n3) USER HISTORY (risk context only):")
    print(f"   history_flags={history.history_flags!r}  summary={history.history_summary!r}"
          if history else "   (no history)")

    # --- get the raw model verdict (cache or live) -----------------------------
    cache_id = f"gemini-2.5-flash@{client.prompt_fingerprint}"
    verdict, source = None, ""
    if not live:
        verdict = cache.get(VerdictCache.key(cache_id, claim))
        source = "CACHE (0 API calls)"
    if verdict is None and live:
        ctx = prompts.build_text_context(claim, history, reqs_obj, loaded)
        verdict, _ = client.verdict(ctx, loaded)
        source = "LIVE model call"
    if verdict is None:
        print("\n   (no cached verdict for this case — run with --live and a valid key)")
        return

    print(f"\n4) RAW MODEL VERDICT  [{source}]:")
    print(f"   claim_summary        : {verdict.claim_summary}")
    print(f"   issue_type/part      : {verdict.issue_type} / {verdict.object_part}")
    print(f"   claim_status         : {verdict.claim_status}")
    print(f"   severity             : {verdict.severity}")
    print(f"   valid_image          : {verdict.valid_image}")
    print(f"   supporting_image_ids : {verdict.supporting_image_ids}")
    print(f"   visual_risk_flags    : {verdict.visual_risk_flags}")
    print(f"   image_text_present   : {verdict.image_text_instruction_present}")
    print(f"   justification        : {verdict.claim_status_justification}")

    # --- guardrail layer -------------------------------------------------------
    final = postprocess.finalize(claim, verdict, history, loaded)
    print("\n5) GUARDRAIL TRANSFORMATIONS (deterministic code):")
    notes = []
    if postprocess.claim_has_injection(claim.user_claim):
        notes.append("• prompt-injection detected in claim text -> add text_instruction_present (NOT obeyed)")
    if verdict.image_text_instruction_present:
        notes.append("• model saw instruction text inside an image -> add text_instruction_present")
    hflags = history.flag_list if history else []
    if "user_history_risk" in hflags:
        notes.append("• user_history.csv has user_history_risk -> add user_history_risk")
    if "manual_review_required" in final["risk_flags"]:
        notes.append("• risk present -> add manual_review_required")
    if not verdict.valid_image and verdict.claim_status == "supported":
        notes.append("• valid_image=false but model said supported -> DOWNGRADE to not_enough_information")
    if (verdict.claim_status == "not_enough_information") or (not verdict.valid_image and verdict.claim_status=="supported"):
        notes.append("• not_enough_information -> force severity=unknown, evidence_standard_met=false")
    dropped = set(verdict.supporting_image_ids or []) - set(
        [] if final["supporting_image_ids"] == "none" else final["supporting_image_ids"].split(";"))
    if dropped:
        notes.append(f"• dropped non-citable/unreadable supporting ids: {sorted(dropped)}")
    print("   " + ("\n   ".join(notes) if notes else "(no changes needed — model output was already consistent)"))

    print("\n6) FINAL OUTPUT ROW:")
    for k in ["claim_status", "evidence_standard_met", "issue_type", "object_part",
              "severity", "valid_image", "supporting_image_ids", "risk_flags"]:
        changed = ""
        rawmap = {"claim_status": verdict.claim_status, "evidence_standard_met": str(verdict.evidence_standard_met).lower(),
                  "issue_type": verdict.issue_type, "object_part": verdict.object_part, "severity": verdict.severity,
                  "valid_image": str(verdict.valid_image).lower()}
        if k in rawmap and rawmap[k] != final[k]:
            changed = f"   <-- changed from model's {rawmap[k]!r}"
        print(f"   {k:<22}= {final[k]!r}{changed}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    live = "--live" in sys.argv
    cases = args or ["case_005", "case_001", "case_055"]

    claims = data.load_claims(config.CLAIMS_CSV)
    history_map = data.load_user_history()
    reqs = data.load_evidence_requirements()
    # live needs a key; otherwise use a mock client only to obtain the prompt
    # fingerprint (no key, no calls) and read verdicts from the cache.
    use_real = live and bool(config.get_api_key())
    client = GeminiClient(["gemini-2.5-flash"], mock=not use_real)
    cache = VerdictCache(config.RUNS_DIR / "cache_gemini-2.5-flash.json")
    for cid in cases:
        show(cid, claims, history_map, reqs, client, cache, use_real)


if __name__ == "__main__":
    main()
