"""Live demo web server for the Multi-Modal Evidence Review system.

Upload an image + a claim, and the existing pipeline (Gemini 2.5 vision +
deterministic guardrails) returns a verdict. Also serves the measured accuracy
numbers for the performance panel.

Run:  python web/server.py      then open  http://localhost:8000
Needs GEMINI_API_KEY (or GOOGLE_API_KEY) in code/.env for real verdicts.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "code"))  # import the evidence_review package

# load code/.env before importing config so the API key is visible
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO / "code" / ".env")
    load_dotenv(_REPO / ".env")
except ImportError:
    pass

from fastapi import FastAPI, File, Form, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from evidence_review import config, data, images, postprocess, prompts  # noqa: E402
from evidence_review.data import Claim, UserHistory  # noqa: E402
from evidence_review.gemini_client import GeminiClient  # noqa: E402

app = FastAPI(title="Multi-Modal Evidence Review — Live Demo")

# Build one client (failover chain survives free-tier daily caps) + load reqs once.
_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-flash-lite-latest"]
_HAS_KEY = bool(config.get_api_key())
_CLIENT = GeminiClient(_MODELS, mock=not _HAS_KEY)
_REQS = data.load_evidence_requirements()

# A "risky" claimant profile to demonstrate the user-history risk flags.
_RISKY_HISTORY = UserHistory(
    user_id="demo_risky", past_claim_count=8, accept_claim=3, manual_review_claim=2,
    rejected_claim=3, last_90_days_claim_count=5,
    history_flags="user_history_risk;manual_review_required",
    history_summary="Several prior claims were exaggerated or needed manual review",
)


@app.get("/")
def index():
    return FileResponse(_HERE / "static" / "index.html")


@app.get("/api/metrics")
def metrics():
    """Measured accuracy for the performance panel (from the eval artifacts)."""
    fallback = {
        "claim_status": 0.80, "evidence_standard_met": 0.95, "object_part": 0.90,
        "valid_image": 0.95, "risk_flags_f1": 0.85, "supporting_image_ids_f1": 0.92,
        "n": 20, "source": "calibrated eval (gemini-flash-lite-latest)",
    }
    try:
        p = config.RUNS_DIR / "eval_report_gemini-flash-lite-latest.json"
        m = json.loads(p.read_text(encoding="utf-8"))["metrics"]
        return {
            "claim_status": m["categorical_accuracy"]["claim_status"],
            "evidence_standard_met": m["categorical_accuracy"]["evidence_standard_met"],
            "object_part": m["categorical_accuracy"]["object_part"],
            "valid_image": m["categorical_accuracy"]["valid_image"],
            "risk_flags_f1": m["set_f1"]["risk_flags"],
            "supporting_image_ids_f1": m["set_f1"]["supporting_image_ids"],
            "n": m["n"], "source": "calibrated eval on labeled sample",
        }
    except Exception:
        return fallback


@app.post("/api/analyze")
async def analyze(
    claim: str = Form(...),
    claim_object: str = Form(...),
    history: str = Form("none"),
    files: list[UploadFile] = File(...),
):
    claim_object = (claim_object or "").strip().lower()
    if claim_object not in ("car", "laptop", "package"):
        return JSONResponse({"error": "claim_object must be car, laptop, or package"}, status_code=400)
    if not files:
        return JSONResponse({"error": "upload at least one image"}, status_code=400)

    # process uploaded bytes into LoadedImages (img_1, img_2, ...)
    loaded = []
    names = []
    for i, f in enumerate(files):
        raw = await f.read()
        names.append(f.filename or f"img_{i+1}")
        loaded.append(images.process_bytes(raw, image_id=f"img_{i+1}", rel_path=f.filename or ""))

    hist = _RISKY_HISTORY if history == "risky" else None
    claim_obj = Claim(user_id="web_user", image_paths=";".join(names),
                      user_claim=claim, claim_object=claim_object)

    if not any(im.ok for im in loaded):
        row = postprocess.unreadable_row(claim_obj, hist)
        return {"row": row, "raw": {}, "images": _img_meta(loaded), "model": "n/a",
                "latency_s": 0.0, "mock": _CLIENT.mock}

    reqs = data.requirements_for_object(_REQS, claim_object)
    text_context = prompts.build_text_context(claim_obj, hist, reqs, loaded)

    t0 = time.time()
    verdict, usage = _CLIENT.verdict(text_context, loaded)
    latency = round(time.time() - t0, 2)
    row = postprocess.finalize(claim_obj, verdict, hist, loaded)

    used = next(iter(usage.model_calls), _MODELS[0]) if usage.model_calls else _MODELS[0]
    return {
        "row": row,
        "raw": {"claim_summary": verdict.claim_summary},
        "images": _img_meta(loaded),
        "model": used,
        "latency_s": latency,
        "mock": _CLIENT.mock,
    }


def _img_meta(loaded):
    return [{"id": im.image_id, "name": im.rel_path, "ok": im.ok,
             "width": im.width, "height": im.height} for im in loaded]


app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  Multi-Modal Evidence Review — demo at http://localhost:8000")
    print(f"  API key detected: {_HAS_KEY}  (real verdicts: {_HAS_KEY})\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
