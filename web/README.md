# Live Demo — Multi-Modal Evidence Review

## Deploy on Render

The repository includes a free-tier `render.yaml` blueprint. Configure
`GEMINI_API_KEY` as a secret in Render; it is never included in the repository.
The service installs both requirements files and starts
`uvicorn web.server:app --host 0.0.0.0 --port $PORT`.
`GET /api/health` reports service health and whether the real model is configured.

The public interface includes a How it works section. Each request accepts up to
three images of 5 MB each and a 4,000-character claim. Missing credentials and
exhausted model quota produce an explicit unavailable response rather than a mock
verdict. The deployed evaluation panel uses the measured, aggregate results in
`web/evaluation_metrics.json` (20 labeled sample cases).

A web page that showcases the model: upload a damage photo, describe the claim,
and the existing pipeline (Gemini 2.5 vision + deterministic guardrails) returns
a verdict. Also shows the measured accuracy of the system.

## Run it

```bash
pip install -r code/requirements.txt
pip install -r web/requirements.txt

# make sure code/.env has GEMINI_API_KEY=... (for real verdicts)
python web/server.py
```

Then open **http://localhost:8000**.

- Drop in a photo, pick the object type (car / laptop / package), write the claim.
- Click **Analyze** → the verdict (supported / contradicted / not enough info),
  the issue type, part, severity, risk flags, and an image-grounded justification.
- The toggle "Simulate a high-risk claimant history" shows how user history adds
  `user_history_risk` / `manual_review_required` without changing the visual verdict.

## Notes
- Without an API key it runs in MOCK mode (placeholder output, clearly labelled).
- Each analysis = 1 model call. The free tier is capped at 20/day per model; the
  server uses the same failover chain as the main pipeline.
- It's a single-process demo server (`uvicorn`), meant for local showcasing.
