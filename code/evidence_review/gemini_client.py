"""Gemini vision client: structured output, rate limiting, daily-quota-aware
model failover, retries, token accounting, and a keyless deterministic mock.

The client accepts an ORDERED list of model ids. Within a single run it paces
requests (RPM), and when a model returns a *per-day* quota 429 it marks that model
exhausted and fails the claim over to the next model in the list. Per-minute /
per-token 429s are retried on the same model after the server-suggested delay.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config
from . import prompts
from .images import LoadedImage
from .schema import VisionVerdict

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_TOKENS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE", "overloaded")
_RETRY_DELAY_RE = re.compile(r"retryDelay': '(\d+)s|retry in ([\d.]+)s")
_PERDAY_RE = re.compile(r"PerDay", re.IGNORECASE)


class RateLimiter:
    """Thread-safe minimum-interval limiter (paces request *starts*)."""

    def __init__(self, min_interval_s: float):
        self.min_interval = max(0.0, min_interval_s)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait = self._next_at - now
            start = now if wait <= 0 else self._next_at
            self._next_at = start + self.min_interval
        if wait > 0:
            time.sleep(wait)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    images: int = 0
    image_tiles: int = 0
    retries: int = 0
    errors: int = 0
    latencies: List[float] = field(default_factory=list)
    model_calls: Dict[str, int] = field(default_factory=dict)

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens
        self.images += other.images
        self.image_tiles += other.image_tiles
        self.retries += other.retries
        self.errors += other.errors
        self.latencies.extend(other.latencies)
        for m, c in other.model_calls.items():
            self.model_calls[m] = self.model_calls.get(m, 0) + c

    def as_dict(self) -> Dict:
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "images": self.images,
            "image_tiles": self.image_tiles,
            "retries": self.retries,
            "errors": self.errors,
            "model_calls": self.model_calls,
            "latency_p50_s": round(p50, 2),
            "latency_p95_s": round(p95, 2),
            "latency_total_s": round(sum(self.latencies), 1),
        }


class GeminiClient:
    def __init__(self, models, mock: bool = False, min_interval_s: float = None):
        if isinstance(models, str):
            models = [models]
        self.models: List[str] = list(models)
        self.model = self.models[0]  # primary, for display/run-stats
        self.mock = mock
        self._client = None
        self._types = None
        self._system = prompts.SYSTEM_PROMPT + "\n\n" + prompts.fewshot_block()
        self.prompt_fingerprint = hashlib.sha256(self._system.encode("utf-8")).hexdigest()[:8]
        self._exhausted: set = set()         # models that hit a per-day quota this run
        self._exhausted_lock = threading.Lock()
        interval = config.MIN_REQUEST_INTERVAL_S if min_interval_s is None else min_interval_s
        self._limiter = RateLimiter(interval)
        if not mock:
            self._init_real()

    def _init_real(self) -> None:
        api_key = config.get_api_key()
        if not api_key:
            raise RuntimeError(
                "No GEMINI_API_KEY / GOOGLE_API_KEY set. Run with --dry-run to use the mock."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("google-genai not installed. `pip install google-genai`.") from e
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def available_models(self) -> List[str]:
        with self._exhausted_lock:
            return [m for m in self.models if m not in self._exhausted]

    def _mark_exhausted(self, model: str) -> None:
        with self._exhausted_lock:
            self._exhausted.add(model)

    # -- public API -----------------------------------------------------------
    def verdict(self, text_context: str, loaded_images: List[LoadedImage]) -> Tuple[VisionVerdict, Usage]:
        if self.mock:
            return self._mock_verdict(text_context, loaded_images)
        return self._real_verdict(text_context, loaded_images)

    # -- real call ------------------------------------------------------------
    def _build_contents(self, text_context: str, usable: List[LoadedImage]):
        types = self._types
        parts = [types.Part.from_text(text=text_context)]
        for im in usable:
            parts.append(types.Part.from_text(text=f"[image_id: {im.image_id}]"))
            parts.append(types.Part.from_bytes(data=im.data, mime_type=im.mime_type))
        return [types.Content(role="user", parts=parts)]

    def _gen_config(self, model: str):
        types = self._types
        return types.GenerateContentConfig(
            system_instruction=self._system,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            response_mime_type="application/json",
            response_schema=VisionVerdict,
            thinking_config=types.ThinkingConfig(
                thinking_budget=config.THINKING_BUDGET.get(model, 0)
            ),
        )

    def _real_verdict(self, text_context: str, loaded_images: List[LoadedImage]) -> Tuple[VisionVerdict, Usage]:
        usable = [im for im in loaded_images if im.ok]
        contents = self._build_contents(text_context, usable)
        usage = Usage(images=len(usable), image_tiles=sum(im.estimated_tiles() for im in usable))

        last_err = None
        for model in self.available_models():
            cfg = self._gen_config(model)
            for attempt in range(config.MAX_RETRIES + 1):
                self._limiter.acquire()
                t0 = time.time()
                try:
                    resp = self._client.models.generate_content(model=model, contents=contents, config=cfg)
                    usage.latencies.append(time.time() - t0)
                    usage.calls += 1
                    usage.model_calls[model] = usage.model_calls.get(model, 0) + 1
                    self._record_usage(resp, usage)
                    return self._parse(resp), usage
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if self._is_perday_quota(e):
                        self._mark_exhausted(model)
                        break  # fail over to next model
                    if attempt < config.MAX_RETRIES and self._is_retryable(e):
                        usage.retries += 1
                        self._sleep_backoff(attempt, e)
                        continue
                    break  # non-retryable -> try next model (if any)
        usage.errors += 1
        usage.calls += 1
        return self._fallback_verdict(f"all models unavailable/failed: {last_err}"), usage

    def _parse(self, resp) -> VisionVerdict:
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, VisionVerdict):
            return parsed
        text = (getattr(resp, "text", None) or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("\n", 1)[1] if "\n" in text else text
        try:
            return VisionVerdict(**json.loads(text))
        except Exception:  # noqa: BLE001
            return self._fallback_verdict("unparseable model response")

    @staticmethod
    def _record_usage(resp, usage: Usage) -> None:
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return
        usage.input_tokens += getattr(um, "prompt_token_count", 0) or 0
        usage.output_tokens += getattr(um, "candidates_token_count", 0) or 0
        usage.cached_tokens += getattr(um, "cached_content_token_count", 0) or 0

    @staticmethod
    def _is_perday_quota(e: Exception) -> bool:
        return getattr(e, "code", None) == 429 and bool(_PERDAY_RE.search(str(e)))

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        if getattr(e, "code", None) in _RETRYABLE_CODES:
            return True
        msg = str(e)
        return any(tok in msg for tok in _RETRYABLE_TOKENS)

    @staticmethod
    def _suggested_delay(e: Exception) -> Optional[float]:
        m = _RETRY_DELAY_RE.search(str(e))
        if not m:
            return None
        return float(m.group(1) or m.group(2))

    def _sleep_backoff(self, attempt: int, e: Exception) -> None:
        suggested = self._suggested_delay(e)
        if suggested is not None:
            time.sleep(min(suggested + 1.0, 65.0))
            return
        delay = min(config.RETRY_BASE_DELAY * (2 ** attempt), config.RETRY_MAX_DELAY)
        time.sleep(delay + random.uniform(0, delay * 0.25))

    @staticmethod
    def _fallback_verdict(reason: str) -> VisionVerdict:
        return VisionVerdict(
            claim_summary="",
            issue_type="unknown", object_part="unknown",
            claim_status="not_enough_information", evidence_standard_met=False,
            evidence_standard_met_reason=reason,
            claim_status_justification="Automated evaluation could not produce a confident verdict.",
            supporting_image_ids=[], valid_image=False, severity="unknown",
            visual_risk_flags=[], image_text_instruction_present=False,
        )

    # -- mock call (no API key) ----------------------------------------------
    _ISSUE_KW = {
        "dent": "dent", "scratch": "scratch", "crack": "crack", "shatter": "glass_shatter",
        "broke": "broken_part", "broken": "broken_part", "missing": "missing_part",
        "torn": "torn_packaging", "crush": "crushed_packaging", "water": "water_damage",
        "wet": "water_damage", "stain": "stain", "oil": "stain", "spill": "stain",
    }
    _PART_KW = {
        "front bumper": "front_bumper", "rear bumper": "rear_bumper", "bumper": "rear_bumper",
        "windshield": "windshield", "front glass": "windshield", "headlight": "headlight",
        "taillight": "taillight", "back light": "taillight", "side mirror": "side_mirror",
        "mirror": "side_mirror", "hood": "hood", "fender": "fender", "door": "door",
        "screen": "screen", "display": "screen", "keyboard": "keyboard", "key": "keyboard",
        "trackpad": "trackpad", "hinge": "hinge", "lid": "lid", "corner": "corner",
        "seal": "seal", "label": "label", "contents": "contents", "item": "item",
        "package corner": "package_corner", "box": "box", "body": "body",
    }

    def _mock_verdict(self, text_context: str, loaded_images: List[LoadedImage]) -> Tuple[VisionVerdict, Usage]:
        low = text_context.lower()
        claim_object = "car"
        for obj in ("laptop", "package", "car"):
            if f"claim object: {obj}" in low:
                claim_object = obj
                break
        issue = "unknown"
        for kw, val in self._ISSUE_KW.items():
            if kw in low:
                issue = val
                break
        part = "unknown"
        from .schema import OBJECT_PARTS
        for kw, val in sorted(self._PART_KW.items(), key=lambda x: -len(x[0])):
            if kw in low and val in OBJECT_PARTS.get(claim_object, []):
                part = val
                break
        usable = [im for im in loaded_images if im.ok]
        if usable:
            status, evidence, sev, sids = "supported", True, "medium", [usable[0].image_id]
        else:
            status, evidence, sev, sids = "not_enough_information", False, "unknown", []
        usage = Usage(
            calls=1, images=len(usable),
            image_tiles=sum(im.estimated_tiles() for im in usable),
            input_tokens=len(self._system + text_context) // 4 + sum(im.estimated_tiles() for im in usable) * 258,
            output_tokens=220, latencies=[0.0], model_calls={"mock": 1},
        )
        verdict = VisionVerdict(
            claim_summary="[MOCK] heuristic verdict (no API key)",
            issue_type=issue, object_part=part, claim_status=status,
            evidence_standard_met=evidence,
            evidence_standard_met_reason="[MOCK] keyless dry-run heuristic.",
            claim_status_justification="[MOCK] Generated without a model call; for pipeline testing only.",
            supporting_image_ids=sids, valid_image=bool(usable),
            severity=("unknown" if status == "not_enough_information" else sev),
            visual_risk_flags=[], image_text_instruction_present=False,
        )
        return verdict, usage
