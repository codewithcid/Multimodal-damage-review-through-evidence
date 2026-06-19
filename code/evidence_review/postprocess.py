"""Deterministic guardrails applied on top of the model's vision verdict.

Responsibilities:
  * snap every field onto its allowed enum,
  * derive user_history_risk / manual_review_required from history (not the model),
  * detect prompt-injection text in the claim conversation as a backstop,
  * enforce the cross-field consistency the gold sample always exhibits,
  * assemble the final 14-column output row.
"""
from __future__ import annotations

import re
from typing import Dict, List

from . import schema
from .data import Claim, UserHistory
from .images import LoadedImage
from .schema import VisionVerdict

# Phrases that indicate an attempt to instruct/override the reviewer. Detected in
# the claim conversation as a backstop to the model's image-text detection.
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"\bapprove\w*\b.{0,20}\b(claim|row|it|this|immediately|quickly|now)\b",
    r"\b(accept|approve)\w*\s+(this|the|it)\b",
    r"mark (this|the)?\s*(row|claim)?\s*(as )?supported",
    r"skip (the )?manual review",
    r"skip review",
    r"follow (the |this )?(note|instruction)",
    r"should be approved",
    r"approve the claim",
    r"\bfollow it\b",
    r"reopen\w*\s+tickets?",
    r"escalate\w*\s+(publicly|this)",
    r"until (someone|you)\s+approv\w*",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Visual flags that, on their own, warrant a human look.
_SERIOUS_VISUAL_FLAGS = {
    "wrong_object",
    "claim_mismatch",
    "non_original_image",
    "possible_manipulation",
    "text_instruction_present",
}


def claim_has_injection(user_claim: str) -> bool:
    return bool(_INJECTION_RE.search(user_claim or ""))


def _order_flags(flags: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for f in schema.RISK_FLAG_ORDER:
        if f in flags and f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered


def finalize(
    claim: Claim,
    verdict: VisionVerdict,
    history: UserHistory | None,
    loaded_images: List[LoadedImage],
) -> Dict[str, str]:
    """Apply guardrails and return the final output row as a dict of strings."""
    valid_image_ids = {im.image_id for im in loaded_images if im.ok}
    all_image_ids = {im.image_id for im in loaded_images}

    # --- enum coercion -------------------------------------------------------
    claim_status = schema.coerce_enum(verdict.claim_status, schema.CLAIM_STATUS, "not_enough_information")
    issue_type = schema.coerce_enum(verdict.issue_type, schema.ISSUE_TYPES, "unknown")
    object_part = schema.coerce_object_part(verdict.object_part, claim.claim_object)
    severity = schema.coerce_enum(verdict.severity, schema.SEVERITY, "unknown")
    valid_image = bool(verdict.valid_image)

    # --- visual risk flags from the model (validated subset) -----------------
    flags = {schema.coerce_enum(f, schema.VISUAL_RISK_FLAGS, "") for f in (verdict.visual_risk_flags or [])}
    flags.discard("")

    # --- prompt-injection backstop on the claim text -------------------------
    if claim_has_injection(claim.user_claim) or verdict.image_text_instruction_present:
        flags.add("text_instruction_present")

    # --- history-derived flags (added by the system, never the model) --------
    hist_flags = history.flag_list if history else []
    if "user_history_risk" in hist_flags:
        flags.add("user_history_risk")

    manual_review = (
        "user_history_risk" in hist_flags
        or "manual_review_required" in hist_flags
        or bool(flags & _SERIOUS_VISUAL_FLAGS)
        or not valid_image
    )
    if manual_review:
        flags.add("manual_review_required")

    # An image set deemed unusable/non-original cannot CONFIRM a claim. The gold
    # never pairs valid_image=false with claim_status=supported. If the model did,
    # downgrade to not_enough_information (the evidence is not trustworthy).
    drop_support = False
    if not valid_image and claim_status == "supported":
        claim_status = "not_enough_information"
        drop_support = True

    # --- cross-field consistency (matches every gold sample row) -------------
    evidence_met = claim_status != "not_enough_information"
    if claim_status == "not_enough_information":
        severity = "unknown"

    # supporting_image_ids: keep only real, readable IDs the model cited
    supporting = [] if drop_support else [i for i in (verdict.supporting_image_ids or []) if i in valid_image_ids]
    # de-dupe while preserving order
    supporting = list(dict.fromkeys(supporting))
    supporting_str = ";".join(supporting) if supporting else "none"

    risk_flags_list = _order_flags(list(flags))
    risk_flags_str = ";".join(risk_flags_list) if risk_flags_list else "none"

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "true" if evidence_met else "false",
        "evidence_standard_met_reason": verdict.evidence_standard_met_reason.strip(),
        "risk_flags": risk_flags_str,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": verdict.claim_status_justification.strip(),
        "supporting_image_ids": supporting_str,
        "valid_image": "true" if valid_image else "false",
        "severity": severity,
    }


def unreadable_row(claim: Claim, history: UserHistory | None) -> Dict[str, str]:
    """Fallback row when every image for a claim is missing/unreadable."""
    flags = {"cropped_or_obstructed", "manual_review_required"}
    if claim_has_injection(claim.user_claim):
        flags.add("text_instruction_present")
    hist_flags = history.flag_list if history else []
    if "user_history_risk" in hist_flags:
        flags.add("user_history_risk")
    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "No usable image could be loaded for this claim.",
        "risk_flags": ";".join(_order_flags(list(flags))),
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "The submitted images could not be read, so the claim cannot be evaluated.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }
