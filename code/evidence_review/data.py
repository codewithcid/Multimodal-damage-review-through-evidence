"""Data loading: claims, user history, and evidence requirements.

Reads the provided CSVs and exposes simple lookups by user_id and claim_object.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import config


@dataclass
class Claim:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    case_id: str = ""  # derived from the first image path, e.g. case_001
    # expected output columns (present only for sample_claims.csv)
    expected: Dict[str, str] = field(default_factory=dict)

    @property
    def image_path_list(self) -> List[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def flag_list(self) -> List[str]:
        if not self.history_flags or self.history_flags.strip().lower() == "none":
            return []
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


def _derive_case_id(image_paths: str) -> str:
    for p in image_paths.split(";"):
        parts = Path(p.strip()).parts
        for part in parts:
            if part.startswith("case_"):
                return part
    return ""


def load_claims(path: Path) -> List[Claim]:
    """Load claims.csv (input only) or sample_claims.csv (input + expected)."""
    claims: List[Claim] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_cols = {"user_id", "image_paths", "user_claim", "claim_object"}
        for row in reader:
            expected = {k: v for k, v in row.items() if k not in input_cols}
            claims.append(
                Claim(
                    user_id=row["user_id"],
                    image_paths=row["image_paths"],
                    user_claim=row["user_claim"],
                    claim_object=row["claim_object"].strip().lower(),
                    case_id=_derive_case_id(row["image_paths"]),
                    expected={k: v for k, v in expected.items() if v is not None},
                )
            )
    return claims


def load_user_history(path: Path = config.USER_HISTORY_CSV) -> Dict[str, UserHistory]:
    history: Dict[str, UserHistory] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def _int(x: str) -> int:
                try:
                    return int(str(x).strip())
                except (ValueError, TypeError):
                    return 0

            history[row["user_id"]] = UserHistory(
                user_id=row["user_id"],
                past_claim_count=_int(row.get("past_claim_count")),
                accept_claim=_int(row.get("accept_claim")),
                manual_review_claim=_int(row.get("manual_review_claim")),
                rejected_claim=_int(row.get("rejected_claim")),
                last_90_days_claim_count=_int(row.get("last_90_days_claim_count")),
                history_flags=row.get("history_flags", "none") or "none",
                history_summary=row.get("history_summary", "") or "",
            )
    return history


def load_evidence_requirements(
    path: Path = config.EVIDENCE_REQUIREMENTS_CSV,
) -> List[EvidenceRequirement]:
    reqs: List[EvidenceRequirement] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reqs.append(
                EvidenceRequirement(
                    requirement_id=row["requirement_id"],
                    claim_object=row["claim_object"].strip().lower(),
                    applies_to=row["applies_to"],
                    minimum_image_evidence=row["minimum_image_evidence"],
                )
            )
    return reqs


def requirements_for_object(
    reqs: List[EvidenceRequirement], claim_object: str
) -> List[EvidenceRequirement]:
    """Requirements that apply to a given object (object-specific + 'all')."""
    return [r for r in reqs if r.claim_object in (claim_object, "all")]


def get_history(history: Dict[str, UserHistory], user_id: str) -> Optional[UserHistory]:
    return history.get(user_id)
