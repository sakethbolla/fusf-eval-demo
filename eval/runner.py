"""Run one LOI through one provider, return a parsed EvalResult."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, asdict
from typing import Any

from .models import Provider
from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass
class CriterionScore:
    id: str
    score: int
    rationale: str


@dataclass
class EvalResult:
    loi_id: str
    provider: str
    model: str
    criterion_scores: list[CriterionScore]
    weighted_score: float
    decision: str  # "FUND" | "DECLINE"
    decision_rationale: str
    raw_response: str
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def evaluate_loi(loi: dict, rubric: dict, provider: Provider) -> EvalResult:
    system = SYSTEM_PROMPT
    user = build_user_prompt(rubric, loi)

    try:
        raw = provider.complete(system, user)
    except Exception as e:
        return EvalResult(
            loi_id=loi["id"], provider=provider.name, model=provider.model,
            criterion_scores=[], weighted_score=0.0,
            decision="DECLINE", decision_rationale=f"Provider error: {e}",
            raw_response="", parse_error=str(e),
        )

    parsed, err = _parse_json(raw)
    if err or parsed is None:
        return EvalResult(
            loi_id=loi["id"], provider=provider.name, model=provider.model,
            criterion_scores=[], weighted_score=0.0,
            decision="DECLINE", decision_rationale="Unparseable model output",
            raw_response=raw, parse_error=err,
        )

    try:
        criterion_scores = [CriterionScore(**c) for c in parsed["criterion_scores"]]
        return EvalResult(
            loi_id=loi["id"], provider=provider.name, model=provider.model,
            criterion_scores=criterion_scores,
            weighted_score=float(parsed["weighted_score"]),
            decision=parsed["decision"].upper(),
            decision_rationale=parsed["decision_rationale"],
            raw_response=raw,
        )
    except (KeyError, TypeError, ValueError) as e:
        return EvalResult(
            loi_id=loi["id"], provider=provider.name, model=provider.model,
            criterion_scores=[], weighted_score=0.0,
            decision="DECLINE", decision_rationale=f"Schema mismatch: {e}",
            raw_response=raw, parse_error=str(e),
        )


def _parse_json(text: str) -> tuple[dict | None, str | None]:
    """Try strict JSON; fall back to extracting the first {...} block."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no JSON object found in response"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, str(e)
