"""Prompt construction: rubric + LOI -> structured eval request."""
import json
from textwrap import dedent


SYSTEM_PROMPT = dedent("""
    You are a grant pre-screening assistant for the Focused Ultrasound Foundation (FUSF).
    Your job is to evaluate a Letter of Intent (LOI) against FUSF's published Stage 1
    criteria and produce a strict JSON response. Do not invent priorities not in the rubric.
    Be calibrated: a project that uses 'focused ultrasound' but does not actually treat
    disease should score low on mission_fit; a project on a priority disease that does
    not use FUS should also score low. You will be evaluated on agreement with FUSF's
    real funding decisions, so reason carefully and be willing to decline.
""").strip()


def build_user_prompt(rubric: dict, loi: dict) -> str:
    """Build the user-facing prompt with rubric + LOI + strict output schema."""
    stage1 = next(s for s in rubric["stages"] if s["stage"] == 1)
    criteria_block = "\n".join(
        f"- {c['id']} ({c['name']}, weight {c['weight']}): {c['description']}\n  Scale: {c['scale']}"
        for c in stage1["criteria"]
    )
    priorities = rubric["strategic_priorities"]
    priorities_block = json.dumps(priorities, indent=2)

    schema_example = {
        "criterion_scores": [
            {"id": "mission_fit", "score": 4, "rationale": "..."},
            {"id": "strategic_alignment", "score": 5, "rationale": "..."},
        ],
        "weighted_score": 4.5,
        "decision": "FUND",
        "decision_rationale": "...",
    }

    return dedent(f"""
        FOUNDATION MISSION
        {rubric["mission"]}

        STRATEGIC PRIORITIES
        {priorities_block}

        STAGE 1 (LOI) CRITERIA
        {criteria_block}

        DECISION RULE
        FUND if weighted Stage 1 score >= 3.5 AND no individual criterion below 2; otherwise DECLINE.

        LETTER OF INTENT
        ID: {loi["id"]}
        Title: {loi["title"]}
        PI: {loi.get("pi", "")}
        Abstract:
        {loi["abstract"]}

        TASK
        Score the LOI on each Stage 1 criterion (1-5) with a one-sentence rationale.
        Compute the weighted score. Apply the decision rule. Return ONLY valid JSON
        matching this exact shape (no markdown fences, no commentary):

        {json.dumps(schema_example, indent=2)}
    """).strip()
