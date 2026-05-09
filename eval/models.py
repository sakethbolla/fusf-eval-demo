"""Provider abstraction over Mock / Groq / Gemini / Anthropic / OpenAI.

Every provider exposes the same .complete(system, user) -> str interface.
Missing API keys are not errors — providers report themselves unavailable
and the runner skips them. The Mock provider is always available so the
pipeline runs end-to-end with no keys configured.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


# ----- Mock provider (always available, deterministic) ----------------------

@dataclass
class MockProvider:
    """Deterministic keyword-driven scorer. Two flavors so the multi-model
    comparison story works without any API keys: 'lenient' gives benefit
    of the doubt, 'strict' penalises ambiguity. They will disagree on the
    borderline LOIs, which is the point."""
    name: str
    model: str
    flavor: str = "lenient"  # 'lenient' or 'strict'

    PRIORITY_DISEASES = [
        "alzheimer", "parkinson", "huntington", "als",
        "glioblastoma", "dipg", "diffuse intrinsic pontine glioma",
        "pancreatic", "breast cancer", "metastatic",
    ]
    PRIORITY_MECHANISMS = [
        "immunomodulation", "immunotherapy", "anti-pd-1", "anti pd-1",
        "neuromodulation", "gene therapy", "drug delivery",
        "sonodynamic", "blood-brain barrier", "bbb",
    ]
    FUS_TERMS = [
        "focused ultrasound", "fus ", "lifu", "hifu", "mr-guided",
        "mr guided", "image-guided ultrasound",
    ]
    DISEASE_TREATMENT_TERMS = [
        "treat", "therap", "tumor", "cancer", "disease", "patient",
        "clinical", "preclinical", "trial",
    ]
    NON_DISEASE_TERMS = ["cosmetic", "wrinkle", "skin tightening", "aesthetic"]

    def complete(self, system: str, user: str) -> str:
        # The user prompt contains both the rubric (which lists every priority
        # disease/mechanism by name) and the LOI. We must score against the LOI
        # only — otherwise every LOI matches every keyword.
        text = self._extract_loi_text(user).lower()

        uses_fus = any(t in text for t in self.FUS_TERMS)
        treats_disease = any(t in text for t in self.DISEASE_TREATMENT_TERMS)
        is_cosmetic = any(t in text for t in self.NON_DISEASE_TERMS)

        if is_cosmetic:
            mission_fit = 1
            mf_rat = "Uses ultrasound for cosmetic, not disease treatment."
        elif uses_fus and treats_disease:
            mission_fit = 5 if self.flavor == "lenient" else 4
            mf_rat = "Uses focused ultrasound to treat disease — central mission match."
        elif uses_fus and not treats_disease:
            mission_fit = 3
            mf_rat = "Uses FUS but disease-treatment intent is unclear."
        elif not uses_fus:
            mission_fit = 1
            mf_rat = "No focused ultrasound mechanism described."
        else:
            mission_fit = 2
            mf_rat = "Marginal mission fit."

        disease_hits = sum(1 for d in self.PRIORITY_DISEASES if d in text)
        mech_hits = sum(1 for m in self.PRIORITY_MECHANISMS if m in text)
        priority_signal = disease_hits + mech_hits

        if priority_signal >= 3:
            strat = 5
            sr_rat = f"Multiple priority intersections (diseases={disease_hits}, mechanisms={mech_hits})."
        elif priority_signal == 2:
            strat = 4
            sr_rat = "Two priority signals."
        elif priority_signal == 1:
            strat = 3 if self.flavor == "lenient" else 2
            sr_rat = "One priority signal — partial alignment."
        else:
            strat = 1
            sr_rat = "No FUSF priority diseases or mechanisms named."

        if not uses_fus:
            strat = min(strat, 2)
            sr_rat += " (Downgraded: no FUS mechanism, so priority match is moot.)"

        weighted = round((mission_fit + strat) / 2, 2)
        below_two = mission_fit < 2 or strat < 2
        decision = "FUND" if (weighted >= 3.5 and not below_two) else "DECLINE"

        if decision == "FUND":
            dec_rat = f"Mission fit {mission_fit}/5 + strategic alignment {strat}/5 clear FUSF profile."
        else:
            dec_rat = f"Weighted {weighted} below 3.5 threshold or critical criterion below 2."

        return json.dumps({
            "criterion_scores": [
                {"id": "mission_fit", "score": mission_fit, "rationale": mf_rat},
                {"id": "strategic_alignment", "score": strat, "rationale": sr_rat},
            ],
            "weighted_score": weighted,
            "decision": decision,
            "decision_rationale": dec_rat,
        })

    @staticmethod
    def _extract_loi_text(user_prompt: str) -> str:
        """Pull just the LOI title + abstract out of the prompt builder's output.
        The prompt has a 'LETTER OF INTENT' section followed by a 'TASK' section."""
        m = re.search(
            r"LETTER OF INTENT\s*(.*?)\s*TASK",
            user_prompt, re.DOTALL,
        )
        return m.group(1) if m else user_prompt


# ----- Real LLM providers ---------------------------------------------------

@dataclass
class GroqProvider:
    name: str = "groq"
    model: str = "llama-3.3-70b-versatile"

    def complete(self, system: str, user: str) -> str:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


@dataclass
class GeminiProvider:
    name: str = "gemini"
    model: str = "gemini-2.5-flash"

    def complete(self, system: str, user: str) -> str:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(
            model=self.model,
            contents=f"{system}\n\n{user}",
            config={"response_mime_type": "application/json", "temperature": 0.0},
        )
        return resp.text or ""


@dataclass
class AnthropicProvider:
    name: str = "anthropic"
    model: str = "claude-sonnet-4-6"

    def complete(self, system: str, user: str) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return _strip_code_fences(text)


@dataclass
class OpenAIProvider:
    name: str = "openai"
    model: str = "gpt-4o-mini"

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


def _strip_code_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` despite being told not to."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ----- Discovery ------------------------------------------------------------

def get_available_providers() -> list[Provider]:
    """Return every provider whose API key is set."""
    providers: list[Provider] = []
    if os.environ.get("GROQ_API_KEY"):
        providers.append(GroqProvider())
    if os.environ.get("GOOGLE_API_KEY"):
        providers.append(GeminiProvider())
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(AnthropicProvider())
    if os.environ.get("OPENAI_API_KEY"):
        providers.append(OpenAIProvider())
    return providers
