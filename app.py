"""Streamlit UI — single-page narrative of the FUSF LOI eval demo."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


@st.cache_data
def load_rubric() -> dict:
    return json.loads((DATA / "rubric.json").read_text())


@st.cache_data
def load_lois() -> list[dict]:
    return json.loads((DATA / "lois.json").read_text())


@st.cache_data
def load_results() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        results = json.loads(path.read_text())
        if not results:
            continue
        model_id = f"{results[0]['provider']} / {results[0]['model']}"
        out[model_id] = results
    return out


# ----- Setup ---------------------------------------------------------------

st.set_page_config(page_title="FUSF LOI Eval", page_icon="🧠", layout="centered")

rubric = load_rubric()
lois = load_lois()
results_by_model = load_results()
results_index = {(m, r["loi_id"]): r for m, rs in results_by_model.items() for r in rs}

if not results_by_model:
    st.warning("No evaluation results. Run `python run_evals.py` first.")
    st.stop()


# ----- Top of page ---------------------------------------------------------

st.title("FUSF Letter of Intent — AI Evaluation Demo")

st.markdown(
    "Two grant Letters of Intent (LOIs) are given to two LLMs. "
    "Each LLM independently decides **FUND** or **DECLINE**. "
    "Then we check each LLM's decision against the **ground truth** — "
    "the right answer for that LOI based on FUSF's published criteria."
)


# ----- FUSF acceptance criteria -------------------------------------------

st.divider()
st.subheader("What FUSF actually funds")

st.markdown(
    """
- **Mission:** the project must use **non-invasive image-guided focused ultrasound** to treat disease
- **Priority diseases:**
    - Neurodegenerative — Alzheimer's, Parkinson's, Huntington's, ALS
    - Oncology — glioblastoma, DIPG, pancreatic, breast, metastatic cancer (immunotherapy emphasis)
    - Companion animal applications also qualify
- **Priority mechanisms:** immunomodulation · neuromodulation · gene therapy · drug delivery · sonodynamic therapy
- **How LOIs are scored:** each model gives a 1–5 score on **mission fit** and **strategic alignment**
- **Decision rule:** weighted score ≥ 3.5 **AND** no single score below 2 → **FUND**, otherwise **DECLINE**
"""
)


# ----- Step 1: pick an LOI -------------------------------------------------

st.divider()
st.subheader("1. The LOI")
loi_options = {f"{l['id']} — {l['title']}": l for l in lois}
choice = st.selectbox("Pick an LOI", list(loi_options.keys()))
loi = loi_options[choice]

st.markdown(f"**Title:** {loi['title']}")
st.markdown(f"**PI:** {loi.get('pi', '—')}")
st.markdown(f"**Abstract:** {loi['abstract']}")


# ----- Step 2: ground truth -----------------------------------------------

truth = loi["ground_truth"]
st.divider()
st.subheader("2. Ground truth — the right answer")
color = "🟢" if truth["decision"] == "FUND" else "🔴"
st.markdown(f"### {color} {truth['decision']}")
st.markdown(f"*Why:* {truth['rationale']}")


# ----- Step 3: model outputs ----------------------------------------------

st.divider()
st.subheader("3. What each LLM said")

table_rows: dict[str, dict[str, str]] = {
    "mission_fit (score)": {},
    "mission_fit (model's reasoning)": {},
    "strategic_alignment (score)": {},
    "strategic_alignment (model's reasoning)": {},
    "Weighted score": {},
    "Decision": {},
    "Matches ground truth?": {},
}
for model_id in results_by_model:
    r = results_index.get((model_id, loi["id"]))
    if not r:
        for k in table_rows:
            table_rows[k][model_id] = "—"
        continue
    scores = {c["id"]: c for c in r["criterion_scores"]}
    mf = scores.get("mission_fit", {"score": "?", "rationale": "—"})
    sa = scores.get("strategic_alignment", {"score": "?", "rationale": "—"})
    table_rows["mission_fit (score)"][model_id] = f"{mf['score']}/5"
    table_rows["mission_fit (model's reasoning)"][model_id] = mf["rationale"]
    table_rows["strategic_alignment (score)"][model_id] = f"{sa['score']}/5"
    table_rows["strategic_alignment (model's reasoning)"][model_id] = sa["rationale"]
    table_rows["Weighted score"][model_id] = str(r["weighted_score"])
    table_rows["Decision"][model_id] = r["decision"]
    table_rows["Matches ground truth?"][model_id] = "✅ yes" if r["decision"] == truth["decision"] else "❌ no"

df = pd.DataFrame(table_rows).T
df.index.name = ""
st.dataframe(df, use_container_width=True)


# ----- Step 4: disagreement (only if there is one) ------------------------

disagreements = [
    (m, results_index[(m, loi["id"])])
    for m in results_by_model
    if (m, loi["id"]) in results_index
    and results_index[(m, loi["id"])]["decision"] != truth["decision"]
]

if disagreements:
    st.divider()
    st.subheader("4. Where it went wrong")
    for model_id, r in disagreements:
        st.error(
            f"**{model_id}** said **{r['decision']}**, but the ground truth is "
            f"**{truth['decision']}**."
        )
        st.markdown(f"**The model's own reasoning:**")
        st.markdown(f"> {r['decision_rationale']}")
        if loi["id"] == "LOI-008":
            st.markdown(
                "**Why this is the smoking gun.** The LOI explicitly says the "
                "delivery mechanism is **lipid nanoparticles** — there is no "
                "focused ultrasound in the project at all. But the model's "
                "`mission_fit` reasoning above claims the project "
                "*\"uses focused ultrasound as a delivery mechanism.\"* "
                "**The model hallucinated focused ultrasound into the project** "
                "to justify its FUND decision. This is exactly the kind of "
                "failure mode an evaluation harness is built to catch."
            )
else:
    st.divider()
    st.success("Both models agreed with the ground truth on this LOI.")


# ----- Limitations ---------------------------------------------------------

st.divider()
st.subheader("Limitations of this demo")
st.markdown(
    """
- **Tiny dataset** — only 2 LOIs. Findings are directional, not statistical
- **One LOI is synthetic** — LOI-008 was deliberately constructed to test a specific failure mode (priority keywords without focused ultrasound)
- **Ground truth labels are mine** — written from FUSF's published criteria, not pulled from FUSF's actual past decisions. A production system would replay real historical reviews
- **Stage 1 only** — the full proposal stage adds scientific merit, feasibility, team, and eligibility criteria, which need the actual proposal text
- **Binary ground truth** — real reviewers disagree with each other; truth is a distribution, not a single label
- **Two models only** — Llama 3.3 70B and GPT-4o-mini. A real evaluation would compare more, including frontier models like Claude and GPT-4o
"""
)


# ----- Footer note ---------------------------------------------------------

st.divider()
st.caption(
    f"FUSF mission: *{rubric['mission']}* · "
    f"Models compared: {', '.join(results_by_model.keys())}"
)
