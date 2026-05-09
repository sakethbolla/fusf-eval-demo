"""Streamlit UI for the FUSF LOI eval demo."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


# ----- Data loading ---------------------------------------------------------

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


def index_results(results: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    return {(m, r["loi_id"]): r for m, rs in results.items() for r in rs}


# ----- Page setup -----------------------------------------------------------

st.set_page_config(page_title="FUSF LOI Eval", page_icon="🧠", layout="wide")
st.title("FUSF Letter of Intent — AI Evaluation Demo")
st.caption(
    "Two contrasting research LOIs, evaluated by two LLMs against the Focused "
    "Ultrasound Foundation's published Stage 1 criteria."
)

rubric = load_rubric()
lois = load_lois()
results_by_model = load_results()
results_index = index_results(results_by_model)

if not results_by_model:
    st.warning(
        "No evaluation results found. Run `python run_evals.py` from the project "
        "root first."
    )
    st.stop()


# ----- Headline finding banner ---------------------------------------------

# Compute agreement per model (used in banner + headline tab)
truth_map = {l["id"]: l["historical_decision"]["decision"] for l in lois}
agreement_rows = []
disagreement_pairs: list[tuple[str, str]] = []  # (model_id, loi_id)
for model_id, rs in results_by_model.items():
    n = len(rs)
    correct = sum(1 for r in rs if r["decision"] == truth_map[r["loi_id"]])
    agreement_rows.append({"Model": model_id, "Correct": f"{correct} / {n}", "Accuracy": f"{correct/n:.0%}"})
    for r in rs:
        if r["decision"] != truth_map[r["loi_id"]]:
            disagreement_pairs.append((model_id, r["loi_id"]))

st.info(
    "**Headline finding.** Both LOIs target oncology + immunotherapy. "
    "**LOI-001** uses MR-guided focused ultrasound to open the blood-brain barrier "
    "for anti-PD-1 delivery — central FUSF mission. **LOI-008** proposes lipid "
    "nanoparticle mRNA vaccines for breast cancer — same therapeutic theme, "
    "but **no focused ultrasound**. "
    f"On this 2-LOI contrast set, **groq/llama-3.3-70b** got both right (2/2). "
    f"**openai/gpt-4o-mini** funded LOI-008 anyway (1/2) — anchoring on the "
    f"keywords 'breast cancer' and 'immunotherapy' and missing that FUSF's "
    f"mission requires focused ultrasound be used."
)


# ----- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.subheader("Models compared")
    for m in results_by_model:
        st.markdown(f"- {m}")
    st.markdown("---")
    st.markdown(f"**FUSF mission**\n\n{rubric['mission']}")
    with st.expander("Strategic priorities"):
        st.json(rubric["strategic_priorities"])


# ----- Tabs -----------------------------------------------------------------

tab_loi, tab_finding, tab_next = st.tabs(
    ["LOI Viewer", "Headline Finding", "What's Next"]
)


# ===== Tab 1: LOI Viewer ====================================================

with tab_loi:
    col_pick, col_tag = st.columns([3, 1])
    with col_pick:
        loi_options = {f"{l['id']} — {l['title'][:80]}": l for l in lois}
        choice = st.selectbox("Pick an LOI", list(loi_options.keys()))
    loi = loi_options[choice]
    truth = loi["historical_decision"]
    with col_tag:
        color = "🟢" if truth["decision"] == "FUND" else "🔴"
        st.metric("Historical Decision", f"{color} {truth['decision']}")

    st.markdown(f"**PI:** {loi.get('pi', '—')}")
    with st.expander("Abstract", expanded=False):
        st.write(loi["abstract"])
    st.info(f"**Why FUSF decided this:** {truth['rationale']}")

    st.markdown("### Model evaluations")
    cols = st.columns(len(results_by_model))
    for col, (model_id, _) in zip(cols, results_by_model.items()):
        with col:
            r = results_index.get((model_id, loi["id"]))
            if not r:
                st.error(f"{model_id}: no result")
                continue
            agree = r["decision"] == truth["decision"]
            badge = "✅" if agree else "❌"
            st.markdown(f"**{model_id}** {badge}")
            color = "🟢" if r["decision"] == "FUND" else "🔴"
            st.markdown(f"{color} **{r['decision']}** · weighted {r['weighted_score']}")
            for c in r["criterion_scores"]:
                st.markdown(f"- *{c['id']}*: **{c['score']}/5** — {c['rationale']}")
            with st.expander("Decision rationale"):
                st.write(r["decision_rationale"])
            if r.get("parse_error"):
                st.warning(f"Parse error: {r['parse_error']}")


# ===== Tab 2: Headline Finding ==============================================

with tab_finding:
    st.markdown("### Per-model agreement with historical decisions")
    st.dataframe(pd.DataFrame(agreement_rows), hide_index=True, use_container_width=True)

    st.markdown("### The single disagreement")
    if not disagreement_pairs:
        st.success("No disagreements — every model matched the historical decision on every LOI.")
    else:
        for model_id, loi_id in disagreement_pairs:
            loi_obj = next(l for l in lois if l["id"] == loi_id)
            r = results_index[(model_id, loi_id)]
            st.markdown(
                f"**{model_id}** funded **{loi_id}** ({loi_obj['title']}) — "
                f"FUSF historically declined this."
            )
            st.markdown(f"> *Model's rationale:* {r['decision_rationale']}")
            st.markdown(f"> *Why FUSF declined:* {loi_obj['historical_decision']['rationale']}")

    st.markdown("### Failure mode: keyword-over-mission")
    st.markdown(
        "LOI-008 is a deliberate trap. The text contains every signal a "
        "keyword-anchored model uses to detect FUSF fit — **breast cancer**, "
        "**metastatic**, **immunotherapy**, **HER2-targeted**, even **mRNA vaccine**. "
        "But the entire project uses lipid nanoparticles for delivery, not "
        "focused ultrasound. Reading the abstract, a human reviewer sees this "
        "in seconds. A smaller language model anchored on the disease + "
        "mechanism keywords and missed the structural absence of FUS.\n\n"
        "**Why this matters for AI evaluation work at FUSF:** if the foundation "
        "ever uses LLMs in any reviewer-augmenting capacity, this is exactly "
        "the failure mode that has to be detected and bounded *before* the model "
        "is trusted with real LOIs. That's what this demo's evaluation harness is "
        "built to do."
    )


# ===== Tab 3: What's Next ===================================================

with tab_next:
    st.markdown("""
    ### Honest limits of this demo

    - **Tiny dataset.** 2 LOIs, deliberately chosen to contrast on one specific
      failure mode. Real evaluation needs hundreds of historical LOIs with
      reviewer notes. Statistical confidence here is zero — call directional only.
    - **Stage 1 only.** Full proposals add scientific merit / feasibility / team /
      eligibility. Those need the actual proposal, not just the LOI.
    - **One synthetic LOI.** LOI-008 was constructed to surface a specific
      failure pattern. Real LOIs are messier.

    ### What I'd build next, with real data and a few weeks

    1. **Replay set.** Pull every LOI from the last ~3 years with the staff
       decision and any reviewer comments. De-identify.
    2. **Calibration study.** Run 3–4 frontier models on the replay set. Report
       agreement, disagreement clusters, per-criterion calibration. Surface the
       LOIs where models *and* humans split — those are the highest-value to study.
    3. **Reviewer-augmentation prototype.** Not "AI decides" — "AI drafts a
       Stage 1 memo with citations to the rubric, reviewer edits in 30 seconds."
       Measure reviewer time saved and edit distance from AI draft to final memo.
    4. **Failure-mode regression suite.** Lock in synthetic LOIs like LOI-008 as
       a permanent test set. Any new model has to pass them before it ships.
    5. **Continuous evaluation.** New evaluations write back to the replay set.
       Drift detection: are model outputs changing over time as base models update?

    ### What this demo is meant to communicate

    - I think about LLM use as **measurement first, deployment second**.
    - I expect models to fail in *kinds*, not just rates — and I structure
      datasets around those kinds.
    - I treat synthetic data as a starting line for surfacing failure modes,
      never as the finish line for claiming performance.
    """)
