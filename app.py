"""Streamlit UI for the FUSF LOI eval demo."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    """Map model_id -> list of eval-result dicts."""
    out: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == ".gitkeep":
            continue
        results = json.loads(path.read_text())
        if not results:
            continue
        model_id = f"{results[0]['provider']} / {results[0]['model']}"
        out[model_id] = results
    return out


def index_results(results: dict[str, list[dict]]) -> dict[tuple[str, str], dict]:
    """(model_id, loi_id) -> single eval-result dict."""
    return {(m, r["loi_id"]): r for m, rs in results.items() for r in rs}


# ----- Page setup -----------------------------------------------------------

st.set_page_config(page_title="FUSF LOI Eval", page_icon="🧠", layout="wide")
st.title("FUSF Letter of Intent — AI Evaluation Demo")
st.caption(
    "Multi-model evaluation of research LOIs against the Focused Ultrasound "
    "Foundation's published Stage 1 criteria, with ground-truth comparison."
)

rubric = load_rubric()
lois = load_lois()
results_by_model = load_results()
results_index = index_results(results_by_model)

if not results_by_model:
    st.warning(
        "No evaluation results found. Run `python run_evals.py` from the project "
        "root first — it will execute every LOI through every available model "
        "(Mock providers always run, real providers run when their API keys are set)."
    )
    st.stop()


# ----- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.subheader("Models in this run")
    for m in results_by_model:
        st.markdown(f"- {m}")
    st.markdown("---")
    st.markdown(f"**Mission**\n\n{rubric['mission']}")
    with st.expander("Strategic priorities"):
        st.json(rubric["strategic_priorities"])


# ----- Tabs -----------------------------------------------------------------

tab_loi, tab_metrics, tab_failures, tab_next = st.tabs(
    ["LOI Viewer", "Agreement Metrics", "Failure Modes", "What's Next"]
)


# ===== Tab 1: LOI Viewer ====================================================

with tab_loi:
    col_pick, col_tag = st.columns([3, 1])
    with col_pick:
        loi_options = {f"{l['id']} — {l['title'][:80]}": l for l in lois}
        choice = st.selectbox("Pick an LOI", list(loi_options.keys()))
    loi = loi_options[choice]
    truth = loi["ground_truth"]
    with col_tag:
        color = "🟢" if truth["decision"] == "FUND" else "🔴"
        st.metric("Ground Truth", f"{color} {truth['decision']}")

    st.markdown(f"**PI:** {loi.get('pi', '—')}")
    with st.expander("Abstract", expanded=False):
        st.write(loi["abstract"])
    st.info(f"**Why this label:** {truth['rationale']}")

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


# ===== Tab 2: Agreement Metrics =============================================

with tab_metrics:
    st.markdown("### Per-model agreement with ground truth")
    rows = []
    for model_id, model_results in results_by_model.items():
        truth_map = {l["id"]: l["ground_truth"]["decision"] for l in lois}
        n = len(model_results)
        agree = sum(1 for r in model_results if r["decision"] == truth_map[r["loi_id"]])
        tp = sum(1 for r in model_results if r["decision"] == "FUND" and truth_map[r["loi_id"]] == "FUND")
        fp = sum(1 for r in model_results if r["decision"] == "FUND" and truth_map[r["loi_id"]] == "DECLINE")
        tn = sum(1 for r in model_results if r["decision"] == "DECLINE" and truth_map[r["loi_id"]] == "DECLINE")
        fn = sum(1 for r in model_results if r["decision"] == "DECLINE" and truth_map[r["loi_id"]] == "FUND")
        rows.append({
            "Model": model_id,
            "Agreement": f"{agree}/{n} ({agree/n:.0%})",
            "TP (correct FUND)": tp,
            "FP (overshoot)": fp,
            "TN (correct DECLINE)": tn,
            "FN (missed fund)": fn,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### Confusion matrices")
    grid_cols = st.columns(min(len(results_by_model), 3))
    for i, (model_id, model_results) in enumerate(results_by_model.items()):
        truth_map = {l["id"]: l["ground_truth"]["decision"] for l in lois}
        cm = [[0, 0], [0, 0]]  # rows: truth FUND/DECLINE; cols: pred FUND/DECLINE
        for r in model_results:
            t_idx = 0 if truth_map[r["loi_id"]] == "FUND" else 1
            p_idx = 0 if r["decision"] == "FUND" else 1
            cm[t_idx][p_idx] += 1
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["pred FUND", "pred DECLINE"],
            y=["true FUND", "true DECLINE"],
            text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(title=model_id, height=280, margin=dict(l=20, r=20, t=40, b=20))
        with grid_cols[i % len(grid_cols)]:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Inter-model agreement")
    st.caption("How often do the models agree with *each other* on the FUND/DECLINE call? "
               "Disagreement on the same LOI is itself a signal of LOI ambiguity.")
    model_ids = list(results_by_model.keys())
    matrix = []
    for a in model_ids:
        row = []
        for b in model_ids:
            decisions_a = {r["loi_id"]: r["decision"] for r in results_by_model[a]}
            decisions_b = {r["loi_id"]: r["decision"] for r in results_by_model[b]}
            common = set(decisions_a) & set(decisions_b)
            agree = sum(1 for k in common if decisions_a[k] == decisions_b[k])
            row.append(agree / len(common) if common else 0)
        matrix.append(row)
    fig = px.imshow(
        matrix, x=model_ids, y=model_ids, text_auto=".0%",
        color_continuous_scale="Greens", zmin=0, zmax=1,
        labels=dict(color="Agreement"),
    )
    fig.update_layout(height=400, margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ===== Tab 3: Failure Modes =================================================

with tab_failures:
    st.markdown("### Where do models disagree with ground truth?")
    st.caption("Each row is one (model, LOI) pair where the model got it wrong. "
               "The taxonomy column groups failures by what the LOI was *designed* "
               "to test — anchoring, prestige bias, form-vs-fit, etc.")

    failure_taxonomy = {
        "LOI-004": "Surface-FUS anchoring (cosmetic, not disease)",
        "LOI-005": "Topic-anchoring + prestige bias (priority disease, no FUS)",
        "LOI-006": "Form vs fit (correct topic, no research content)",
        "LOI-007": "Off-priority indication (legit FUS, wrong disease area)",
        "LOI-008": "Keyword over mission (priority disease, no FUS)",
    }

    rows = []
    for model_id, model_results in results_by_model.items():
        truth_map = {l["id"]: l for l in lois}
        for r in model_results:
            truth = truth_map[r["loi_id"]]["ground_truth"]
            if r["decision"] != truth["decision"]:
                rows.append({
                    "Model": model_id,
                    "LOI": r["loi_id"],
                    "Title": truth_map[r["loi_id"]]["title"][:60] + "…",
                    "Failure mode": failure_taxonomy.get(r["loi_id"], "Other"),
                    "Truth": truth["decision"],
                    "Predicted": r["decision"],
                    "Model's rationale": r["decision_rationale"][:120] + "…",
                })

    if not rows:
        st.success("No disagreements — every model matched ground truth on every LOI.")
    else:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### Failure-mode taxonomy")
    st.caption("These are the kinds of mistakes we *expect* from LLM evaluators on grant LOIs. "
               "The synthetic decline LOIs were designed to surface each pattern.")
    for loi_id, mode in failure_taxonomy.items():
        loi = next((l for l in lois if l["id"] == loi_id), None)
        if not loi:
            continue
        with st.expander(f"{loi_id} · {mode}"):
            st.markdown(f"**Title:** {loi['title']}")
            st.markdown(f"**Why it's a trap:** {loi['ground_truth']['rationale']}")


# ===== Tab 4: What's Next ===================================================

with tab_next:
    st.markdown("""
    ### Honest limits of this demo

    - **Tiny dataset.** 8 LOIs, 3 funded + 5 declined, half synthetic. Real evaluation
      needs hundreds of historical LOIs with reviewer notes. Statistical confidence
      from this dataset is low — call directional, not conclusive.
    - **No reviewer disagreement modeling.** Real FUSF reviewers disagree with each
      other. Ground truth here is binary; in practice it's a distribution.
    - **Stage 1 only.** Full proposals add scientific merit / feasibility / team /
      eligibility. Those need the actual proposal, not just the LOI.
    - **Mock providers are deterministic keyword scorers**, included so the pipeline
      runs end-to-end without API keys. They are *not* a stand-in for real LLM behavior.

    ### What I'd build next, with real data and a few weeks

    1. **Replay set.** Pull every LOI from the last ~3 years with the staff decision
       and any reviewer comments. De-identify.
    2. **Calibration study.** Run 3–4 frontier models on the replay set. Report
       agreement, disagreement clusters, per-criterion calibration. Surface the LOIs
       where models *and* humans split — those are the highest-value to study.
    3. **Reviewer-augmentation prototype.** Not "AI decides" — "AI drafts a Stage 1
       memo with citations to the rubric, reviewer edits in 30 seconds." Measure
       reviewer time saved and edit distance from AI draft to final memo.
    4. **Continuous evaluation.** New LOI evaluations write back to the replay set.
       Drift detection: are model outputs changing over time as base models update?
    5. **Failure-mode regression suite.** Lock in the synthetic LOIs from this demo
       as a permanent test set. Any new model has to pass them before it ships.

    ### What this demo is meant to communicate

    - I think about LLM use as **measurement first, deployment second**.
    - I expect models to fail in *kinds*, not just rates — and I structure datasets
      around those kinds.
    - I treat synthetic data as a starting line for surfacing failure modes,
      never as the finish line for claiming performance.
    """)
