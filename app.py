"""Streamlit UI — FUSF LOI eval, organized around a 5-part evaluation framework."""
from __future__ import annotations
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


# ----- Secrets bridge -------------------------------------------------------
for _k in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
    except (FileNotFoundError, KeyError):
        pass


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


# ----- Specificity check ---------------------------------------------------
# Cheap proxy for "is the model citing things actually in the LOI?"
# Tokenize content words from the model's rationale and the LOI text;
# report % overlap. Low overlap = generic filler or hallucination risk.

_STOP = set("""
a an and are as at be but by for from has have he in is it its of on or our she
that the their this to was we were will with you your has been have not no nor
which who whom whose if then than into about above below between within across
also more most much such only just very some any all each every other another
""".split())


def specificity_score(rationale: str, source_text: str) -> tuple[float, list[str]]:
    """Return (overlap %, missing content words) — words in the rationale
    that don't appear in the LOI. Many missing content words = the model
    is reaching for facts that aren't there."""
    def words(s: str) -> list[str]:
        return [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", s.lower()) if w not in _STOP and len(w) > 3]
    rat = words(rationale)
    src = set(words(source_text))
    if not rat:
        return 0.0, []
    missing = [w for w in rat if w not in src]
    overlap = 1 - (len(missing) / len(rat))
    return overlap, sorted(set(missing))[:8]


def render_score_table(per_model: dict[str, dict], include_truth_column: bool, truth_decision: str | None) -> None:
    rows: dict[str, dict[str, str]] = {
        "mission_fit (score)": {},
        "mission_fit (model's reasoning)": {},
        "strategic_alignment (score)": {},
        "strategic_alignment (model's reasoning)": {},
        "Weighted score": {},
        "Decision": {},
    }
    if include_truth_column:
        rows["Matches ground truth?"] = {}

    for model_id, r in per_model.items():
        if not r:
            for k in rows:
                rows[k][model_id] = "—"
            continue
        scores = {c["id"]: c for c in r["criterion_scores"]} if isinstance(r.get("criterion_scores"), list) else {}
        mf = scores.get("mission_fit", {"score": "?", "rationale": "—"})
        sa = scores.get("strategic_alignment", {"score": "?", "rationale": "—"})
        rows["mission_fit (score)"][model_id] = f"{mf['score']}/5"
        rows["mission_fit (model's reasoning)"][model_id] = mf["rationale"]
        rows["strategic_alignment (score)"][model_id] = f"{sa['score']}/5"
        rows["strategic_alignment (model's reasoning)"][model_id] = sa["rationale"]
        rows["Weighted score"][model_id] = str(r["weighted_score"])
        rows["Decision"][model_id] = r["decision"]
        if include_truth_column:
            rows["Matches ground truth?"][model_id] = (
                "✅ yes" if r["decision"] == truth_decision else "❌ no"
            )

    df = pd.DataFrame(rows).T
    df.index.name = ""
    st.dataframe(df, use_container_width=True)


# ----- Setup ---------------------------------------------------------------

st.set_page_config(page_title="FUSF LOI Eval", page_icon="🧠", layout="centered")

rubric = load_rubric()
lois = load_lois()
results_by_model = load_results()
results_index = {(m, r["loi_id"]): r for m, rs in results_by_model.items() for r in rs}

if not results_by_model:
    st.warning("No evaluation results. Run `python run_evals.py` first.")
    st.stop()


# ----- Header --------------------------------------------------------------

st.title("FUSF LOI Eval — How would I check if the model is good enough?")

st.markdown(
    "If FUSF is going to use an LLM to triage Letters of Intent, the question "
    "isn't *can it produce a FUND/DECLINE label* — every model can do that. "
    "The question is **how do you know it's actually any good?** "
    "This demo walks through the framework I'd use, applied to two LOIs and two models."
)

st.info(
    "**The framework — one baseline + four checks:**  \n"
    "**0.** How much do FUSF's own reviewers agree with each other? *(without this, every accuracy number is meaningless)*  \n"
    "**1.** Does the model agree with the final decision?  \n"
    "**2.** Does it agree for the *right reasons* — per-criterion, not just yes/no?  \n"
    "**3.** Is it making things up — hallucinating facts, or writing generic filler?  \n"
    "**4.** Where does it work and where does it break — by disease area, by clarity, by model confidence?"
)


# ----- 0. The baseline -----------------------------------------------------

st.divider()
st.header("0. Baseline — human reviewer agreement")

st.markdown(
    "Before scoring the model, score the humans. If two FUSF reviewers given the "
    "same LOI only agree 70% of the time, then a model agreeing 75% is **already at "
    "human level**. Without this number, an accuracy of 80% could be excellent or "
    "terrible — you can't tell."
)
st.markdown(
    "**What I'd compute (with FUSF's data):** pull ~50 historical LOIs that were "
    "reviewed by ≥2 reviewers, compute pairwise % agreement and Cohen's κ on the "
    "FUND/DECLINE label and on each rubric criterion. That number becomes the "
    "ceiling every other section is measured against."
)

base_a, base_b, base_c = st.columns(3)
base_a.metric("Reviewer pairs", "—", help="N pairs of reviewers who scored the same LOI")
base_b.metric("% agreement", "—", help="How often two reviewers picked the same final decision")
base_c.metric("Cohen's κ", "—", help="Agreement adjusted for chance. <0.4 weak, 0.4–0.6 moderate, >0.6 strong.")
st.caption("Empty until FUSF shares historical reviewer data — the placeholder is the point.")


# ----- 1. Decision agreement -----------------------------------------------

st.divider()
st.header("1. Did the model agree with the final decision?")

st.markdown(
    "The headline number. For each LOI, did the model's FUND/DECLINE match what "
    "FUSF actually did?"
)

# Build the agreement table across all LOIs and models we have
agree_rows = []
for loi in lois:
    truth_dec = loi["ground_truth"]["decision"]
    row = {"LOI": loi["id"], "Truth": truth_dec}
    for model_id in results_by_model:
        r = results_index.get((model_id, loi["id"]))
        if not r:
            row[model_id] = "—"
        else:
            row[model_id] = f"{r['decision']} {'✅' if r['decision'] == truth_dec else '❌'}"
    agree_rows.append(row)

st.dataframe(pd.DataFrame(agree_rows), use_container_width=True, hide_index=True)

# Per-model accuracy
acc_cols = st.columns(len(results_by_model))
for i, model_id in enumerate(results_by_model):
    correct = sum(
        1 for loi in lois
        if (r := results_index.get((model_id, loi["id"])))
        and r["decision"] == loi["ground_truth"]["decision"]
    )
    n = len(lois)
    acc_cols[i].metric(model_id, f"{correct}/{n}", f"{int(100 * correct / n)}% match")

st.caption(
    "**Honest caveat:** N=2 LOIs is too small to draw conclusions — these are "
    "directional. With 50+ historical LOIs we'd report accuracy with confidence "
    "intervals, plus a confusion matrix (false-fund vs false-decline costs are "
    "different — funding the wrong project costs $100K, declining a good one is invisible)."
)


# ----- 2. Right reasons ----------------------------------------------------

st.divider()
st.header("2. Did it agree for the right reasons?")

st.markdown(
    "A model can pick the right FUND/DECLINE while completely misreading the science. "
    "Section 1 misses that. So we score each rubric criterion separately and check "
    "whether the model's per-criterion scores line up with what reviewers actually wrote."
)
st.markdown(
    "**With FUSF data:** correlate model-vs-reviewer scores per criterion (Spearman's "
    "ρ on `mission_fit`, `strategic_alignment`, `scientific_merit`, `feasibility`, `team`). "
    "A model that's 80% accurate on the final decision but ρ=0.1 on `scientific_merit` "
    "is **right by accident** — it can't be trusted to triage borderline cases."
)

st.markdown("**On this dataset (illustrative):**")
loi_options = {f"{l['id']} — {l['title']}": l for l in lois}
choice = st.selectbox("Inspect an LOI", list(loi_options.keys()))
loi = loi_options[choice]
truth = loi["ground_truth"]

st.markdown(f"**Abstract:** {loi['abstract']}")
st.markdown(f"**Ground truth:** {'🟢' if truth['decision'] == 'FUND' else '🔴'} **{truth['decision']}** — {truth['rationale']}")

per_model_canned = {
    model_id: results_index.get((model_id, loi["id"]))
    for model_id in results_by_model
}
render_score_table(per_model_canned, include_truth_column=True, truth_decision=truth["decision"])


# ----- 3. Hallucination & specificity audit --------------------------------

st.divider()
st.header("3. Is the model making things up?")

st.markdown(
    "The model writes a justification for each score. Two questions to ask of every one:  \n"
    "  **(a)** Are the facts it cites actually in the LOI?  \n"
    "  **(b)** Is the reasoning specific to *this* proposal, or generic filler that could apply to anything?"
)
st.markdown(
    "**With FUSF data:** sample 30–50 explanations and hand-rate each one *yes/no* "
    "on those two questions. Report the hallucination rate and the generic-filler rate "
    "as separate numbers."
)

st.markdown("**Cheap automatic proxy on this LOI:** what % of the model's rationale words also appear in the LOI? Low overlap = the model is reaching for facts that aren't on the page.")

source_text = f"{loi['title']} {loi['abstract']}"
spec_rows = []
for model_id, r in per_model_canned.items():
    if not r or not isinstance(r.get("criterion_scores"), list):
        continue
    for c in r["criterion_scores"]:
        overlap, missing = specificity_score(c["rationale"], source_text)
        spec_rows.append({
            "Model": model_id,
            "Criterion": c["id"],
            "Score": f"{c['score']}/5",
            "Words from LOI": f"{int(overlap * 100)}%",
            "Words not in LOI": ", ".join(missing) if missing else "—",
        })
if spec_rows:
    st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)

# Spotlight the LOI-008 hallucination if that LOI is selected
disagreements = [
    (m, results_index[(m, loi["id"])])
    for m in results_by_model
    if (m, loi["id"]) in results_index
    and results_index[(m, loi["id"])]["decision"] != truth["decision"]
]
if loi["id"] == "LOI-008" and disagreements:
    st.error(
        "**Caught one.** LOI-008 explicitly says delivery is via **lipid nanoparticles** — "
        "no focused ultrasound at all. But at least one model's `mission_fit` rationale "
        "claims the project *uses focused ultrasound*. That's not a low score — it's a "
        "hallucinated fact used to justify a FUND. The kind of failure a binary-accuracy "
        "metric would never surface."
    )

st.caption(
    "This proxy only flags missing words, not subtle misreadings. The real audit is human "
    "reading, which is exactly why we'd cap the sample at 30–50."
)


# ----- 4. Where it works / where it breaks ---------------------------------

st.divider()
st.header("4. Where does it work, and where does it break?")

st.markdown(
    "One overall accuracy hides everything that matters. Slice the results so leadership "
    "knows where to trust it and where not to:"
)
st.markdown(
    "- **By disease area** — does it handle oncology better than neurology? Veterinary?\n"
    "- **By clarity** — does it nail clear yes/no LOIs but flounder on borderline ones?\n"
    "- **By stated confidence** — when the model says \"I'm confident,\" is it actually right more often? "
    "(If not, the confidence signal is unusable for triage.)"
)
st.markdown("**Sketch on this dataset (N=2 — illustrative only):**")

slice_rows = []
for loi in lois:
    truth_dec = loi["ground_truth"]["decision"]
    # Crude disease-area tag from the title/abstract
    text = (loi["title"] + " " + loi["abstract"]).lower()
    if any(w in text for w in ["glioblastoma", "dipg", "glioma", "brain tumor"]):
        area = "Oncology — brain"
    elif any(w in text for w in ["breast", "pancreatic", "metastatic"]):
        area = "Oncology — other"
    elif any(w in text for w in ["alzheimer", "parkinson", "huntington", "als"]):
        area = "Neurodegenerative"
    else:
        area = "Other"
    n_models = sum(
        1 for m in results_by_model
        if (r := results_index.get((m, loi["id"]))) and r["decision"] == truth_dec
    )
    slice_rows.append({
        "LOI": loi["id"],
        "Disease area": area,
        "Truth": truth_dec,
        "Models correct": f"{n_models}/{len(results_by_model)}",
    })
st.dataframe(pd.DataFrame(slice_rows), use_container_width=True, hide_index=True)

st.caption(
    "With ~50 LOIs we'd compute accuracy *per slice* with confidence intervals, plus "
    "a confidence-vs-correctness calibration curve. A pretty headline number that "
    "hides 40% accuracy on neurodegenerative LOIs is a deployment landmine."
)


# ----- 5. Recommendation ---------------------------------------------------

st.divider()
st.header("What I'd hand FUSF at the end")

st.markdown(
    "**Not a single accuracy number — a deployment recommendation.** Something like:"
)
st.success(
    "> *Use the model to triage **clear-cut** cases (auto-decline LOIs that don't use FUS, "
    "auto-flag obvious priority matches for human review). Send **borderline** cases "
    "straight to humans. Don't use it on **neurodegenerative** LOIs yet — accuracy is "
    "below reviewer-agreement floor in that slice. Re-evaluate after [N] more historical "
    "LOIs are labeled.*"
)
st.markdown(
    "That's the deliverable. It tells leadership where the model adds value, where it "
    "doesn't, and what would have to change for it to be trusted in the harder slices. "
    "Which is the responsible-AI-deployment piece in the job description."
)


# ----- 6. Try it on any paper ----------------------------------------------

st.divider()
st.header("Live demo — try it on any paper")
st.caption(
    "Upload a PDF or paste an abstract. Both models will score it live. No ground-truth "
    "column here — for arbitrary text we don't know the right answer, so this is model "
    "output only."
)


def extract_pdf_text(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file.getvalue()))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


col_pdf, col_text = st.columns(2)
with col_pdf:
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="upload")
with col_text:
    pasted = st.text_area(
        "...or paste an abstract",
        height=180,
        placeholder="Paste 1–3 paragraphs of an abstract or proposal text here.",
    )

submit = st.button(
    "Run both models",
    type="primary",
    disabled=not (uploaded or pasted.strip()),
)

if submit:
    text = ""
    try:
        if uploaded:
            text = extract_pdf_text(uploaded)
        else:
            text = pasted
    except Exception as e:
        st.error(f"Could not read input: {e}")
        text = ""

    text = text.strip()[:3500]
    if not text:
        st.warning("No usable text extracted. Try pasting the abstract directly.")
    else:
        from eval import evaluate_loi, get_available_providers
        providers = get_available_providers()
        if not providers:
            st.error(
                "No API keys configured. On Streamlit Cloud, add `GROQ_API_KEY` and/or "
                "`OPENAI_API_KEY` in **App settings → Secrets**."
            )
        else:
            user_loi = {
                "id": "user-input",
                "title": "User-submitted text",
                "pi": "—",
                "abstract": text,
            }
            live_results: dict[str, dict] = {}
            for p in providers:
                model_id = f"{p.name} / {p.model}"
                with st.spinner(f"Scoring with {model_id}…"):
                    res = evaluate_loi(user_loi, rubric, p)
                    live_results[model_id] = res.to_dict()
            st.session_state["live_results"] = live_results
            st.session_state["live_text"] = text

if "live_results" in st.session_state:
    with st.expander("Text the models actually saw", expanded=False):
        st.write(st.session_state["live_text"])
    render_score_table(
        st.session_state["live_results"],
        include_truth_column=False,
        truth_decision=None,
    )


# ----- Limitations ---------------------------------------------------------

st.divider()
st.subheader("Limitations of this demo")
st.markdown(
    """
- **N=2 LOIs** — directional only. The framework is the deliverable; numbers fill in once FUSF shares historical data.
- **One LOI is synthetic** — LOI-008 was deliberately constructed to test a specific failure mode (priority keywords without focused ultrasound).
- **Ground truth is mine** — written from FUSF's published criteria, not pulled from real review records.
- **No reviewer-pair data** — the baseline panel is empty for that reason. With FUSF's data it's the first thing I'd compute.
- **Stage 1 only** — full proposal stage adds scientific merit, feasibility, team, and eligibility, which need the full proposal text.
- **Two models** — Llama 3.3 70B and GPT-4o-mini. Real evaluation would include frontier models.
- **Specificity proxy is naive** — word overlap flags missing facts, not subtle misreadings. Real audit is human reading 30–50 cases.
"""
)


# ----- Footer --------------------------------------------------------------

st.divider()
st.caption(
    f"FUSF mission: *{rubric['mission']}* · "
    f"Models compared: {', '.join(results_by_model.keys())}"
)
