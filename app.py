"""Streamlit UI — single-page narrative of the FUSF LOI eval demo."""
from __future__ import annotations
import io
import json
import os
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


# ----- Top of page ---------------------------------------------------------

st.title("FUSF Letter of Intent — AI Evaluation Demo")


# ----- FUSF acceptance criteria -------------------------------------------

st.divider()
st.subheader("FUSF Funding criteria")

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
st.caption(
    "The **Decision** row shows the final FUND/DECLINE call. "
    "The **per-criterion score** rows show *why* — a model can pick the right "
    "FUND/DECLINE while completely misreading the science. Per-criterion scoring catches that."
)

per_model_canned = {
    model_id: results_index.get((model_id, loi["id"]))
    for model_id in results_by_model
}
render_score_table(per_model_canned, include_truth_column=True, truth_decision=truth["decision"])


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
    st.caption("Is the model making things up?")
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
                "*\"uses focused ultrasound for targeted immunomodulation.\"* "
                "**The model hallucinated focused ultrasound into the project** "
                "to justify its FUND decision. This is exactly the kind of "
                "failure mode an evaluation harness is built to catch."
            )
else:
    st.divider()
    st.success("Both models agreed with the ground truth on this LOI.")


# ----- Step 5: what this demo can't show ----------------------------------

st.divider()
st.subheader("5. What this demo can't show — and why that matters")
st.markdown(
    "Two LOIs and two models is enough to demonstrate the *mechanics* of checks 1–3. "
    "The pieces below need a real LOI corpus to fill in:"
)
st.markdown(
    "- **Rationale faithfulness audit (check 3, the proper version):** sample 30–50 model rationales "
    "and hand-rate each one for (a) facts cited that aren't in the LOI, (b) generic filler that could apply to any proposal.\n"
    "- **Failure-pattern slicing (check 4):** break accuracy down by disease area (oncology vs neuro), "
    "by how clear-cut the LOI is, and by stated model confidence. One overall accuracy "
    "hides everything that matters for deployment.\n"
    "- **Inter-reviewer baseline (out of scope here):** at full scale you'd also want pairwise reviewer "
    "agreement as a ceiling — but this demo uses FUSF's published criteria as a single ground truth, so that doesn't apply."
)


# ----- Step 6: the deliverable --------------------------------------------

st.divider()
st.subheader("What I'd hand FUSF at the end")
st.success(
    "Not one accuracy number — a **deployment recommendation**. Something like: "
    "*use the model to triage clear-cut cases, send borderline ones to humans, "
    "don't use it for [disease area X] yet.* That's where the evaluation framework "
    "earns its keep — telling leadership where the model can be trusted and where it can't."
)


# ----- Step 7: try it on any paper ----------------------------------------

st.divider()
st.subheader("6. Try it on any paper or abstract")
st.caption(
    "Upload a PDF or paste an abstract. Both models will score it live. "
    "There's no ground truth column here — we don't know the right answer "
    "for an arbitrary paper, so this is *model output only*."
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
                "No API keys configured on the server. "
                "On Streamlit Cloud, add `GROQ_API_KEY` and/or `OPENAI_API_KEY` "
                "in **App settings → Secrets**, then refresh."
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
- **Tiny dataset** — only 2 LOIs in the canned comparison. Findings are directional, not statistical
- **One LOI is synthetic** — LOI-008 was deliberately constructed to test a specific failure mode (priority keywords without focused ultrasound)
- **Ground truth labels are mine** — written from FUSF's published criteria, not pulled from FUSF's actual past decisions. A production system would replay real historical reviews
- **No reviewer-pair data** — so check #0 (baseline) is described, not computed. With FUSF's data it's the first thing I'd run
- **Stage 1 only** — the full proposal stage adds scientific merit, feasibility, team, and eligibility criteria, which need the actual proposal text
- **Two models only** — Llama 3.3 70B and GPT-4o-mini. A real evaluation would compare more, including frontier models like Claude and GPT-4o
- **PDF text extraction is naive** — we pass the first ~3,500 characters as raw text. Real pipelines would parse out the abstract specifically, handle scanned PDFs with OCR, and validate the extraction
"""
)


# ----- Footer note ---------------------------------------------------------

st.divider()
st.caption(
    f"FUSF mission: *{rubric['mission']}* · "
    f"Models compared: {', '.join(results_by_model.keys())}"
)
