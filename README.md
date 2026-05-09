# FUSF LOI Evaluation Demo

A working prototype of how AI could assist the **Focused Ultrasound Foundation** with first-pass screening of Letters of Intent (LOIs) — and, more importantly, **how to evaluate whether the AI is doing it well**.

## What this is

FUSF receives Letters of Intent from researchers proposing focused-ultrasound studies. Foundation staff screen each LOI against two questions:
1. Does it fit the mission (non-invasive image-guided focused ultrasound to treat disease)?
2. Does it align with current strategic priorities (specific diseases + mechanisms)?

This demo runs a small batch of LOIs through multiple LLMs, compares each model's decision to a known ground truth, and surfaces where models agree, disagree, and fail.

## The pipeline

```
data/lois.json  ─┐
                 ├──► eval/runner.py ──► results/<model>.json ──► app.py (Streamlit)
data/rubric.json ┘         │
                           └─ provider abstraction (Groq / Gemini / Claude / GPT / Mock)
```

## Quick start

```bash
cd ~/fusf-eval-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys (or leave empty to use mock provider)
python run_evals.py    # runs every LOI through every configured model
streamlit run app.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `data/rubric.json` | FUSF's evaluation criteria, encoded |
| `data/lois.json` | Sample LOIs with ground-truth FUND/DECLINE labels |
| `eval/models.py` | Provider abstraction (Groq, Gemini, Claude, OpenAI, Mock) |
| `eval/prompts.py` | Rubric → prompt template |
| `eval/runner.py` | One LOI × one model → structured JSON eval |
| `run_evals.py` | Batch runner |
| `app.py` | Streamlit UI |

## What this is *not*

- A production system. The dataset is tiny and partly synthetic.
- A claim that LLMs should replace FUSF reviewers. The point is to *measure* AI performance honestly so the foundation can decide where (if anywhere) it adds value.
