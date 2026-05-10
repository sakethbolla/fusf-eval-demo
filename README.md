# FUSF LOI Evaluation Demo

A small prototype showing how AI could help the **Focused Ultrasound Foundation (FUSF)** screen Letters of Intent — and, more importantly, **how to check if the AI is doing it correctly**.

## What it does

FUSF gets research proposals (LOIs) and screens each one for two things:
1. Does it use focused ultrasound to treat disease?
2. Does it match FUSF's current priorities (specific diseases + mechanisms)?

This demo:
- Sends a batch of LOIs to multiple LLMs (Groq, Gemini, Claude, GPT)
- Each model returns a score + FUND/DECLINE decision in JSON
- Compares every model's answer to the known correct answer
- Shows agreement, disagreement, and failure points in a Streamlit UI

## How it works

```
data/lois.json   ─┐
                  ├──► eval/runner.py ──► results/<model>.json ──► app.py (Streamlit)
data/rubric.json ─┘
```

- `data/rubric.json` — FUSF's criteria (mission, priorities, scoring rules)
- `data/lois.json` — sample LOIs, each labeled with the real FUND/DECLINE outcome
- `eval/prompts.py` — turns the rubric into the prompt the LLM sees
- `eval/runner.py` — runs one LOI through one model, parses the JSON
- `run_evals.py` — runs every LOI through every model
- `app.py` — Streamlit dashboard

## Run it

```bash
cd ~/fusf-eval-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys (or leave empty for mock mode)
python run_evals.py
streamlit run app.py
```

## How "is the LLM good enough?" is measured — one baseline + four checks

**0. Baseline — human reviewer agreement.** Score the humans first. If two reviewers
only agree 70% on the same LOI, then a model agreeing 75% is at human level. Without
this number, every accuracy figure is meaningless. Almost every eval skips this.

**1. Did the model agree with the final decision?** Per-LOI FUND/DECLINE match.
The headline number — but only meaningful relative to (0).

**2. Did it agree for the *right reasons*?** Per-criterion scoring (mission fit,
strategic alignment, scientific merit, feasibility, team) correlated against
reviewer scores. Catches models that get the final answer right while completely
misreading the science.

**3. Is it making things up?** Sample 30–50 model rationales. For each, check
(a) are the cited facts actually in the LOI, and (b) is the reasoning specific
to *this* proposal or generic filler that could apply to anything? The demo also
shows a cheap automatic proxy: % of rationale words that appear in the LOI text.

**4. Where does it work, and where does it break?** Slice results by disease
area, by how clear-cut the LOI is, and by stated model confidence. Tells
leadership where to trust it and where not to.

**Final deliverable:** not one accuracy score but a deployment recommendation —
"use it for clear-cut cases, send borderline ones to humans, don't use it on
disease-area X yet." That's the responsible-AI-deployment piece.

## What this is not

- Not a production tool. The dataset is small and partly synthetic.
- Not a claim that LLMs should replace FUSF's reviewers. The point is to **measure honestly** so the foundation can decide if and where AI adds value.
