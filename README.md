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

## The framework — 4 checks against ground truth

For each LOI, FUSF's published criteria define the right answer. Each LLM
independently reads the LOI and produces a decision. We then check:

1. **Decision agreement** — did the LLM reach the same FUND/DECLINE call as the
   ground truth?
2. **Per-criterion alignment** — did it score mission fit and strategic
   alignment the way FUSF's rubric would? A right decision for the wrong
   reasons is still a problem.
3. **Rationale faithfulness** — are the facts cited in the LLM's rationale
   actually present in the LOI, or is it making things up?
4. **Failure pattern** — where does each LLM break — by disease area, by how
   clear-cut the LOI is, by the model's own confidence?

*Note: at full scale this would also include inter-reviewer agreement as a
ceiling. This demo uses FUSF's published criteria as a single ground truth, so
that baseline doesn't apply here.*

## What this is not

- Not a production tool. The dataset is small and partly synthetic.
- Not a claim that LLMs should replace FUSF's reviewers. The point is to **measure honestly** so the foundation can decide if and where AI adds value.
