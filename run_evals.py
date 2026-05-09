"""Batch runner: every LOI x every available provider, save to results/."""
from __future__ import annotations
import json
from pathlib import Path

from dotenv import load_dotenv

from eval import evaluate_loi, get_available_providers


ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def main() -> None:
    load_dotenv()
    rubric = json.loads((DATA / "rubric.json").read_text())
    lois = json.loads((DATA / "lois.json").read_text())
    providers = get_available_providers()

    print(f"LOIs: {len(lois)}    Providers: {[p.model for p in providers]}\n")

    for provider in providers:
        out_path = RESULTS / f"{provider.name}__{provider.model}.json".replace("/", "_")
        results = []
        for loi in lois:
            print(f"  [{provider.model}] {loi['id']} ... ", end="", flush=True)
            result = evaluate_loi(loi, rubric, provider)
            print(f"{result.decision} (weighted {result.weighted_score})")
            results.append(result.to_dict())
        out_path.write_text(json.dumps(results, indent=2))
        print(f"  -> {out_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
