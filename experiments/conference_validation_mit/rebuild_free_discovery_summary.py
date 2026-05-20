#!/usr/bin/env python3
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_RESULTS = HERE / "results_real_llm_free_discovery"
sys.path.insert(0, str(HERE))

from experiments.conference_validation_mit.run_llm_sweep_free_discovery import (  # noqa: E402
    build_comparison_summary, LLM_LINEUP,
)


def main() -> None:
    # Load per-LLM run_config.json
    per_llm = []
    for cfg in LLM_LINEUP:
        rc_file = ROOT_RESULTS / f"regime_B_{cfg['key']}" / "run_config.json"
        if rc_file.exists():
            per_llm.append(json.loads(rc_file.read_text(encoding="utf-8")))
        else:
            print(f"  MISSING: {rc_file}")

    # Load Regime A reference summary
    regime_a_file = ROOT_RESULTS / "regime_A_conservative" / "summary.json"
    regime_a = {}
    if regime_a_file.exists():
        regime_a = json.loads(regime_a_file.read_text(encoding="utf-8"))

    build_comparison_summary(per_llm, regime_a, ROOT_RESULTS)


if __name__ == "__main__":
    main()
