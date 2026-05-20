#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from rdflib import Graph as _Graph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Reuse runner internals
from run_conference_experiment import (                                       # noqa: E402
    REGIMES, train_detector, evaluate_detector,
    run_regime, diff_graphs, write_diff_markdown, consistency_check,
    render_score_figure, render_ontology_figure, _to_jsonable,
)
from data.mit_data import load_mit_dataset                                    # noqa: E402
from reasoning.semantic_reasoner import SemanticReasoner                       # noqa: E402
from enrichment.llm_enrichment import (                                       # noqa: E402
    LLMOntologyEnricher, HumanValidator, build_enrichment_prompt,
)


LLM_LINEUP = [
    {"key": "qwen",     "name": "Qwen2.5-1.5B-Instruct",
     "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"key": "phi3",     "name": "Phi-3-mini-4k-instruct",
     "model_id": "microsoft/Phi-3-mini-4k-instruct"},
    {"key": "smollm2",  "name": "SmolLM2-1.7B-Instruct",
     "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
]


def snapshot_ontology(reasoner: SemanticReasoner, path: Path) -> _Graph:
    reasoner.save_ontology(str(path))
    g = _Graph()
    g.parse(str(path), format="turtle")
    return g


def run_single_llm(llm_cfg: dict, dataset: dict, model, scores, recons,
                   normal_scores, ontology_path: Path,
                   results_root: Path, train_history,
                   regime_key: str = "B") -> dict:
    """Run one (regime, LLM) combination into its own results folder."""
    out_dir = results_root / f"regime_{regime_key}_{llm_cfg['key']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fresh reasoner per LLM so each one starts from the same "before" graph.
    reasoner = SemanticReasoner(str(ontology_path))
    before_path = out_dir / "ontology_before.owl"
    g_before = snapshot_ontology(reasoner, before_path)

    print(f"\n{'='*78}\n  Loading LLM: {llm_cfg['name']}\n{'='*78}")
    t_load_start = time.time()
    enricher = LLMOntologyEnricher(
        model_name=llm_cfg["model_id"], device="cpu", use_mock=False,
    )
    t_load = time.time() - t_load_start
    print(f"  LLM loaded in {t_load:.1f}s")

    validator = HumanValidator(auto_approve=True)

    # run_regime writes to REGIMES[regime_key]["out_dir"]. Override that
    # transiently so per-LLM artifacts go into our results folder instead
    # of clobbering the original regime_B_permissive/.
    saved_out = REGIMES[regime_key]["out_dir"]
    REGIMES[regime_key]["out_dir"] = str(out_dir.relative_to(HERE))
    try:
        t_run_start = time.time()
        regime_summary = run_regime(
            regime_key, dataset, model, scores, recons, normal_scores,
            reasoner, enricher, validator, HERE,
            run_llms_for_unknown=True,
        )
        t_run = time.time() - t_run_start
    finally:
        REGIMES[regime_key]["out_dir"] = saved_out

    after_path = out_dir / "ontology_after.owl"
    g_after = snapshot_ontology(reasoner, after_path)

    diff = diff_graphs(g_before, g_after)
    (out_dir / "ontology_diff.json").write_text(
        json.dumps(_to_jsonable(diff), indent=2), encoding="utf-8")
    write_diff_markdown(diff, before_path, after_path,
                        out_dir / "ontology_diff.md")

    cons = {"before": consistency_check(g_before),
            "after":  consistency_check(g_after)}
    (out_dir / "consistency_check.json").write_text(
        json.dumps(cons, indent=2), encoding="utf-8")

    run_config = {
        "llm_key": llm_cfg["key"],
        "llm_name": llm_cfg["name"],
        "llm_model_id": llm_cfg["model_id"],
        "device": "cpu",
        "regime": REGIMES[regime_key]["name"],
        "regime_key": regime_key,
        "seed": 0,
        "llm_load_seconds": t_load,
        "regime_total_seconds": t_run,
        "regime_summary": regime_summary,
        "ontology_diff_summary": {
            "n_added": diff["n_added"],
            "added_classes": diff["added_classes"],
            "added_object_properties": diff["added_object_properties"],
            "added_datatype_properties": diff["added_datatype_properties"],
        },
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(_to_jsonable(run_config), indent=2), encoding="utf-8")

    # Free model memory before loading the next one
    del enricher
    try:
        import gc; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return run_config


def copy_existing_regime_a(results_root: Path) -> dict:
    """Copy the existing regime_A_conservative artifacts (detector-only,
    0 Stage-5 calls) into results_root for the comparison summary."""
    src = HERE / "regime_A_conservative"
    dst = results_root / "regime_A_conservative"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    summary = {}
    summary_file = dst / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return summary


def build_comparison_summary(per_llm: list, regime_a: dict,
                             results_root: Path) -> None:
    """Build comparison_summary.json — Qwen vs Phi-3 vs SmolLM2 on Regime B."""
    rows = []
    for run in per_llm:
        s = run["regime_summary"]
        d = run["ontology_diff_summary"]
        # Load per-call Stage-5 detail
        stage5 = []
        out_dir = results_root / f"regime_B_{run['llm_key']}"
        s5_file = out_dir / "stage5_llm_enrichments.json"
        if s5_file.exists():
            stage5 = json.loads(s5_file.read_text(encoding="utf-8"))
        total_llm_secs = sum(r.get("llm_seconds", 0.0) for r in stage5)
        fallback_count = sum(1 for r in stage5 if r.get("fallback_used"))
        parse_err = sum(1 for r in stage5
                        if (r.get("raw_proposal") or {}).get("_parse_error"))
        rows.append({
            "llm_key": run["llm_key"],
            "llm_name": run["llm_name"],
            "llm_model_id": run["llm_model_id"],
            "device": run["device"],
            "regime": run["regime"],
            "threshold": s["threshold"],
            "pre_filter": s["metrics_pre_filter"],
            "post_filter": s["metrics_post_stage3"],
            "n_candidates_post_stage3_validated": s["n_candidates_post_stage3_validated"],
            "n_stage5_calls": s["n_stage5_calls"],
            "n_stage7_triples_added_total": s["n_stage7_triples_added_total"],
            "ontology_diff": d,
            "llm_load_seconds": run["llm_load_seconds"],
            "regime_total_seconds": run["regime_total_seconds"],
            "stage5_total_llm_seconds": total_llm_secs,
            "n_fallback_activations": fallback_count,
            "n_parse_errors": parse_err,
            "per_call_records": [
                {"anomaly_id": r.get("anomaly_id"),
                 "trigger_path": r.get("trigger_path"),
                 "llm_seconds": r.get("llm_seconds"),
                 "fallback_used": r.get("fallback_used"),
                 "fallback_reason": r.get("fallback_reason"),
                 "stage6_validation_decision": r.get("stage6_validation_decision"),
                 "stage7_triples_added": r.get("stage7_triples_added"),
                 "proposed_classes": [
                     t.get("name") for t in
                     (r.get("raw_proposal") or {}).get("new_anomaly_types", [])
                 ],
                 } for r in stage5
            ],
        })

    summary = {
        "experiment": "Regime B real-LLM sweep on MIT-batch1 (A1-A7), TS-only",
        "regime_A_reference": {
            "note": ("Regime A (conservative) detector misses every injected "
                     "anomaly under the p95 threshold and fires 0 Stage-5 "
                     "calls. Included here only as the filter-protective "
                     "reference run; the LLM comparison runs under Regime B."),
            "summary": regime_a,
        },
        "regime_B_runs": rows,
    }
    (results_root / "comparison_summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"\nWrote {results_root / 'comparison_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", default="results_real_llm")
    parser.add_argument("--llms", nargs="+",
                        default=["qwen", "phi3", "smollm2"],
                        help="Subset of LLM keys to run.")
    args = parser.parse_args()

    results_root = HERE / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    print("#" * 78)
    print("# Real-LLM sweep: TS-only LSTM-AE on MIT-batch1, Regime B x LLMs")
    print("#" * 78)

    # 1. Dataset
    dataset = load_mit_dataset(seq_len=50, standardize=True)

    # 2. Train detector ONCE (seed 0)
    print(f"\n[detector] training LSTM-AE for {args.epochs} epochs, "
          f"seed {args.seed}")
    model, history, ts_normal, n_params = train_detector(
        dataset, epochs=args.epochs, seed=args.seed)

    # 3. Evaluate once
    scores, recons = evaluate_detector(model, dataset)
    normal_scores = scores[dataset["labels"] == 0]

    ontology_path = ROOT / "data" / "battery_ontology.ttl"

    # 4. Copy Regime A artifacts for context
    print("\n[regime A] copying existing detector-only artifacts...")
    regime_a_summary = copy_existing_regime_a(results_root)

    # 5. Per-LLM Regime B sweep
    selected = [c for c in LLM_LINEUP if c["key"] in args.llms]
    print(f"\nLLMs to run: {[c['name'] for c in selected]}")

    per_llm_results = []
    for cfg in selected:
        try:
            res = run_single_llm(
                cfg, dataset, model, scores, recons, normal_scores,
                ontology_path, results_root, history, regime_key="B",
            )
            per_llm_results.append(res)
        except Exception as e:
            print(f"\n[ERROR] LLM run failed for {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
            (results_root / f"regime_B_{cfg['key']}").mkdir(
                parents=True, exist_ok=True)
            (results_root / f"regime_B_{cfg['key']}" / "ERROR.txt").write_text(
                f"{cfg['name']} failed: {e}", encoding="utf-8")

    # 6. Save the exact Stage-5 prompt template (for the LaTeX)
    # Build a representative prompt using the trigger record for A2/window 28.
    example_anomaly_info = {
        "score": 0.51,
        "patterns": ["temperature_rise", "voltage_drop"],
        "failure_mode": "Overdischarge",
        "subtype_proposal": "AbnormalTemperatureRise",
        "observations": {
            "anomaly_id": "A2",
            "type": "thermal_runaway",
            "cell_id": "MIT_b1_cell002",
            "cycle": 1044,
        },
    }
    example_semantic_context = {
        "existing_failure_modes": [
            "Overheating", "Overcharge", "Overdischarge",
            "ThermalRunaway", "InternalShortCircuit", "CapacityFade",
        ],
    }
    example_prompt = build_enrichment_prompt(
        example_anomaly_info, example_semantic_context)
    (results_root / "stage5_prompt_template_example.txt").write_text(
        example_prompt, encoding="utf-8")

    # 7. Aggregate
    build_comparison_summary(per_llm_results, regime_a_summary, results_root)

    print("\nDone.")
    print(f"  results: {results_root}")


if __name__ == "__main__":
    main()
