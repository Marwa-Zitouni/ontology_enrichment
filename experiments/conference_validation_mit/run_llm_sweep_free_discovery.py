#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from rdflib import Graph as _Graph

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_conference_experiment import (                                       # noqa: E402
    REGIMES, train_detector, evaluate_detector,
    run_regime, diff_graphs, write_diff_markdown, consistency_check,
    _to_jsonable,
)
from data.mit_data import load_mit_dataset                                    # noqa: E402
from reasoning.semantic_reasoner import SemanticReasoner                       # noqa: E402
from enrichment.llm_enrichment import (                                       # noqa: E402
    LLMOntologyEnricher, HumanValidator, build_free_discovery_prompt,
)


LLM_LINEUP = [
    {"key": "qwen",     "name": "Qwen2.5-1.5B-Instruct",
     "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"key": "phi3",     "name": "Phi-3-mini-4k-instruct",
     "model_id": "microsoft/Phi-3-mini-4k-instruct"},
    {"key": "smollm2",  "name": "SmolLM2-1.7B-Instruct",
     "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
]


_CHANNEL_NAMES = ["temperature", "current", "voltage"]
_CHANNEL_UNITS = ["C", "A", "V"]
_CHANNEL_BOUNDS = [(0.0, 100.0), (-150.0, 150.0), (2.5, 4.5)]


def _unscale(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + values * (hi - lo)


def build_signal_statistics(window: np.ndarray) -> dict:
    """Return per-channel stats (mean, min, max, delta, std) in physical
    units for one (seq_len, 3) MIT window."""
    seq_len = window.shape[0]
    half = seq_len // 2
    stats = {}
    for ch_idx, (name, unit, (lo, hi)) in enumerate(
            zip(_CHANNEL_NAMES, _CHANNEL_UNITS, _CHANNEL_BOUNDS)):
        scaled = window[:, ch_idx]
        phys = _unscale(scaled, lo, hi)
        delta = float(phys[half:].mean() - phys[:half].mean())
        stats[name] = {
            "unit": unit,
            "mean": round(float(phys.mean()), 3),
            "min":  round(float(phys.min()), 3),
            "max":  round(float(phys.max()), 3),
            "std":  round(float(phys.std()), 3),
            "second_half_minus_first_half_mean": round(delta, 3),
        }
    return stats


def build_segment_description(stats: dict, patterns: list) -> str:
    """Short prose description grounded in the per-channel stats."""
    t = stats.get("temperature", {})
    c = stats.get("current", {})
    v = stats.get("voltage", {})
    parts = []
    if t:
        parts.append(
            f"temperature spans {t['min']:.2f}-{t['max']:.2f} C "
            f"(mean {t['mean']:.2f}, half-to-half delta "
            f"{t['second_half_minus_first_half_mean']:+.2f} C)")
    if c:
        parts.append(
            f"current spans {c['min']:.2f}-{c['max']:.2f} A "
            f"(mean {c['mean']:.2f}, std {c['std']:.2f})")
    if v:
        parts.append(
            f"voltage spans {v['min']:.2f}-{v['max']:.2f} V "
            f"(mean {v['mean']:.2f}, std {v['std']:.2f})")
    pat_str = ", ".join(patterns) if patterns else "(none derived by rule engine)"
    return ("; ".join(parts) +
            f". Rule-engine patterns observed: {pat_str}.")


def channels_carrying_anomaly(stats: dict) -> list:
    """Best-effort identification of which channel(s) carry the anomaly,
    based on per-channel deviation from typical ranges."""
    out = []
    t = stats.get("temperature", {})
    c = stats.get("current", {})
    v = stats.get("voltage", {})
    if t and (t.get("max", 0) > 45.0 or
              abs(t.get("second_half_minus_first_half_mean", 0)) > 3.0):
        out.append("temperature")
    if c and c.get("std", 0) > 25.0:
        out.append("current")
    if v and (v.get("min", 5.0) < 3.2 or v.get("std", 0) > 0.15):
        out.append("voltage")
    return out or ["(no single channel dominant)"]


# Ontology vocabulary extraction for the free-discovery prompt
def extract_ontology_vocabulary(reasoner: SemanticReasoner) -> dict:
    """Pull existing classes / object properties / datatype properties /
    rule labels from the live graph so the prompt only references the real
    vocabulary."""
    from rdflib import RDF, RDFS, OWL

    NS_LOCAL = "http://example.org/battery-ontology#"

    def _is_local(uri):
        return str(uri).startswith(NS_LOCAL)

    def _local(uri):
        s = str(uri)
        return s.split("#")[-1] if "#" in s else s

    g = reasoner.graph
    classes = sorted({_local(s) for s, _, _ in g.triples((None, RDF.type, OWL.Class))
                      if _is_local(s)})
    obj_props = sorted({_local(s) for s, _, _ in g.triples(
        (None, RDF.type, OWL.ObjectProperty)) if _is_local(s)})
    data_props = sorted({_local(s) for s, _, _ in g.triples(
        (None, RDF.type, OWL.DatatypeProperty)) if _is_local(s)})
    failure_modes = sorted({lbl for _, lbl in reasoner.get_all_failure_modes()})
    rule_labels = sorted({r["label"] for r in reasoner.get_causal_rules()})
    return {
        "existing_classes": classes,
        "existing_failure_modes": failure_modes,
        "existing_object_properties": obj_props or ["leadsTo"],
        "existing_datatype_properties": data_props,
        "known_rules": rule_labels,
    }


# Free-discovery enricher: subclass that builds rich anomaly_info 

class FreeDiscoveryEnricher(LLMOntologyEnricher):
    """Wraps the base enricher to inject per-trigger signal statistics and
    a real ontology-vocabulary block into the free-discovery prompt.

    The base run_regime() calls enricher.generate_enrichment(anomaly_info,
    semantic_context) with a thin anomaly_info; this subclass intercepts
    that call, looks up the candidate's window in `dataset` and the live
    vocabulary in `reasoner`, builds the evidence + vocabulary blocks, and
    then forwards to the underlying free-discovery LLM call.
    """

    def __init__(self, model_name, dataset, reasoner, **kwargs):
        super().__init__(model_name=model_name, prompt_mode="free_discovery",
                         **kwargs)
        self._dataset = dataset
        self._reasoner = reasoner
        # Cache the ontology vocabulary at construction; it does not change
        # during a single LLM run (we snapshot before each LLM).
        self._vocab = extract_ontology_vocabulary(reasoner)
        # Capture per-call evidence so the wrapper can serialise it later.
        self.last_evidence = None

    def generate_enrichment(self, anomaly_info: dict,
                            semantic_context: dict) -> dict:
        # Build evidence from the candidate's window
        obs = anomaly_info.get("observations") or {}
        sample_idx = None
        # Heuristic: cell_id + cycle lookup in the dataset
        cell_id = obs.get("cell_id")
        cycle = obs.get("cycle")
        ts_data = self._dataset["ts_data"]
        cells = self._dataset["cell_ids"]
        cycles = self._dataset["cycles"]
        for i in range(ts_data.shape[0]):
            if cells[i] == cell_id and int(cycles[i]) == int(cycle):
                sample_idx = i
                break

        if sample_idx is not None:
            window = ts_data[sample_idx]
            stats = build_signal_statistics(window)
            segment_desc = build_segment_description(
                stats, anomaly_info.get("patterns") or [])
            channels = channels_carrying_anomaly(stats)
        else:
            stats = {}
            segment_desc = "(window lookup failed)"
            channels = []

        # Augment anomaly_info with evidence fields the free-discovery
        enriched = dict(anomaly_info)
        enriched["signal_statistics"] = stats
        enriched["segment_description"] = segment_desc
        enriched["channels"] = channels
        enriched["confidence"] = anomaly_info.get("score")
        enriched["reconstruction_error"] = anomaly_info.get("score")
        enriched["stage4_status"] = ("confirmed" if anomaly_info.get("failure_mode")
                                     else "unknown")

        # Replace semantic_context with the live vocabulary block.
        enriched_ctx = dict(self._vocab)
        # Keep callers that expect existing_failure_modes happy.
        if "existing_failure_modes" not in enriched_ctx:
            enriched_ctx["existing_failure_modes"] = []

        self.last_evidence = {
            "candidate_id": obs.get("anomaly_id") or f"cell={cell_id},cycle={cycle}",
            "sample_idx": sample_idx,
            "signal_statistics": stats,
            "segment_description": segment_desc,
            "channels": channels,
            "vocabulary_snapshot": self._vocab,
        }

        return super().generate_enrichment(enriched, enriched_ctx)


def snapshot_ontology(reasoner: SemanticReasoner, path: Path) -> _Graph:
    reasoner.save_ontology(str(path))
    g = _Graph()
    g.parse(str(path), format="turtle")
    return g


def run_single_llm(llm_cfg: dict, dataset: dict, model, scores, recons,
                   normal_scores, ontology_path: Path,
                   results_root: Path,
                   regime_key: str = "B") -> dict:
    out_dir = results_root / f"regime_{regime_key}_{llm_cfg['key']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    reasoner = SemanticReasoner(str(ontology_path))
    before_path = out_dir / "ontology_before.owl"
    g_before = snapshot_ontology(reasoner, before_path)

    print(f"\n{'='*78}\n  Loading LLM (free_discovery): {llm_cfg['name']}\n{'='*78}")
    t_load_start = time.time()
    enricher = FreeDiscoveryEnricher(
        model_name=llm_cfg["model_id"], dataset=dataset, reasoner=reasoner,
        device="cpu", use_mock=False,
    )
    t_load = time.time() - t_load_start
    print(f"  LLM loaded in {t_load:.1f}s")

    validator = HumanValidator(auto_approve=True)

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
        "prompt_mode": "free_discovery",
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


def _decision_of(rec: dict) -> str:
    """Return the LLM's actual decision, preferring the pre-fallback record
    when the fallback replaced the proposal (so we never report an LLM's
    'add' decision as 'unknown' just because dedup ate its proposed class)."""
    prop = rec.get("raw_proposal") or {}
    pre_fb = prop.get("pre_fallback_free_discovery") or {}
    if pre_fb.get("decision"):
        return pre_fb["decision"]
    fd = prop.get("free_discovery") or {}
    return fd.get("decision") or "unknown"


def _confidences_of_record(rec: dict) -> list:
    """Pull confidences from the pre-fallback free_discovery block when
    available (so fallback-replaced records still expose the LLM's
    confidence numbers)."""
    prop = rec.get("raw_proposal") or {}
    pre_fb = prop.get("pre_fallback_free_discovery") or {}
    fd = pre_fb if pre_fb.get("decision") else (prop.get("free_discovery") or {})
    out = []
    for c in fd.get("candidate_new_classes", []) or []:
        try:
            out.append(float(c.get("confidence")))
        except Exception:
            pass
    for r in fd.get("candidate_new_relationships", []) or []:
        try:
            out.append(float(r.get("confidence")))
        except Exception:
            pass
    return out


def _confidences_of(rec: dict) -> list:
    fd = (rec.get("raw_proposal") or {}).get("free_discovery") or {}
    out = []
    for c in fd.get("candidate_new_classes", []) or []:
        v = c.get("confidence")
        try:
            out.append(float(v))
        except Exception:
            pass
    for r in fd.get("candidate_new_relationships", []) or []:
        v = r.get("confidence")
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def build_comparison_summary(per_llm: list, regime_a: dict,
                             results_root: Path) -> None:
    rows = []
    for run in per_llm:
        s = run["regime_summary"]
        d = run["ontology_diff_summary"]
        stage5 = []
        out_dir = results_root / f"regime_B_{run['llm_key']}"
        s5_file = out_dir / "stage5_llm_enrichments.json"
        if s5_file.exists():
            stage5 = json.loads(s5_file.read_text(encoding="utf-8"))
        total_llm_secs = sum(r.get("llm_seconds", 0.0) for r in stage5)
        fallback_count = sum(1 for r in stage5 if r.get("fallback_used"))
        parse_err = sum(1 for r in stage5
                        if (r.get("raw_proposal") or {}).get("_parse_error"))

        decisions = [_decision_of(r) for r in stage5]
        decision_counts = {d: decisions.count(d)
                           for d in set(decisions)}
        all_confs = [c for r in stage5 for c in _confidences_of_record(r)]
        mean_conf = (sum(all_confs) / len(all_confs)) if all_confs else None

        def _drisk(r):
            prop = r.get("raw_proposal") or {}
            pre_fb = prop.get("pre_fallback_free_discovery") or {}
            fd = pre_fb if pre_fb.get("decision") else (
                prop.get("free_discovery") or {})
            return fd.get("duplicate_risk", []) or []

        duplicate_risk_flags = sum(len(_drisk(r)) for r in stage5)

        rows.append({
            "llm_key": run["llm_key"],
            "llm_name": run["llm_name"],
            "llm_model_id": run["llm_model_id"],
            "device": run["device"],
            "regime": run["regime"],
            "prompt_mode": "free_discovery",
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
            "decision_counts": decision_counts,
            "mean_confidence": mean_conf,
            "n_duplicate_risk_flags": duplicate_risk_flags,
            "per_call_records": [
                {"anomaly_id": r.get("anomaly_id"),
                 "trigger_path": r.get("trigger_path"),
                 "llm_seconds": r.get("llm_seconds"),
                 "fallback_used": r.get("fallback_used"),
                 "fallback_reason": r.get("fallback_reason"),
                 "stage6_validation_decision": r.get("stage6_validation_decision"),
                 "stage7_triples_added": r.get("stage7_triples_added"),
                 "decision": _decision_of(r),
                 "confidences": _confidences_of_record(r),
                 "duplicate_risk_flags": _drisk(r),
                 "proposed_classes": [
                     t.get("name") for t in
                     (r.get("raw_proposal") or {}).get("new_anomaly_types", [])
                 ],
                 } for r in stage5
            ],
        })

    summary = {
        "experiment": ("Regime B free-discovery LLM sweep on MIT-batch1 "
                       "(A1-A7), TS-only"),
        "prompt_mode": "free_discovery",
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
    parser.add_argument("--results-dir",
                        default="results_real_llm_free_discovery")
    parser.add_argument("--llms", nargs="+",
                        default=["qwen", "phi3", "smollm2"],
                        help="Subset of LLM keys to run.")
    args = parser.parse_args()

    results_root = HERE / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    print("#" * 78)
    print("# Free-discovery LLM sweep: TS-only LSTM-AE on MIT-batch1")
    print("#" * 78)

    dataset = load_mit_dataset(seq_len=50, standardize=True)

    print(f"\n[detector] training LSTM-AE for {args.epochs} epochs, "
          f"seed {args.seed}")
    model, history, ts_normal, n_params = train_detector(
        dataset, epochs=args.epochs, seed=args.seed)

    scores, recons = evaluate_detector(model, dataset)
    normal_scores = scores[dataset["labels"] == 0]

    ontology_path = ROOT / "data" / "battery_ontology.ttl"

    print("\n[regime A] copying existing detector-only artifacts...")
    regime_a_summary = copy_existing_regime_a(results_root)

    # Save a representative free-discovery prompt for the LaTeX (using a
    # fresh reasoner just for the snapshot).
    snap_reasoner = SemanticReasoner(str(ontology_path))
    snap_vocab = extract_ontology_vocabulary(snap_reasoner)
    # Build evidence from window 28 (A2) for the example.
    a2_idx = None
    for i in range(dataset["ts_data"].shape[0]):
        if dataset["anomaly_ids"][i] == "A2":
            a2_idx = i; break
    if a2_idx is not None:
        stats = build_signal_statistics(dataset["ts_data"][a2_idx])
        seg = build_segment_description(
            stats, ["temperature_rise", "voltage_drop"])
        example_anomaly_info = {
            "score": 0.51,
            "reconstruction_error": 0.01249,
            "confidence": 0.51,
            "patterns": ["temperature_rise", "voltage_drop"],
            "channels": channels_carrying_anomaly(stats),
            "failure_mode": "Overdischarge",
            "stage4_status": "confirmed",
            "signal_statistics": stats,
            "segment_description": seg,
            "observations": {
                "candidate_id": "A2",
                "anomaly_id": "A2",
                "cell_id": "MIT_b1_cell002",
                "cycle": int(dataset["cycles"][a2_idx]),
            },
        }
        example_prompt = build_free_discovery_prompt(
            example_anomaly_info, snap_vocab)
        (results_root / "stage5_prompt_template_example.txt").write_text(
            example_prompt, encoding="utf-8")
        print(f"  wrote example prompt to "
              f"{results_root / 'stage5_prompt_template_example.txt'}")

    selected = [c for c in LLM_LINEUP if c["key"] in args.llms]
    print(f"\nLLMs to run: {[c['name'] for c in selected]}")

    per_llm_results = []
    for cfg in selected:
        try:
            res = run_single_llm(
                cfg, dataset, model, scores, recons, normal_scores,
                ontology_path, results_root, regime_key="B",
            )
            per_llm_results.append(res)
        except Exception as e:
            print(f"\n[ERROR] LLM run failed for {cfg['name']}: {e}")
            import traceback; traceback.print_exc()
            (results_root / f"regime_B_{cfg['key']}").mkdir(
                parents=True, exist_ok=True)
            (results_root / f"regime_B_{cfg['key']}" / "ERROR.txt").write_text(
                f"{cfg['name']} failed: {e}", encoding="utf-8")

    build_comparison_summary(per_llm_results, regime_a_summary, results_root)

    print("\nDone.")
    print(f"  results: {results_root}")


if __name__ == "__main__":
    main()
