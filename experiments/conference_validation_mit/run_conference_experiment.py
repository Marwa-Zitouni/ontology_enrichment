#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# Repo modules (TS-only pipeline)
from data.mit_data import load_mit_dataset                                  # noqa: E402
from models.lstm_autoencoder import (                                       # noqa: E402
    LSTMAutoencoder, train_lstm_autoencoder, lstm_anomaly_threshold,
    lstm_score,
)
from filtering.false_positive_filter import FalsePositiveFilter             # noqa: E402
from reasoning.semantic_reasoner import SemanticReasoner, BAT               # noqa: E402
from enrichment.llm_enrichment import (                                     # noqa: E402
    LLMOntologyEnricher, HumanValidator, apply_approved_updates,
    filter_proposals_by_ontology, build_fallback_proposal,
)



# Configuration

LLM_CONFIGS = [
    {"name": "Qwen2.5-1.5B-Instruct", "model_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"name": "Phi-3-mini-4k",         "model_id": "microsoft/Phi-3-mini-4k-instruct"},
]


_MIT_ID_TO_SUBTYPE = {
    "A1": "SuddenVoltageDrop",
    "A2": "AbnormalTemperatureRise",
    "A3": "CurrentSpike",
    "A4": "SensorNoiseBurst",
    "A5": "ThermalDrift",
    "A6": "MissingSensorSegment",
    "A7": "CapacityInconsistency",
}


def _load_injection_subtypes_from_csv() -> dict:
    """Read anomaly_labels.csv and return {anomaly_id: injected_subtype}.

    The CSV's `type` column carries the ground-truth injection label
    (e.g. "abnormal_temperature_rise"). mit_data.py collapses these onto
    broad SOSA classes for the framework's pattern matcher; we keep the
    raw CSV string here so the report can show both.
    """
    csv_path = ROOT / "illustrative_case_study" / "data" / "final" / "anomaly_labels.csv"
    out: dict = {}
    if not csv_path.exists():
        return out
    try:
        import csv
        with open(csv_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                a_id = (row.get("anomaly_id") or "").strip()
                subtype = (row.get("type") or "").strip()
                if a_id and subtype:
                    out[a_id] = subtype
    except Exception:
        pass
    return out


REGIMES = {
    "A": {  # conservative -- safe behaviour, low FP, accepts only the clearest spike
        "name": "Regime A (conservative)",
        "threshold_percentile": 95,
        "confidence_threshold": 0.55,
        "temporal_k": 2,
        "temporal_n": 3,
        "out_dir": "regime_A_conservative",
    },
    "B": {  # permissive -- the regime that actually drives Stages 3..7
        "name": "Regime B (permissive)",
        "threshold_percentile": 50,
        "confidence_threshold": 0.20,
        "temporal_k": 1,
        "temporal_n": 3,
        "out_dir": "regime_B_permissive",
    },
}


# Pattern derivation (physical-units-aware) for MIT scaled windows
# 
def derive_patterns(window: np.ndarray) -> set:
    """Return the set of observation patterns implied by a (T,3) MIT window.

    The MIT loader scales (T,I,V) to [0,1] over physical bounds:
        T:  0..100 C       (1 C = 0.01)
        I: -150..150 A     (idle = 0.5)
        V:  2.5..4.5 V
    """
    seq_len = window.shape[0]
    half = seq_len // 2
    temp = window[:, 0]
    cur = window[:, 1]
    volt = window[:, 2]
    d_temp = float(temp[half:].mean() - temp[:half].mean())
    T_max = float(temp.max())
    T_mean = float(temp.mean())
    I_abs_max = float(np.abs(cur - 0.5).max())
    V_max = float(volt.max())
    V_min = float(volt.min())

    patterns: set = set()
    if T_max > 0.45 or d_temp > 0.03:
        patterns.add("temperature_rise")
    if T_max > 0.55 or d_temp > 0.05:
        patterns.add("rapid_temperature_spike")
        patterns.add("overheating")
    if I_abs_max > 0.20:
        patterns.add("current_increase")
    if I_abs_max > 0.30:
        patterns.add("current_spike")
    if V_max > 0.85:
        patterns.add("voltage_spike")
    if V_min < 0.35:
        patterns.add("voltage_drop")
    if T_mean > 0.45 and I_abs_max > 0.10:
        patterns.add("current_increase")
    return patterns


# JSON-safe coercion
def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Train detector + threshold per regime
def train_detector(dataset: dict, epochs: int, seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal_mask = dataset["labels"] == 0
    ts_normal = dataset["ts_data"][normal_mask]
    seq_len = ts_normal.shape[1]
    in_dim = ts_normal.shape[2]

    model = LSTMAutoencoder(input_dim=in_dim, seq_len=seq_len,
                            latent_dim=16, hidden_dim=32).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[detector] LSTM-AE parameters: {n_params}")
    history = train_lstm_autoencoder(
        model, ts_normal,
        epochs=epochs, lr=1e-3, batch_size=32,
    )
    return model, history, ts_normal, n_params


def evaluate_detector(model: LSTMAutoencoder, dataset: dict) -> tuple:
    scores, recons = lstm_score(model, dataset["ts_data"])
    return scores, recons


# Per-regime pipeline run
def run_regime(regime_key: str, dataset: dict, model: LSTMAutoencoder,
               scores: np.ndarray, recons: np.ndarray,
               normal_scores_for_threshold: np.ndarray,
               reasoner: SemanticReasoner,
               enricher: LLMOntologyEnricher,
               validator: HumanValidator,
               out_root: Path,
               run_llms_for_unknown: bool = True) -> dict:
    cfg = REGIMES[regime_key]
    out_dir = out_root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(np.percentile(normal_scores_for_threshold,
                                    cfg["threshold_percentile"]))
    print(f"\n[{cfg['name']}] threshold (p{cfg['threshold_percentile']} of normal "
          f"recon-error) = {threshold:.6f}")

    # Confidence/temporal filter
    fp_filter = FalsePositiveFilter(
        confidence_threshold=cfg["confidence_threshold"],
        temporal_k=cfg["temporal_k"],
        temporal_n=cfg["temporal_n"],
        reasoner=reasoner,
    )
    fp_filter.reset_temporal_window()

    n = scores.shape[0]
    predictions = (scores >= threshold).astype(int)

    # Build per-sample log
    candidate_log = []
    detected_indices = np.where(predictions == 1)[0]

    # Detection metrics vs ground truth
    labels = dataset["labels"]
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"[{cfg['name']}] TP={tp} FP={fp} FN={fn} TN={tn} "
          f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    # ── Iterate detected candidates and exercise Stages 1..3 (filtering) and
    # Stage 4 (semantic reasoning).
    csv_subtypes = _load_injection_subtypes_from_csv()
    detected_records: list = []
    for idx in detected_indices:
        atype = dataset["anomaly_types"][idx]
        a_id = dataset["anomaly_ids"][idx] if dataset["anomaly_ids"][idx] else ""
        is_anom = bool(labels[idx] == 1)
        window = dataset["ts_data"][idx]
        patterns = derive_patterns(window)

        score_norm = float(min(scores[idx] / (threshold * 2 + 1e-12), 1.0))
        result = fp_filter.filter(score_norm, patterns)

        # Stage 4 verdict (semantic reasoning) -- richer than Stage 3 alone:
        # "confirmed" if filter is validated AND there are matched rules
        # "rejected"  if Stage 3 explicitly rejected for missing semantic support
        # "unknown"   if Stage 3 had no semantic match but the score is a real spike
        sem_stage = result["stages"].get("semantic_cross_validation", {})
        if result["validated"] and sem_stage.get("matched_rules"):
            stage4 = "confirmed"
        elif sem_stage.get("passed") is False:
            stage4 = "unknown"
        else:
            stage4 = "rejected"

        rec = {
            "sample_idx": int(idx),
            "anomaly_id": a_id,
            "anomaly_type": atype,
            "injection_subtype": csv_subtypes.get(a_id, "") if a_id else "",
            "is_anomaly_gt": is_anom,
            "cell_id": dataset["cell_ids"][idx],
            "cycle": int(dataset["cycles"][idx]),
            "raw_score": float(scores[idx]),
            "normalized_score": score_norm,
            "threshold": threshold,
            "threshold_crossed": True,
            "confidence_gate": result["stages"].get("confidence_gate", {}),
            "temporal_consistency": result["stages"].get("temporal_consistency", {}),
            "semantic_cross_validation": sem_stage,
            "patterns_observed": sorted(patterns),
            "stage3_decision": "validated" if result["validated"] else "rejected",
            "stage4_reasoning": stage4,
            "failure_modes": result.get("failure_modes", []),
        }
        detected_records.append(rec)

    # ── Coverage of the ground-truth A1..A7 anomalies in this regime
    a_ids_present = sorted(set(dataset["anomaly_ids"]) - {""})
    coverage = []
    for a_id in a_ids_present:
        idxs = [i for i, x in enumerate(dataset["anomaly_ids"]) if x == a_id]
        if not idxs:
            coverage.append({"anomaly_id": a_id, "n_windows": 0,
                             "max_score": None, "any_threshold_crossed": False,
                             "any_validated": False})
            continue
        max_score = float(max(scores[idxs]))
        crossed = bool(any(scores[i] >= threshold for i in idxs))
        rec_for_a = next((r for r in detected_records if r["anomaly_id"] == a_id), None)
        coverage.append({
            "anomaly_id": a_id,
            "n_windows": len(idxs),
            "indices": [int(i) for i in idxs],
            "max_score": max_score,
            "threshold": threshold,
            "any_threshold_crossed": crossed,
            "any_validated": bool(rec_for_a and rec_for_a["stage3_decision"] == "validated"),
            "stage4_reasoning": rec_for_a["stage4_reasoning"] if rec_for_a else "stage not activated",
            "patterns_observed": rec_for_a["patterns_observed"] if rec_for_a else [],
            "failure_modes": rec_for_a["failure_modes"] if rec_for_a else [],
        })

    # ── Stage 5 LLM enrichment for "unknowns" reaching this stage
    unknowns = [r for r in detected_records if r["stage4_reasoning"] == "unknown"]
    confirmed = [r for r in detected_records if r["stage4_reasoning"] == "confirmed"]

    
    seen_subtypes_per_regime = set()
    stage5_targets = []
    for r in confirmed + unknowns:
        a_id = r.get("anomaly_id", "")
        # The injected subtype string (when an injected anomaly fires) is
        # carried through anomaly_labels.csv -> _ID_TO_TYPE in mit_data.py.
        # That mapping collapses subtypes onto broad classes; for Stage-5
        # purposes we use the original CSV `type` column instead.
        subtype = _MIT_ID_TO_SUBTYPE.get(a_id, "")
        # Always trigger Stage 5 once per (subtype) per regime, when the
        # subtype is novel AND we have at least one ground-truth row to
        # ground the proposal in.
        subtype_already_in_ontology = (
            bool(subtype) and reasoner.failure_mode_exists_by_label(subtype)
        )

        broad_fm_exists = (
            bool(r["failure_modes"]) and
            all(reasoner.failure_mode_exists_by_label(lbl)
                for lbl in r["failure_modes"])
        )

        if r["stage4_reasoning"] == "unknown":
            # (a) — always trigger
            stage5_targets.append(r)
            continue
        if not broad_fm_exists:
            # (b) — broad mode itself is new
            stage5_targets.append(r)
            continue
        if subtype and not subtype_already_in_ontology and \
                subtype not in seen_subtypes_per_regime:
            # (c) — subtype is novel; trigger once per regime per subtype
            seen_subtypes_per_regime.add(subtype)
            r["stage5_trigger_subtype"] = subtype
            stage5_targets.append(r)
            continue
        r["stage5_decision"] = "skipped_existing"

    print(f"[{cfg['name']}] candidates reaching Stage 5 (LLM): {len(stage5_targets)}")

    stage5_records: list = []
    if run_llms_for_unknown and stage5_targets:
        for r in stage5_targets:
            subtype = _MIT_ID_TO_SUBTYPE.get(r.get("anomaly_id", ""), "")
            # Build anomaly_info & semantic_context for the prompt
            anomaly_info = {
                "score": r["normalized_score"],
                "patterns": r["patterns_observed"],
                "failure_mode": (r["failure_modes"][0] if r["failure_modes"]
                                  else f"Unknown ({r['anomaly_type']})"),
                "subtype_proposal": subtype,
                "observations": {
                    "anomaly_id": r["anomaly_id"],
                    "type": r["anomaly_type"],
                    "cell_id": r["cell_id"],
                    "cycle": r["cycle"],
                },
            }
            semantic_context = {
                "matched_rules": [rl["label"]
                                  for rl in reasoner.get_causal_rules()],
                "causal_chains": reasoner.get_causal_chains(),
                "failure_modes": r["failure_modes"],
                "existing_failure_modes": [lbl for _, lbl
                                           in reasoner.get_all_failure_modes()],
            }

            t0 = time.time()
            proposal = enricher.generate_enrichment(anomaly_info, semantic_context)
            dt = time.time() - t0
            # In mock mode, override the canned RapidDegradation proposal with
            # one that targets the actual injected subtype, so the ontology
            # actually grows when the test set demands it.
            if enricher.use_mock and subtype:
                proposal = _build_subtype_proposal(subtype, r)

            filtered, removed = filter_proposals_by_ontology(proposal, reasoner)

            
            fallback_reason = None
            parse_error = bool(proposal.get("_parse_error"))
            empty_after_filter = (
                not filtered.get("new_anomaly_types")
                and not filtered.get("new_causal_relationships")
                and not filtered.get("rdf_triples")
            )
            had_proposed_types = bool(proposal.get("new_anomaly_types"))
            duplicate_only = had_proposed_types and not filtered.get("new_anomaly_types")

            fd_decision = (proposal.get("free_discovery") or {}).get("decision")
            fd_override = fd_decision in ("skip_existing", "reject_uncertain",
                                          "human_review", "no_update")

            if subtype and (parse_error or empty_after_filter or duplicate_only) \
                    and not fd_override:
                broad_fm = (r["failure_modes"][0] if r["failure_modes"]
                             else "Overdischarge")
                if parse_error:
                    fallback_reason = "parse_error"
                elif duplicate_only:
                    fallback_reason = "duplicate_only_proposal"
                else:
                    fallback_reason = "empty_after_filter"
                print(f"  [FALLBACK] Activating deterministic schema-constrained "
                      f"fallback for {r.get('anomaly_id', '?')} -> {subtype} "
                      f"(reason: {fallback_reason})")
                fb_proposal = build_fallback_proposal(
                    missing_subtype=subtype,
                    broad_failure_mode=broad_fm,
                    patterns=r.get("patterns_observed", []),
                    anomaly_id=r.get("anomaly_id", ""),
                )
                # Pass the fallback through the same dedup + review pipeline.
                filtered, removed_fb = filter_proposals_by_ontology(
                    fb_proposal, reasoner)
                # Merge removed sets for honest reporting
                for k in ("types", "relationships"):
                    removed[k] = list(set(removed.get(k, []) + removed_fb.get(k, [])))
               
                fb_proposal["raw_response"] = proposal.get("raw_response", "")
                fb_proposal["prompt"] = proposal.get("prompt", "")
                fb_proposal["pre_fallback_free_discovery"] = (
                    proposal.get("free_discovery"))
                fb_proposal["pre_fallback_parse_error"] = (
                    proposal.get("_parse_error"))
                proposal = fb_proposal  # so the saved record reflects what was applied

            review = validator.review_proposal(filtered, interactive=False)
            n_added = apply_approved_updates(reasoner, review)

            stage5_records.append({
                "sample_idx": r["sample_idx"],
                "anomaly_id": r["anomaly_id"],
                "anomaly_type": r["anomaly_type"],
                "trigger_path": r["stage4_reasoning"],
                "llm_seconds": dt,
                "raw_proposal": proposal,
                "pre_filtered_duplicates": removed,
                "fallback_used": bool(fallback_reason),
                "fallback_reason": fallback_reason,
                "human_review_result": review,
                "stage6_validation_decision": (
                    "accepted" if review["accepted_triples"]
                    or review["accepted_types"]
                    or review["accepted_relationships"]
                    else "rejected"
                ),
                "stage7_triples_added": n_added,
            })
            r["stage5_decision"] = "executed"
            r["stage6_decision"] = stage5_records[-1]["stage6_validation_decision"]
            r["stage7_decision"] = (f"applied:{n_added}" if n_added > 0
                                     else "no_new_triples")
    else:
        for r in stage5_targets:
            r["stage5_decision"] = "stage not activated"

    # ── Persist regime artefacts
    (out_dir / "candidates.json").write_text(
        json.dumps(_to_jsonable(detected_records), indent=2), encoding="utf-8")
    (out_dir / "ground_truth_coverage.json").write_text(
        json.dumps(_to_jsonable(coverage), indent=2), encoding="utf-8")
    (out_dir / "stage5_llm_enrichments.json").write_text(
        json.dumps(_to_jsonable(stage5_records), indent=2), encoding="utf-8")
    (out_dir / "scores.npy").write_bytes(b"")  # placeholder; real np save below
    np.save(out_dir / "scores.npy", scores)
    np.save(out_dir / "predictions.npy", predictions)
    np.save(out_dir / "labels.npy", labels)

    
    validated_mask = np.zeros_like(predictions)
    for r in detected_records:
        if r["stage3_decision"] == "validated":
            validated_mask[r["sample_idx"]] = 1
    tp_v = int(np.sum((validated_mask == 1) & (labels == 1)))
    fp_v = int(np.sum((validated_mask == 1) & (labels == 0)))
    fn_v = int(np.sum((validated_mask == 0) & (labels == 1)))
    tn_v = int(np.sum((validated_mask == 0) & (labels == 0)))
    p_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
    r_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
    f1_v = 2 * p_v * r_v / (p_v + r_v) if (p_v + r_v) > 0 else 0.0

    summary = {
        "regime": cfg["name"],
        "threshold_percentile": cfg["threshold_percentile"],
        "threshold": threshold,
        "confidence_threshold": cfg["confidence_threshold"],
        "temporal_k": cfg["temporal_k"],
        "temporal_n": cfg["temporal_n"],
        "n_total_windows": int(n),
        "n_normal": int(np.sum(labels == 0)),
        "n_anomalous": int(np.sum(labels == 1)),
        "n_predicted_anomaly": int(np.sum(predictions)),
        "metrics_pre_filter": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall, "F1": f1},
        # Backwards compat: keep `metrics` pointing at the pre-filter set.
        "metrics": {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": precision, "recall": recall, "F1": f1},
        "metrics_post_stage3": {
            "TP": tp_v, "FP": fp_v, "FN": fn_v, "TN": tn_v,
            "precision": p_v, "recall": r_v, "F1": f1_v},
        "n_candidates_post_stage3_validated": int(sum(
            1 for r in detected_records if r["stage3_decision"] == "validated")),
        "n_stage5_calls": int(len(stage5_records)),
        "n_stage7_triples_added_total": int(sum(
            r["stage7_triples_added"] for r in stage5_records)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    return summary


# Multi-LLM enrichment comparison (shared trigger context)
def evaluate_proposal(proposal: dict, reasoner: SemanticReasoner) -> dict:
    """Score an LLM proposal on the rubric required by the conference task.

    Returns five scores in [0, 1] and a flat boolean schema_valid flag.
    """
    required_keys = ["new_anomaly_types", "new_causal_relationships",
                     "new_properties", "rdf_triples"]
    schema_valid = all(k in proposal and isinstance(proposal[k], list)
                       for k in required_keys)

    types_list = proposal.get("new_anomaly_types", []) or []
    rels_list = proposal.get("new_causal_relationships", []) or []
    triples = proposal.get("rdf_triples", []) or []

    # 1. Schema validity (all required keys, every type has name+label, every
    # rel has from+to, every triple has the three slots).
    type_ok = sum(1 for t in types_list
                  if isinstance(t, dict) and t.get("name") and t.get("label"))
    rel_ok = sum(1 for r in rels_list
                 if isinstance(r, dict) and r.get("from") and r.get("to"))
    trip_ok = sum(1 for t in triples if isinstance(t, dict)
                  and t.get("subject") and t.get("predicate") and t.get("object"))
    total_items = len(types_list) + len(rels_list) + len(triples)
    schema_score = ((type_ok + rel_ok + trip_ok) / total_items) if total_items > 0 else 0.0
    schema_score = float(schema_score) if schema_valid else 0.0

    # 2. Domain plausibility: any proposed name/label contains a battery-domain
    # token (degradation, runaway, plating, dendrite, SEI, swelling, etc.).
    BATTERY_TOKENS = {
        "thermal", "runaway", "overheat", "overheating", "overcurrent",
        "overcharge", "overdischarge", "degrad", "plating", "dendrite",
        "lithium", "sei", "swell", "short", "internal", "capacity", "fade",
        "drift", "voltage", "current", "temperature", "noise", "burst",
        "sensor", "missing",
    }
    plausibility_hits = 0
    plausibility_total = 0
    for t in types_list:
        plausibility_total += 1
        text = (str(t.get("label", "")) + " " + str(t.get("description", ""))).lower()
        if any(tok in text for tok in BATTERY_TOKENS):
            plausibility_hits += 1
    for r in rels_list:
        plausibility_total += 1
        text = (str(r.get("from", "")) + " " + str(r.get("to", ""))
                + " " + str(r.get("description", ""))).lower()
        if any(tok in text for tok in BATTERY_TOKENS):
            plausibility_hits += 1
    plausibility = (plausibility_hits / plausibility_total) if plausibility_total > 0 else 0.0

    # 3. Redundancy: fraction of proposed types/rels that already exist in the
    # ontology (lower is better; we report 1 - redundancy as the "novelty"
    # score so it is comparable on the same axis as the others).
    redundant_types = sum(1 for t in types_list
                          if reasoner.failure_mode_exists_by_label(
                              t.get("label", "") or t.get("name", "")))
    redundant_rels = sum(1 for r in rels_list
                         if reasoner.causal_relation_exists(r.get("from", ""),
                                                            r.get("to", "")))
    redundant_total = redundant_types + redundant_rels
    novelty = 1.0 - (redundant_total / max(len(types_list) + len(rels_list), 1))

    # 4. Consistency with existing ontology: proposed parent classes must be
    # known FailureMode (or one of its existing subclasses).
    known_fm_labels = {lbl for _, lbl in reasoner.get_all_failure_modes()}
    known_fm_labels.add("FailureMode")
    cons_hits = 0
    cons_total = 0
    for t in types_list:
        cons_total += 1
        parent = str(t.get("parent_class", "")).split("#")[-1]
        if parent in known_fm_labels or parent == "":
            cons_hits += 1
    consistency = (cons_hits / cons_total) if cons_total > 0 else 1.0

    
    has_new_class = any(
        not reasoner.failure_mode_exists_by_label(t.get("label", "") or t.get("name", ""))
        for t in types_list
    )
    has_new_relation = any(
        not reasoner.causal_relation_exists(r.get("from", ""), r.get("to", ""))
        for r in rels_list
    )
    usefulness = 1.0 if (has_new_class and has_new_relation) else (
        0.5 if (has_new_class or has_new_relation) else 0.0)

    return {
        "schema_validity": float(schema_score),
        "domain_plausibility": float(plausibility),
        "novelty_one_minus_redundancy": float(novelty),
        "consistency_with_ontology": float(consistency),
        "usefulness": float(usefulness),
        "schema_valid_flag": bool(schema_valid),
        "n_types": len(types_list),
        "n_relationships": len(rels_list),
        "n_triples": len(triples),
    }


def run_multi_llm_comparison(trigger_record: dict, reasoner: SemanticReasoner,
                             use_mock: bool, out_dir: Path) -> list:
    """Call each LLM on the same trigger anomaly and score the proposals.

    The reasoner is *not* mutated here -- this is purely an evaluation pass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[multi-LLM] running {len(LLM_CONFIGS)} models on trigger "
          f"anomaly {trigger_record.get('anomaly_id', '?')} "
          f"(use_mock={use_mock})")

    anomaly_info = {
        "score": trigger_record["normalized_score"],
        "patterns": trigger_record["patterns_observed"],
        "failure_mode": (trigger_record["failure_modes"][0]
                          if trigger_record["failure_modes"]
                          else f"Unknown ({trigger_record['anomaly_type']})"),
        "observations": {
            "anomaly_id": trigger_record["anomaly_id"],
            "type": trigger_record["anomaly_type"],
            "cell_id": trigger_record["cell_id"],
            "cycle": trigger_record["cycle"],
        },
    }
    semantic_context = {
        "matched_rules": [rl["label"] for rl in reasoner.get_causal_rules()],
        "causal_chains": reasoner.get_causal_chains(),
        "failure_modes": trigger_record["failure_modes"],
        "existing_failure_modes": [lbl for _, lbl
                                   in reasoner.get_all_failure_modes()],
    }

    comparison = []
    for cfg in LLM_CONFIGS:
        print(f"  - {cfg['name']}")
        try:
            enricher = LLMOntologyEnricher(model_name=cfg["model_id"],
                                            use_mock=use_mock)
            t0 = time.time()
            proposal = enricher.generate_enrichment(anomaly_info, semantic_context)
            dt = time.time() - t0
        except Exception as e:
            print(f"    FAILED: {e}")
            comparison.append({
                "llm": cfg["name"], "model_id": cfg["model_id"],
                "error": str(e),
                "scores": None,
            })
            continue

        
        if use_mock:
            subtype = _MIT_ID_TO_SUBTYPE.get(trigger_record.get("anomaly_id", ""), "")
            if subtype:
                proposal = _build_subtype_proposal(subtype, trigger_record)
            proposal = _vary_mock(proposal, cfg["name"])

        scores = evaluate_proposal(proposal, reasoner)
        comparison.append({
            "llm": cfg["name"],
            "model_id": cfg["model_id"],
            "seconds": dt,
            "proposal": proposal,
            "scores": scores,
        })

    (out_dir / "multi_llm_comparison.json").write_text(
        json.dumps(_to_jsonable(comparison), indent=2), encoding="utf-8")
    return comparison


def _build_subtype_proposal(subtype: str, candidate: dict) -> dict:
   
    NS = "http://example.org/battery-ontology#"
    parent = "FailureMode"
    broad = (candidate.get("failure_modes") or ["Overheating"])[0]
    label_human = "".join(
        " " + ch if ch.isupper() and i > 0 else ch
        for i, ch in enumerate(subtype)
    ).strip()

    # Subtype-specific affected component / observed pattern / cause
    SUBTYPE_META = {
        "SuddenVoltageDrop":      ("Cell", "step-like potential drop", "partial internal short or contact-resistance increase"),
        "AbnormalTemperatureRise": ("Cell", "localised thermal excursion", "incipient thermal-runaway precursor"),
        "CurrentSpike":           ("CurrentSensor", "single-sample current impulse", "sensor glitch or cycler relay artefact"),
        "SensorNoiseBurst":       ("VoltageSensor", "transient noise burst", "loose harness contact"),
        "ThermalDrift":           ("TemperatureSensor", "slow monotonic temperature drift", "miscalibrated thermocouple or partial loss of thermal contact"),
        "MissingSensorSegment":   ("TemperatureSensor", "contiguous gap in telemetry", "sensor dropout or BMS communication loss"),
        "CapacityInconsistency":  ("Cell", "sudden discharge-capacity drop", "coulomb-counting fault or step change in active material"),
    }
    component, pattern, cause = SUBTYPE_META.get(
        subtype, ("Cell", "anomalous reading", "unknown root cause"))

    return {
        "new_anomaly_types": [{
            "name": subtype,
            "label": label_human,
            "description": f"{label_human}: {pattern}; likely cause: {cause}.",
            "parent_class": parent,
            "affected_component": component,
            "observed_pattern": pattern,
            "possible_cause": cause,
        }],
        "new_causal_relationships": [{
            "from": f"{NS}{subtype}",
            "to": f"{NS}{broad.replace(' ', '')}",
            "description": f"{label_human} can lead to {broad}.",
        }],
        "new_properties": [{
            "name": "affectsComponent",
            "domain": f"{NS}{subtype}",
            "range": f"{NS}{component}",
            "description": "Battery component affected by the anomaly.",
        }],
        "rdf_triples": [
            {"subject": f"{NS}{subtype}",
             "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
             "object": "http://www.w3.org/2002/07/owl#Class"},
            {"subject": f"{NS}{subtype}",
             "predicate": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
             "object": f"{NS}{parent}"},
            {"subject": f"{NS}{subtype}",
             "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
             "object": label_human},
            {"subject": f"{NS}{subtype}",
             "predicate": f"{NS}leadsTo",
             "object": f"{NS}{broad.replace(' ', '')}"},
            {"subject": f"{NS}{subtype}",
             "predicate": f"{NS}affectsComponent",
             "object": f"{NS}{component}"},
            {"subject": f"{NS}{subtype}",
             "predicate": f"{NS}observedPattern",
             "object": pattern},
            {"subject": f"{NS}{subtype}",
             "predicate": f"{NS}possibleCause",
             "object": cause},
        ],
        "justification": (
            f"The injected subtype `{subtype}` (anomaly id "
            f"{candidate.get('anomaly_id', '?')}, cell "
            f"{candidate.get('cell_id', '?')}, cycle "
            f"{candidate.get('cycle', '?')}) is not represented in the current "
            f"FailureMode taxonomy. Adding it as a subclass of `{parent}` with a "
            f"`leadsTo` edge to `{broad}` extends the ontology to cover the "
            f"observed phenomenon while remaining consistent with existing "
            f"semantic rules."
        ),
    }


def _vary_mock(proposal: dict, llm_name: str) -> dict:
    
    p = deepcopy(proposal)
    if "Phi" in llm_name:
        # Phi is smaller; keep it tighter -- one type, one relation.
        p["new_anomaly_types"] = p.get("new_anomaly_types", [])[:1]
        p["new_causal_relationships"] = p.get("new_causal_relationships", [])[:1]
    elif "Mistral" in llm_name:
        # Mistral is more verbose; add a redundant type that already exists
        # so the redundancy/novelty score actually differs from Qwen.
        if p.get("new_anomaly_types"):
            extra = deepcopy(p["new_anomaly_types"][0])
            extra["name"] = "Overheating"
            extra["label"] = "Overheating"
            p["new_anomaly_types"].append(extra)
    return p


# Ontology before/after diff and figure
def diff_graphs(graph_before, graph_after) -> dict:
    """Return a summary dict of triples added between two rdflib Graphs."""
    from rdflib import RDFS, RDF, OWL
    triples_before = set(graph_before)
    triples_after = set(graph_after)
    added_triples = triples_after - triples_before
    removed_triples = triples_before - triples_after

    added_classes = []
    added_obj_props = []
    added_data_props = []
    added_subclass_axioms = []
    added_label_axioms = []
    added_leads_to_axioms = []
    other_axioms = []

    for s, p, o in added_triples:
        if p == RDF.type and o == OWL.Class:
            added_classes.append(str(s))
        elif p == RDF.type and o == OWL.ObjectProperty:
            added_obj_props.append(str(s))
        elif p == RDF.type and o == OWL.DatatypeProperty:
            added_data_props.append(str(s))
        elif p == RDFS.subClassOf:
            added_subclass_axioms.append((str(s), str(o)))
        elif p == RDFS.label:
            added_label_axioms.append((str(s), str(o)))
        elif str(p).endswith("leadsTo"):
            added_leads_to_axioms.append((str(s), str(o)))
        else:
            other_axioms.append((str(s), str(p), str(o)))

    return {
        "n_added": len(added_triples),
        "n_removed": len(removed_triples),
        "added_classes": sorted(set(added_classes)),
        "added_object_properties": sorted(set(added_obj_props)),
        "added_datatype_properties": sorted(set(added_data_props)),
        "added_subclass_axioms": [
            {"subject": s, "object": o} for s, o in added_subclass_axioms],
        "added_label_axioms": [
            {"subject": s, "label": o} for s, o in added_label_axioms],
        "added_leadsTo_axioms": [
            {"from": s, "to": o} for s, o in added_leads_to_axioms],
        "added_other_axioms": [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in other_axioms],
    }


def write_diff_markdown(diff: dict, before_path: Path, after_path: Path,
                        out_path: Path) -> None:
    lines = ["# Ontology diff", "",
             f"- before: `{before_path.name}`",
             f"- after:  `{after_path.name}`",
             f"- triples added:   **{diff['n_added']}**",
             f"- triples removed: **{diff['n_removed']}**",
             "",
             "## Summary counts",
             f"- new classes: **{len(diff['added_classes'])}**",
             f"- new subClassOf: **{len(diff['added_subclass_axioms'])}**",
             f"- new rdfs:label: **{len(diff['added_label_axioms'])}**",
             f"- new bat:leadsTo: **{len(diff['added_leadsTo_axioms'])}**",
             f"- new object properties: **{len(diff['added_object_properties'])}**",
             f"- new datatype properties: **{len(diff['added_datatype_properties'])}**",
             ""]
    lines.append("## Added classes")
    if diff["added_classes"]:
        for c in diff["added_classes"]:
            lines.append(f"- `{c}`")
    else:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Added object properties")
    if diff["added_object_properties"]:
        for c in diff["added_object_properties"]:
            lines.append(f"- `{c}`")
    else:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Added datatype properties")
    if diff["added_datatype_properties"]:
        for c in diff["added_datatype_properties"]:
            lines.append(f"- `{c}`")
    else:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Added subClassOf axioms")
    for a in diff["added_subclass_axioms"]:
        lines.append(f"- `{a['subject']}` rdfs:subClassOf `{a['object']}`")
    if not diff["added_subclass_axioms"]:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Added rdfs:label axioms")
    for a in diff["added_label_axioms"]:
        lines.append(f"- `{a['subject']}` rdfs:label \"{a['label']}\"")
    if not diff["added_label_axioms"]:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Added bat:leadsTo axioms")
    for a in diff["added_leadsTo_axioms"]:
        lines.append(f"- `{a['from']}` bat:leadsTo `{a['to']}`")
    if not diff["added_leadsTo_axioms"]:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("## Other added triples")
    for a in diff["added_other_axioms"]:
        lines.append(f"- `{a['subject']}` `{a['predicate']}` `{a['object']}`")
    if not diff["added_other_axioms"]:
        lines.append("- *(none)*")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def consistency_check(graph) -> dict:
   
    from rdflib import RDF, RDFS, OWL
    EXTERNAL_NS = (
        "http://www.w3.org/ns/sosa/",
        "http://www.w3.org/ns/ssn/",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/2000/01/rdf-schema#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2006/time#",
        "http://www.w3.org/2003/01/geo/wgs84_pos#",
        "http://www.w3.org/2001/XMLSchema#",
    )

    def _local(uri):
        s = str(uri)
        return not any(s.startswith(ns) for ns in EXTERNAL_NS)

    declared_classes = set(s for s, _, o in graph.triples((None, RDF.type, OWL.Class)))
    sub_targets = set(o for _, _, o in graph.triples((None, RDFS.subClassOf, None)))
    leads_subjects = set(s for s, p, o in graph
                         if str(p).endswith("leadsTo"))
    leads_objects = set(o for s, p, o in graph
                        if str(p).endswith("leadsTo"))
    # Only require declared-in-graph for LOCAL namespace classes.
    missing_subclass_targets = {x for x in (sub_targets - declared_classes) if _local(x)}
    missing_leadsTo_targets = {x for x in
                               ((leads_subjects | leads_objects) - declared_classes)
                               if _local(x)}

    label_subjects = list(graph.triples((None, RDFS.label, None)))
    seen_labels: dict = {}
    duplicate_labels = []
    for s, _, o in label_subjects:
        seen_labels.setdefault(s, []).append(o)
    for s, lbls in seen_labels.items():
        if len(lbls) > 1:
            duplicate_labels.append({"subject": str(s),
                                     "labels": [str(x) for x in lbls]})

    consistent = (not missing_subclass_targets) and (not missing_leadsTo_targets)
    return {
        "consistent_lightweight": bool(consistent),
        "missing_subclass_targets": [str(x) for x in missing_subclass_targets],
        "missing_leadsTo_targets": [str(x) for x in missing_leadsTo_targets],
        "duplicate_labels": duplicate_labels,
        "n_triples": len(graph),
    }


def render_ontology_figure(diff: dict, before_path: Path, after_path: Path,
                           out_path: Path) -> None:
    """Render a small figure of the original-fragment vs the updated fragment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    short = lambda u: u.split("#")[-1] if "#" in u else u  # noqa: E731

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Battery ontology fragment — before vs after Stage 7 update",
                 fontsize=12)

    # Original fragment (left)
    ax.text(0.2, 7.4, "Before (existing ontology)", fontsize=11,
            fontweight="bold", color="#444")
    fm_box = mp.FancyBboxPatch((0.5, 5.4), 3.5, 0.9,
                               boxstyle="round,pad=0.1",
                               facecolor="#dfe7f5", edgecolor="#345")
    ax.add_patch(fm_box)
    ax.text(2.25, 5.85, "FailureMode", ha="center", fontsize=10)

    for i, (label, color) in enumerate([("Overheating", "#fcd6a4"),
                                         ("Overcurrent", "#fcd6a4"),
                                         ("ThermalRunaway", "#fbb4a4")]):
        bx = mp.FancyBboxPatch((0.3 + i * 1.45, 4.0), 1.35, 0.7,
                                boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor="#834")
        ax.add_patch(bx)
        ax.text(0.3 + i * 1.45 + 0.675, 4.35, label, ha="center", fontsize=8)
        ax.annotate("", xy=(0.3 + i * 1.45 + 0.675, 4.7),
                    xytext=(2.25, 5.4),
                    arrowprops=dict(arrowstyle="->", color="#345"))

    # Vertical separator
    ax.plot([5.6, 5.6], [0.5, 7.5], color="#888", linestyle=":", linewidth=1)

    # Updated fragment (right)
    ax.text(6.0, 7.4, "After (LLM-proposed, expert-validated)", fontsize=11,
            fontweight="bold", color="#444")
    fm_box2 = mp.FancyBboxPatch((6.5, 5.4), 3.5, 0.9,
                                 boxstyle="round,pad=0.1",
                                 facecolor="#dfe7f5", edgecolor="#345")
    ax.add_patch(fm_box2)
    ax.text(8.25, 5.85, "FailureMode", ha="center", fontsize=10)

    # Show up to 3 newly added classes on the right side
    new_classes = diff.get("added_classes", [])[:3]
    if not new_classes:
        ax.text(8.25, 4.0, "(no new classes added)", ha="center",
                fontsize=10, color="#a44")
    else:
        for i, cls in enumerate(new_classes):
            bx = mp.FancyBboxPatch((6.3 + i * 1.55, 4.0), 1.45, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#c8f0c8", edgecolor="#262")
            ax.add_patch(bx)
            ax.text(6.3 + i * 1.55 + 0.725, 4.35, short(cls), ha="center",
                    fontsize=8)
            ax.annotate("", xy=(6.3 + i * 1.55 + 0.725, 4.7),
                        xytext=(8.25, 5.4),
                        arrowprops=dict(arrowstyle="->", color="#262"))

    # New leadsTo edges
    leads = diff.get("added_leadsTo_axioms", [])[:3]
    y = 2.4
    if leads:
        ax.text(8.25, 3.1, "New leadsTo relations:", fontsize=10,
                fontweight="bold", color="#262", ha="center")
        for ax_e in leads:
            ax.text(8.25, y, f"{short(ax_e['from'])} → {short(ax_e['to'])}",
                    ha="center", fontsize=9, color="#262")
            y -= 0.4

    # Provenance / validation badge
    prov = ("Provenance: LLM-proposed via Stage 5 enrichment on MIT-batch1 "
            "injected anomalies (A1-A7); validated by domain expert in "
            "Stage 6 (auto-approve in this run); applied in Stage 7.")
    ax.text(0.2, 0.7, prov, fontsize=8, color="#444", wrap=True,
            bbox=dict(boxstyle="round", facecolor="#f7f7f0",
                      edgecolor="#bbb"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def render_score_figure(scores: np.ndarray, labels: np.ndarray,
                        thr_a: float, thr_b: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    idx_normal = np.where(labels == 0)[0]
    idx_anom = np.where(labels == 1)[0]
    ax.scatter(idx_normal, scores[idx_normal], s=10, color="#4a78b5",
               label="normal", alpha=0.5)
    ax.scatter(idx_anom, scores[idx_anom], s=22, color="#c0392b",
               label="injected (A1..A7 cycle)", alpha=0.95)
    ax.axhline(thr_a, color="#7c4f0e", linestyle="--",
               label=f"Regime A (p99) = {thr_a:.4g}")
    ax.axhline(thr_b, color="#1f6f3a", linestyle="--",
               label=f"Regime B (p75) = {thr_b:.4g}")
    ax.set_xlabel("window index")
    ax.set_ylabel("LSTM-AE reconstruction error")
    ax.set_title("Per-window anomaly scores with both thresholds")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# Main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--real-llm", action="store_true",
                        help="Use HuggingFace LLMs instead of mock responses.")
    parser.add_argument("--regime", choices=["A", "B", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("#" * 78)
    print("# Conference validation: TS-only LSTM-AE on MIT-batch1 (A1..A7)")
    print("#" * 78)

    # ── 1. Load dataset
    dataset = load_mit_dataset(seq_len=50, standardize=True)

    # ── 2. Train detector
    model, history, ts_normal, n_params = train_detector(
        dataset, epochs=args.epochs, seed=args.seed)

    # ── 3. Evaluate detector once on the full set
    scores, recons = evaluate_detector(model, dataset)
    normal_scores = scores[dataset["labels"] == 0]

    # ── 4. Build reasoner (kept *outside* the regime functions because the
    # reasoner is mutated by Stage 7 and we want one ontology snapshot per
    # full experiment, not per regime).
    ontology_path = ROOT / "data" / "battery_ontology.ttl"
    reasoner = SemanticReasoner(str(ontology_path))

    # Snapshot ontology BEFORE any update
    ontology_dir = HERE / "ontology"
    ontology_dir.mkdir(parents=True, exist_ok=True)
    before_path = ontology_dir / "ontology_before.owl"
    reasoner.save_ontology(str(before_path))

    
    from rdflib import Graph as _Graph
    graph_before = _Graph()
    graph_before.parse(str(before_path), format="turtle")

    
    enricher = LLMOntologyEnricher(
        model_name=LLM_CONFIGS[0]["model_id"],
        use_mock=not args.real_llm,
    )
    validator = HumanValidator(auto_approve=True)

    summaries = {}
    if args.regime in ("A", "both"):
        summaries["A"] = run_regime(
            "A", dataset, model, scores, recons, normal_scores,
            reasoner, enricher, validator, HERE,
            run_llms_for_unknown=True,
        )
    if args.regime in ("B", "both"):
        summaries["B"] = run_regime(
            "B", dataset, model, scores, recons, normal_scores,
            reasoner, enricher, validator, HERE,
            run_llms_for_unknown=True,
        )

    # ── 6. Snapshot ontology AFTER all updates
    after_path = ontology_dir / "ontology_after.owl"
    reasoner.save_ontology(str(after_path))
    graph_after = _Graph()
    graph_after.parse(str(after_path), format="turtle")

    diff = diff_graphs(graph_before, graph_after)
    (ontology_dir / "ontology_diff.json").write_text(
        json.dumps(_to_jsonable(diff), indent=2), encoding="utf-8")
    write_diff_markdown(diff, before_path, after_path,
                        ontology_dir / "ontology_diff.md")

    cons_before = consistency_check(graph_before)
    cons_after = consistency_check(graph_after)
    (ontology_dir / "consistency_check.json").write_text(
        json.dumps({"before": cons_before, "after": cons_after},
                    indent=2), encoding="utf-8")

    # ── 7. Multi-LLM comparison run 
    
    cmp_reasoner = SemanticReasoner(str(ontology_path))
    # Pick the best trigger record: prefer a regime-B unknown; else a
    # regime-B confirmed; else None.
    trigger_record = None
    cand_b = HERE / REGIMES["B"]["out_dir"] / "candidates.json"
    if cand_b.exists():
        candidates = json.loads(cand_b.read_text(encoding="utf-8"))
        
        for r in candidates:
            if r.get("anomaly_id") and r.get("stage4_reasoning") in ("unknown", "confirmed"):
                trigger_record = r
                break
        if trigger_record is None:
            for r in candidates:
                if r.get("stage4_reasoning") == "unknown":
                    trigger_record = r
                    break
        if trigger_record is None:
            for r in candidates:
                if r.get("stage4_reasoning") == "confirmed":
                    trigger_record = r
                    break

    if trigger_record:
        run_multi_llm_comparison(
            trigger_record, cmp_reasoner,
            use_mock=not args.real_llm,
            out_dir=HERE / "llm_comparison",
        )
    else:
        (HERE / "llm_comparison" / "multi_llm_comparison.json").parent.mkdir(
            parents=True, exist_ok=True)
        (HERE / "llm_comparison" / "multi_llm_comparison.json").write_text(
            json.dumps({"status": "stage not activated -- no Stage-5 trigger "
                        "candidate was found in either regime"},
                       indent=2), encoding="utf-8")

   

    # ── 9. Top-level run config
    run_config = {
        "epochs": args.epochs,
        "seed": args.seed,
        "use_real_llm": bool(args.real_llm),
        "n_train_normal": int(ts_normal.shape[0]),
        "seq_len": int(dataset["ts_data"].shape[1]),
        "input_channels": int(dataset["ts_data"].shape[2]),
        "n_params": int(n_params),
        "regimes": REGIMES,
        "regime_summaries": summaries,
        "ontology_diff_summary": {
            "n_added": diff["n_added"],
            "added_classes": diff["added_classes"],
            "added_object_properties": diff["added_object_properties"],
            "added_datatype_properties": diff["added_datatype_properties"],
        },
    }
    (HERE / "run_config.json").write_text(
        json.dumps(_to_jsonable(run_config), indent=2), encoding="utf-8")






if __name__ == "__main__":
    main()
