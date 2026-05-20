"""
Module 7
"""

import os
import sys
import numpy as np
import torch
np.set_printoptions(precision=4, suppress=True)

# Local modules
from data.mit_data import load_mit_dataset
from models.autoencoders import MultimodalFusionDetector, train_autoencoders, get_anomaly_threshold
from reasoning.semantic_reasoner import SemanticReasoner
from filtering.false_positive_filter import FalsePositiveFilter
from enrichment.llm_enrichment import LLMOntologyEnricher, HumanValidator, apply_approved_updates



class BatteryAnomalyPipeline:
    """
    End-to-end pipeline for battery anomaly detection with
    semantic reasoning and ontology enrichment.
    """

    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize components
        self.reasoner = None
        self.model = None
        self.fp_filter = None
        self.enricher = None
        self.validator = None
        self.threshold = None

    def _default_config(self):
        return {
            # Data
            "n_normal": 200,
            "n_anomalous": 60,
            "seq_len": 50,
            "img_size": 64,
            # Model
            "ts_latent_dim": 32,
            "img_latent_dim": 32,
            "epochs": 20,
            "lr": 1e-3,
            "batch_size": 32,
            # Filtering
            "confidence_threshold": 0.8,
            "temporal_k": 3,
            "temporal_n": 5,
            "threshold_percentile": 95,
            # LLM
            "use_mock_llm": True,  # Set False to use real HuggingFace model
            "llm_model": "Qwen/Qwen2.5-7B-Instruct",
            # Human review
            "auto_approve": True,  # Set False for interactive review
            # Data source -- 'mit' is the only supported source for the
            # TS-only pipeline. Legacy values ('synthetic', 'real') are
            # accepted for compatibility but routed to the MIT loader.
            "data_source": "mit",
            "mit_final_dir": None,         # default: <repo>/illustrative_case_study/data/final
            # Output
            "output_dir": "output",
        }

    # Step 1: Initialize Components
    def initialize(self):
        """Set up all pipeline components."""
        print("\n" + "="*70)
        print("STEP 1: Initializing Pipeline Components")
        print("="*70)

        # Semantic Reasoner
        ontology_path = os.path.join(os.path.dirname(__file__), "data", "battery_ontology.ttl")
        self.reasoner = SemanticReasoner(ontology_path)

        # Deep Learning Model
        self.model = MultimodalFusionDetector(
            ts_latent_dim=self.config["ts_latent_dim"],
            img_latent_dim=self.config["img_latent_dim"],
        ).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # False Positive Filter
        self.fp_filter = FalsePositiveFilter(
            confidence_threshold=self.config["confidence_threshold"],
            temporal_k=self.config["temporal_k"],
            temporal_n=self.config["temporal_n"],
            reasoner=self.reasoner,
        )

        # LLM Enricher
        self.enricher = LLMOntologyEnricher(
            model_name=self.config["llm_model"],
            use_mock=self.config["use_mock_llm"],
        )

        # Human Validator
        self.validator = HumanValidator(auto_approve=self.config["auto_approve"])

        os.makedirs(self.config["output_dir"], exist_ok=True)
        print("All components initialized.")

    # Step 2:  Load Data
    def generate_data(self):
        """Load the MIT case-study dataset (TS-only)."""
        print("\n" + "="*70)
        print("STEP 2: Loading MIT case-study dataset (TS-only)")
        print("="*70)
        self.dataset = load_mit_dataset(
            final_dir=self.config.get("mit_final_dir"),
            seq_len=self.config["seq_len"],
            standardize=True,
        )
        print(f"  Time-series: {self.dataset['ts_data'].shape}")
        print(f"  Normal: {sum(self.dataset['labels']==0)}, "
              f"Anomalous: {sum(self.dataset['labels']==1)}")

    
    # Step 3: Train Autoencoders
    
    def train(self):
        """Train autoencoders on normal data only."""
        print("\n" + "="*70)
        print("STEP 3: Training Multimodal Autoencoders")
        print("="*70)

        # Train on NORMAL data only (unsupervised anomaly detection)
        normal_mask = self.dataset["labels"] == 0
        ts_normal = self.dataset["ts_data"][normal_mask]

        self.history = train_autoencoders(
            self.model, ts_normal, None,
            epochs=self.config["epochs"],
            lr=self.config["lr"],
            batch_size=self.config["batch_size"],
        )

        # Compute threshold from normal data
        self.threshold = get_anomaly_threshold(
            self.model, ts_normal, None,
            percentile=self.config["threshold_percentile"],
        )
        print(f"  Anomaly threshold (p{self.config['threshold_percentile']}): {self.threshold:.6f}")

        # Plot training loss
        plot_training_loss(self.history, os.path.join(self.config["output_dir"], "training_loss.png"))

    # Step 4: Detect Anomalies
    def detect(self):
        """Run anomaly detection on full dataset."""
        print("\n" + "="*70)
        print("STEP 4: Running Anomaly Detection")
        print("="*70)

        self.model.eval()
        with torch.no_grad():
            ts_tensor = torch.FloatTensor(self.dataset["ts_data"]).to(self.device)

            # Capture reconstructions for visualization
            ts_recon, _, _, _ = self.model(ts_tensor, None)
            self.ts_reconstructions = ts_recon.cpu().numpy()
            self.img_reconstructions = None  # image modality removed

            # Compute anomaly scores (TS-only reconstruction error)
            self.scores, self.ts_errors, self.img_errors = \
                self.model.compute_anomaly_scores(ts_tensor, None)
            self.scores = self.scores.cpu().numpy()
            self.ts_errors = self.ts_errors.cpu().numpy()
            self.img_errors = self.img_errors.cpu().numpy()

        # Basic detection metrics
        predictions = (self.scores >= self.threshold).astype(int)
        labels = self.dataset["labels"]
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # Plot scores
        plot_anomaly_scores(
            self.scores, labels, self.threshold,
            os.path.join(self.config["output_dir"], "anomaly_scores.png"),
        )
        plot_score_distribution(
            self.scores, labels, self.threshold,
            os.path.join(self.config["output_dir"], "score_distribution.png"),
        )

        return predictions

    # Step 5: False Positive Filtering
    
    @staticmethod
    def _derive_patterns_from_window(window):

        
        patterns = set()
        seq_len = window.shape[0]
        half = seq_len // 2
        temp = window[:, 0]
        cur = window[:, 1]
        volt = window[:, 2]

        # Within-window trend
        d_temp = temp[half:].mean() - temp[:half].mean()
        d_cur = cur[half:].mean() - cur[:half].mean()

        # Absolute physical thresholds (in scaled units):
        #   T_max > 0.45  =>  >45 C cell temperature
        #   T_max > 0.55  =>  >55 C cell temperature (battery-safety limit)
        #   I deviates from idle by > 0.20 in either direction => >60 A current
        #   V_max > 0.85  =>  >4.20 V terminal voltage
        #   V_min < 0.35  =>  <3.20 V terminal voltage
        T_max = float(temp.max())
        T_mean = float(temp.mean())
        I_abs_max = float(np.abs(cur - 0.5).max())  # distance from idle
        V_max = float(volt.max())
        V_min = float(volt.min())

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
        # If a strong temperature signal coincides with high current, treat
        # the current channel as confirming the heating event.
        if T_mean > 0.45 and I_abs_max > 0.10:
            patterns.add("current_increase")
        return patterns

    def filter_detections(self, predictions):
        """Apply 3-stage false positive filtering on detected anomalies."""
        print("\n" + "="*70)
        print("STEP 5: False Positive Filtering Pipeline")
        print("="*70)

        self.fp_filter.reset_temporal_window()
        filter_results = []
        detected_indices = np.where(predictions == 1)[0]

        # Map anomaly types to observation patterns (used for synthetic data)
        type_to_patterns = {
            "overheating": {"temperature_rise", "current_increase"},
            "overcurrent": {"current_spike"},
            "thermal_runaway": {"rapid_temperature_spike", "overheating"},
            "overcharge": {"voltage_spike"},
            "overdischarge": {"voltage_drop"},
            "normal": set(),
        }
        # MIT data are physically scaled to [0, 1]; derive patterns from the
        # window itself rather than from synthetic anomaly labels.
        use_window_patterns = self.config.get("data_source") in ("mit", "real")

        for idx in detected_indices:
            atype = self.dataset["anomaly_types"][idx]
            if use_window_patterns:
                patterns = self._derive_patterns_from_window(
                    self.dataset["ts_data"][idx]
                )
            else:
                patterns = type_to_patterns.get(atype, set())

            # Normalize score to [0, 1] for the filter
            normalized_score = min(self.scores[idx] / (self.threshold * 2), 1.0)
            result = self.fp_filter.filter(normalized_score, patterns)
            result["sample_idx"] = int(idx)
            result["anomaly_type"] = atype
            # Attach raw data for visualization
            result["ts_data"] = self.dataset["ts_data"][idx]
            # Image modality removed; expose explicit None for legacy callers
            result["img_data"] = None
            result["img_jpg_bytes"] = None
            # Attach reconstructions for comparison
            result["ts_recon"] = self.ts_reconstructions[idx]
            result["img_recon"] = None
            filter_results.append(result)

        # Summary
        validated = sum(1 for r in filter_results if r["validated"])
        rejected = len(filter_results) - validated
        print(f"  Total detections: {len(filter_results)}")
        print(f"  Validated: {validated}, Rejected as FP: {rejected}")

        # Plot pipeline summary
        plot_pipeline_summary(
            filter_results,
            os.path.join(self.config["output_dir"], "pipeline_summary.png"),
        )

        self.filter_results = filter_results
        return filter_results

    # Step 6: Semantic Reasoning on Validated Anomalies
    
    def reason(self, filter_results):
        """Run semantic reasoning on validated anomalies."""
        print("\n" + "="*70)
        print("STEP 6: Semantic Reasoning")
        print("="*70)

        # Run SPARQL demo queries
        self.reasoner.demo_queries()

        # For each validated anomaly, check causal pathways
        validated = [r for r in filter_results if r["validated"]]
        print(f"\n  Checking causal pathways for {len(validated)} validated anomalies...")

        for result in validated[:5]:  # Show first 5
            failure_modes = result.get("failure_modes", [])
            for fm in failure_modes:
                print(f"\n  Failure mode: {fm}")
                # Look up downstream consequences
                from reasoning.semantic_reasoner import BAT
                fm_short = fm.split("#")[-1] if "#" in fm else fm
                fm_uri = str(BAT[fm_short])
                downstream = self.reasoner.check_causal_pathway(fm_uri)
                for d in downstream:
                    print(f"    >> Could lead to: {d['label']}")

    # Step 7: LLM Enrichment + Human Review
    
    def enrich_ontology(self, filter_results):
        """
        Use LLM to propose ontology updates for truly NEW anomalies only.

        Process:
        1. Group validated anomalies by anomaly_type
        2. Check if failure modes already exist in ontology
        3. Skip types where all failure modes already exist (no LLM call)
        4. For truly new types → call LLM with semantic context
        5. Pre-filter LLM proposals to remove any duplicates
        6. Human review → apply approved updates
        7. Save enriched ontology
        """
        print("\n" + "="*70)
        print("STEP 7: LLM-Based Ontology Enrichment")
        print("="*70)

        validated = [r for r in filter_results if r["validated"]]
        if not validated:
            print("  No validated anomalies to enrich from.")
            return

        from reasoning.semantic_reasoner import BAT
        from enrichment.llm_enrichment import filter_proposals_by_ontology

        
        # 1. Collect distinct anomaly types and their resolved failure modes
        type_to_modes = {}  # anomaly_type_str -> set of (uri_str, label_str)

        for record in validated:
            atype = record.get("anomaly_type", "unknown")
            if atype not in type_to_modes:
                type_to_modes[atype] = set()

            # Prefer full URIs from matched_rules (most precise)
            matched = (record["stages"]
                       .get("semantic_cross_validation", {})
                       .get("matched_rules", []))
            for rule in matched:
                uri = rule.get("failure_uri", "")
                label = rule.get("implied_failure", "")
                if uri:
                    type_to_modes[atype].add((uri, label))

            # Also capture label-only entries from failure_modes list
            for label in record.get("failure_modes", []):
                if not any(lbl == label for _, lbl in type_to_modes[atype]):
                    type_to_modes[atype].add(("", label))

        
        # 2. Filter out anomaly types whose failure modes already exist
        new_types = {}
        for atype, modes in type_to_modes.items():
            truly_new_modes = set()
            for uri, label in modes:
                # Check by URI first (fast O(1) lookup)
                if uri and self.reasoner.failure_mode_exists_by_uri(uri):
                    print(f"  [SKIP] Failure mode already in ontology (URI): {uri}")
                    continue
                # Check by label (SPARQL ASK, case-insensitive)
                if label and self.reasoner.failure_mode_exists_by_label(label):
                    print(f"  [SKIP] Failure mode already in ontology (label): {label}")
                    continue
                truly_new_modes.add((uri, label))

            if truly_new_modes:
                new_types[atype] = truly_new_modes
            else:
                print(f"  [SKIP] All failure modes for anomaly type '{atype}' "
                      f"already exist — skipping LLM enrichment.")

        if not new_types:
            print("  All validated anomaly failure modes already exist in the ontology. "
                  "No LLM enrichment needed.")
            # Still save the ontology to maintain consistency
            enriched_path = os.path.join(self.config["output_dir"], "enriched_ontology.ttl")
            self.reasoner.save_ontology(enriched_path)
            return

        # 3. For each truly new type, call LLM + human review + apply
        all_review_results = []

        for atype, truly_new_modes in new_types.items():
            # Find a representative validated record for this type
            example = next(r for r in validated if r.get("anomaly_type") == atype)

            # Build anomaly_info
            anomaly_info = {
                "score": example["anomaly_score"],
                "patterns": list(example["stages"]
                                 .get("semantic_cross_validation", {})
                                 .get("matched_rules", [{}])),
                "failure_mode": next((lbl for _, lbl in truly_new_modes if lbl),
                                     "Unknown"),
                "observations": {"type": atype},
            }

            # Build semantic_context with both new and existing modes
            semantic_context = {
                "matched_rules": [r["label"] for r in self.reasoner.get_causal_rules()],
                "causal_chains": self.reasoner.get_causal_chains(),
                # Only NEW modes — those we're trying to add
                "failure_modes": [lbl for _, lbl in truly_new_modes if lbl],
                # Existing modes — inform LLM what NOT to propose again
                "existing_failure_modes": [lbl for _, lbl
                                          in self.reasoner.get_all_failure_modes()],
            }

            print(f"\n  Generating enrichment for new anomaly type: '{atype}'")
            print(f"    Truly new failure modes: {[lbl for _, lbl in truly_new_modes if lbl]}")

            # Generate enrichment proposal
            proposal = self.enricher.generate_enrichment(anomaly_info, semantic_context)

            # Pre-filter: remove items from LLM output that already exist
            filtered_proposal, pre_filtered = filter_proposals_by_ontology(proposal,
                                                                            self.reasoner)
            if pre_filtered["types"] or pre_filtered["relationships"]:
                print(f"  Pre-filtered {len(pre_filtered['types'])} types and "
                      f"{len(pre_filtered['relationships'])} relations already in ontology.")

            # Display filtered proposal summary
            print("\n  Filtered proposal updates:")
            for key, items in filtered_proposal.items():
                if isinstance(items, list) and items:
                    print(f"    {key}: {len(items)} items")

            # Human-in-the-loop review
            print("\n" + "="*70)
            print(f"STEP 7b: Human Review — Anomaly Type '{atype}'")
            print("="*70)

            review_result = self.validator.review_proposal(filtered_proposal,
                                                            interactive=False)
            all_review_results.append(review_result)

            # Apply accepted updates
            n_applied = apply_approved_updates(self.reasoner, review_result)

        # 4. Save final enriched ontology after all types are processed
        enriched_path = os.path.join(self.config["output_dir"], "enriched_ontology.ttl")
        self.reasoner.save_ontology(enriched_path)

        return all_review_results[-1] if all_review_results else None

    # Full Pipeline Execution
    
    def run(self):
        """Execute the complete pipeline end-to-end."""
        print("\n" + "#"*70)
        print("#  BATTERY ANOMALY DETECTION — FULL PIPELINE")
        print("#"*70)

        self.initialize()
        self.generate_data()
        self.train()
        predictions = self.detect()
        filter_results = self.filter_detections(predictions)
        self.reason(filter_results)
        self.enrich_ontology(filter_results)

        print("\n" + "#"*70)
        print("#  PIPELINE COMPLETE")
        print(f"#  Output saved to: {os.path.abspath(self.config['output_dir'])}")
        print("#"*70)

        return {
            "scores": self.scores,
            "predictions": predictions,
            "filter_results": filter_results,
            "threshold": self.threshold,
        }
