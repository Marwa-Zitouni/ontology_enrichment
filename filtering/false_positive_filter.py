"""
Module 4: False Positive Filtering Pipeline.


"""

import numpy as np
from collections import deque


class FalsePositiveFilter:
    

    def __init__(self, confidence_threshold=0.95, temporal_k=3, temporal_n=5, reasoner=None):
        self.confidence_threshold = confidence_threshold
        self.temporal_k = temporal_k
        self.temporal_n = temporal_n
        self.reasoner = reasoner

        # Sliding window of recent anomaly flags (True/False)
        self.temporal_window = deque(maxlen=temporal_n)

    def reset_temporal_window(self):
        """Clear the temporal consistency window."""
        self.temporal_window.clear()

    # Stage 1: Confidence Gate
    def confidence_gate(self, anomaly_score):
        
        passed = anomaly_score >= self.confidence_threshold
        reason = (
            f"Score {anomaly_score:.4f} >= {self.confidence_threshold}"
            if passed
            else f"Score {anomaly_score:.4f} < {self.confidence_threshold} — rejected"
        )
        return passed, reason

    
    # Stage 2: Temporal Consistency
    def temporal_consistency(self, is_anomaly_in_window):
        
        self.temporal_window.append(is_anomaly_in_window)
        count = sum(self.temporal_window)
        passed = count >= self.temporal_k

        reason = (
            f"Anomaly in {count}/{len(self.temporal_window)} windows "
            f"(need {self.temporal_k}/{self.temporal_n})"
        )
        return passed, reason, count

    # Stage 3: Semantic Cross-Validation
    def semantic_cross_validation(self, detected_patterns):
        
        if self.reasoner is None:
            return True, "No reasoner — skipping semantic validation", []

        matched_rules = self.reasoner.validate_anomaly_semantically(detected_patterns)

        if matched_rules:
            rule_names = [r["rule"] for r in matched_rules]
            failures = [r["implied_failure"] for r in matched_rules]
            return (
                True,
                f"Semantically validated — rules: {rule_names}, failures: {failures}",
                matched_rules,
            )
        else:
            return (
                False,
                f"No semantic support for patterns {detected_patterns} — likely false positive",
                [],
            )

    # Full Pipeline
    def filter(self, anomaly_score, detected_patterns=None):
        result = {
            "validated": False,
            "stages": {},
            "failure_modes": [],
            "anomaly_score": anomaly_score,
        }

        # Stage 1: Confidence Gate
        s1_passed, s1_reason = self.confidence_gate(anomaly_score)
        result["stages"]["confidence_gate"] = {"passed": s1_passed, "reason": s1_reason}

        if not s1_passed:
            self.temporal_window.append(False)
            return result

        # Stage 2: Temporal Consistency
        s2_passed, s2_reason, s2_count = self.temporal_consistency(True)
        result["stages"]["temporal_consistency"] = {
            "passed": s2_passed, "reason": s2_reason, "count": s2_count
        }

        if not s2_passed:
            return result

        # Stage 3: Semantic Cross-Validation
        if detected_patterns is None:
            detected_patterns = set()

        s3_passed, s3_reason, matched_rules = self.semantic_cross_validation(detected_patterns)
        result["stages"]["semantic_cross_validation"] = {
            "passed": s3_passed, "reason": s3_reason, "matched_rules": matched_rules
        }

        if s3_passed and matched_rules:
            result["failure_modes"] = [r["implied_failure"] for r in matched_rules]

        result["validated"] = s1_passed and s2_passed and s3_passed
        return result

    def filter_batch(self, anomaly_scores, patterns_list=None):
        """
        Filter a batch of detections. Resets temporal window at start.

        Args:
            anomaly_scores: array of scores
            patterns_list: list of pattern sets (one per sample), or None

        Returns: list of filter results
        """
        self.reset_temporal_window()
        results = []

        for i, score in enumerate(anomaly_scores):
            patterns = patterns_list[i] if patterns_list else None
            result = self.filter(score, patterns)
            results.append(result)

        return results


def demonstrate_filter(reasoner=None):
    """Show example of the filtering pipeline in action."""
    fp_filter = FalsePositiveFilter(
        confidence_threshold=0.8,
        temporal_k=3,
        temporal_n=5,
        reasoner=reasoner,
    )

    print("\n" + "="*70)
    print("FALSE POSITIVE FILTERING PIPELINE DEMO")
    print("="*70)

    # Simulate a stream of detections
    test_cases = [
        (0.5, set(), "Low score — should be rejected at gate"),
        (0.9, {"temperature_rise", "current_increase"}, "High score, valid patterns"),
        (0.95, {"temperature_rise", "current_increase"}, "High score, valid patterns"),
        (0.85, {"temperature_rise", "current_increase"}, "High score, valid patterns"),
        (0.92, {"voltage_spike"}, "High score, voltage pattern"),
        (0.88, set(), "High score but no semantic support"),
    ]

    for i, (score, patterns, description) in enumerate(test_cases):
        print(f"\n--- Window {i+1}: {description} ---")
        result = fp_filter.filter(score, patterns)

        for stage_name, stage_info in result["stages"].items():
            status = "✓" if stage_info["passed"] else "✗"
            print(f"  {status} {stage_name}: {stage_info['reason']}")

        verdict = "VALIDATED" if result["validated"] else "REJECTED"
        print(f"  → Result: {verdict}")
        if result["failure_modes"]:
            print(f"  → Failure modes: {result['failure_modes']}")

    print("="*70)


if __name__ == "__main__":
    # Run standalone demo (without semantic reasoner)
    demonstrate_filter()
