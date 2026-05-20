
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from enrichment.llm_enrichment import (  # noqa: E402
    LLMOntologyEnricher,
    apply_approved_updates,
    build_fallback_proposal,
    filter_proposals_by_ontology,
    normalize_proposal_schema,
    robust_parse_json,
)
from reasoning.semantic_reasoner import BAT, SemanticReasoner  # noqa: E402
from rdflib import RDF, RDFS, OWL, Literal, URIRef  # noqa: E402


def _build_fixture_reasoner() -> SemanticReasoner:
    
    r = SemanticReasoner.__new__(SemanticReasoner)
    from rdflib import Graph
    r.graph = Graph()
    r.graph.bind("bat", BAT)

    # Declare FailureMode as a class
    r.graph.add((BAT.FailureMode, RDF.type, OWL.Class))
    r.graph.add((BAT.FailureMode, RDFS.label, Literal("FailureMode")))

    for name, label in [
        ("Overdischarge", "Overdischarge"),
        ("Overheating", "Overheating"),
        ("ThermalRunaway", "Thermal Runaway"),
    ]:
        uri = BAT[name]
        r.graph.add((uri, RDF.type, OWL.Class))
        r.graph.add((uri, RDFS.subClassOf, BAT.FailureMode))
        r.graph.add((uri, RDFS.label, Literal(label)))
    return r


# 1. JSON parsing 
class TestRobustParser(unittest.TestCase):
    def test_strips_markdown_fences(self):
        text = '```json\n{"k": 1}\n```'
        self.assertEqual(robust_parse_json(text), {"k": 1})

    def test_strips_line_comments(self):
        text = '''
        {
            "rdf_triples": [
                {
                    "subject": "A2",
                    "predicate": "hasTemperature",
                    "object": 35 // example temperature value
                }
            ]
        }
        '''
        parsed = robust_parse_json(text)
        self.assertIn("rdf_triples", parsed)
        self.assertEqual(parsed["rdf_triples"][0]["object"], 35)

    def test_strips_block_comments(self):
        text = '{"a": 1, /* explanation */ "b": 2}'
        parsed = robust_parse_json(text)
        self.assertEqual(parsed, {"a": 1, "b": 2})

    def test_strips_trailing_commas(self):
        text = '{"a": 1, "b": [1, 2,],}'
        self.assertEqual(robust_parse_json(text), {"a": 1, "b": [1, 2]})

    def test_extracts_first_object_when_prose_follows(self):
        text = 'Here is the proposal: {"k": "v"}\n\nThanks!'
        self.assertEqual(robust_parse_json(text), {"k": "v"})

    def test_returns_parse_error_on_hard_failure(self):
        text = "definitely not json at all"
        parsed = robust_parse_json(text)
        self.assertIn("_parse_error", parsed)


# 2. Duplicate filtering

class TestFilterProposalsByOntology(unittest.TestCase):
    def setUp(self):
        self.reasoner = _build_fixture_reasoner()

    def test_drops_existing_broad_class_keeps_missing_subtype(self):
        proposal = {
            "new_anomaly_types": [
                {"name": "Overdischarge", "label": "Overdischarge",
                 "parent_class": "FailureMode"},
                {"name": "AbnormalTemperatureRise",
                 "label": "Abnormal Temperature Rise",
                 "parent_class": "FailureMode"},
            ],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [],
        }
        filtered, removed = filter_proposals_by_ontology(proposal, self.reasoner)
        kept_names = [t["name"] for t in filtered["new_anomaly_types"]]
        self.assertNotIn("Overdischarge", kept_names)
        self.assertIn("AbnormalTemperatureRise", kept_names)
        self.assertIn("Overdischarge", removed["types"])

    def test_drops_triples_for_dropped_class(self):
        proposal = {
            "new_anomaly_types": [
                {"name": "Overdischarge", "label": "Overdischarge",
                 "parent_class": "FailureMode"},
            ],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [
                {"subject": "http://example.org/battery-ontology#Overdischarge",
                 "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
                 "object": "Overdischarge"},
                {"subject": "http://example.org/battery-ontology#Overdischarge",
                 "predicate": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                 "object": "http://example.org/battery-ontology#FailureMode"},
            ],
        }
        filtered, _ = filter_proposals_by_ontology(proposal, self.reasoner)
        self.assertEqual(filtered["rdf_triples"], [])



# 3. Fallback proposal
class TestFallbackProposal(unittest.TestCase):
    def test_fallback_proposes_missing_subtype(self):
        fb = build_fallback_proposal(
            missing_subtype="AbnormalTemperatureRise",
            broad_failure_mode="Overdischarge",
        )
        names = [t["name"] for t in fb["new_anomaly_types"]]
        self.assertIn("AbnormalTemperatureRise", names)
        # leadsTo edge present
        froms = [r["from"].split("#")[-1] for r in fb["new_causal_relationships"]]
        tos = [r["to"].split("#")[-1] for r in fb["new_causal_relationships"]]
        self.assertIn("AbnormalTemperatureRise", froms)
        self.assertIn("Overdischarge", tos)
        # Required triples present
        triples = fb["rdf_triples"]
        preds = {t["predicate"].split("#")[-1] for t in triples}
        self.assertIn("type", preds)        # rdf:type owl:Class
        self.assertIn("subClassOf", preds)  # rdfs:subClassOf
        self.assertIn("label", preds)       # rdfs:label
        self.assertIn("leadsTo", preds)     # bat:leadsTo

    def test_fallback_survives_filter_against_partial_ontology(self):
        reasoner = _build_fixture_reasoner()  # has Overdischarge
        fb = build_fallback_proposal(
            missing_subtype="AbnormalTemperatureRise",
            broad_failure_mode="Overdischarge",
        )
        filtered, _ = filter_proposals_by_ontology(fb, reasoner)
        # The new subtype must survive
        self.assertTrue(any(
            t["name"] == "AbnormalTemperatureRise"
            for t in filtered["new_anomaly_types"]
        ))


# 4. Parser activates fallback on real-LLM-style malformed JSON
class TestEnricherFallbackPath(unittest.TestCase):
    def test_parse_error_flag_propagates(self):
        enricher = LLMOntologyEnricher(use_mock=True)
        # Bypass the mock and test the parser directly
        bad = "this is not json"
        result = enricher._parse_llm_response(bad)
        self.assertIn("_parse_error", result)

    def test_real_llm_a2_response_normalizes_overdischarge_only(self):
        """The real-LLM A2 response (Markdown-fenced JSON with // comments
        proposing the broad Overdischarge class) must parse, normalize, and
        when filtered against an ontology that already contains Overdischarge,
        produce an empty types list — which triggers the fallback path."""
        a2_response = '''```json
{
    "new_anomaly_types": [
        {
            "name": "Overdischarge",
            "label": "Overdischarge",
            "description": "Failure mode indicating insufficient charging.",
            "parent_class": "FailureMode"
        }
    ],
    "new_causal_relationships": [],
    "new_properties": [],
    "rdf_triples": [
        {
            "subject": "http://example.org/battery-ontology#A2",
            "predicate": "http://example.org/battery-ontology#hasTemperature",
            "object": 35 // Example temperature value from observation
        }
    ]
}
```'''
        enricher = LLMOntologyEnricher(use_mock=True)
        parsed = enricher._parse_llm_response(a2_response)
        self.assertIsNone(parsed.get("_parse_error"))
        self.assertEqual(len(parsed["new_anomaly_types"]), 1)
        self.assertEqual(parsed["new_anomaly_types"][0]["name"], "Overdischarge")

        reasoner = _build_fixture_reasoner()
        filtered, removed = filter_proposals_by_ontology(parsed, reasoner)
        # All proposed types are duplicates -> filtered to empty
        self.assertEqual(len(filtered["new_anomaly_types"]), 0)
        self.assertIn("Overdischarge", removed["types"])


# 5. apply_approved_updates serializes types AND relationships
class TestApplyApprovedUpdates(unittest.TestCase):
    def test_serializes_class_subclass_label_and_leadsto(self):
        reasoner = _build_fixture_reasoner()
        review_result = {
            "accepted_types": [{
                "name": "AbnormalTemperatureRise",
                "label": "Abnormal Temperature Rise",
                "description": "Fine-grained thermal anomaly subtype.",
                "parent_class": "FailureMode",
            }],
            "accepted_relationships": [{
                "from": "http://example.org/battery-ontology#AbnormalTemperatureRise",
                "to": "http://example.org/battery-ontology#Overdischarge",
                "description": "leadsTo",
            }],
            "accepted_triples": [],
        }
        n_added = apply_approved_updates(reasoner, review_result)
        # Expect: type owl:Class + subClassOf + label + comment + leadsTo = 5
        self.assertGreaterEqual(n_added, 4)

        atr = BAT.AbnormalTemperatureRise
        self.assertIn((atr, RDF.type, OWL.Class), reasoner.graph)
        self.assertIn((atr, RDFS.subClassOf, BAT.FailureMode), reasoner.graph)
        self.assertIn((atr, RDFS.label, Literal("Abnormal Temperature Rise")),
                      reasoner.graph)
        self.assertIn((atr, BAT.leadsTo, BAT.Overdischarge), reasoner.graph)

    def test_relationship_only_input_still_serialized(self):
        """Regression: previous bug where accepted_relationships were not
        serialized at all (only accepted_triples)."""
        reasoner = _build_fixture_reasoner()
        review_result = {
            "accepted_types": [],
            "accepted_relationships": [{
                "from": "http://example.org/battery-ontology#Overheating",
                "to": "http://example.org/battery-ontology#ThermalRunaway",
            }],
            "accepted_triples": [],
        }
        n_added = apply_approved_updates(reasoner, review_result)
        self.assertEqual(n_added, 1)
        self.assertIn((BAT.Overheating, BAT.leadsTo, BAT.ThermalRunaway),
                      reasoner.graph)


# 6. End-to-end: A2 fallback flow yields the expected ontology delta
class TestEndToEndStage5Fallback(unittest.TestCase):
    def test_a2_fallback_writes_full_minimum_update(self):
        reasoner = _build_fixture_reasoner()
        before = len(reasoner.graph)

        fb = build_fallback_proposal(
            missing_subtype="AbnormalTemperatureRise",
            broad_failure_mode="Overdischarge",
        )
        filtered, _ = filter_proposals_by_ontology(fb, reasoner)
        # Auto-approve everything
        review = {
            "accepted_types": filtered["new_anomaly_types"],
            "accepted_relationships": filtered["new_causal_relationships"],
            "accepted_triples": filtered["rdf_triples"],
        }
        n_added = apply_approved_updates(reasoner, review)
        after = len(reasoner.graph)

        # Class declaration + subClassOf + label + comment + leadsTo
        self.assertGreaterEqual(n_added, 4)
        self.assertGreaterEqual(after - before, 4)

        atr = BAT.AbnormalTemperatureRise
        self.assertIn((atr, RDF.type, OWL.Class), reasoner.graph)
        self.assertIn((atr, RDFS.subClassOf, BAT.FailureMode), reasoner.graph)
        self.assertIn((atr, RDFS.label,
                       Literal("Abnormal Temperature Rise")), reasoner.graph)
        self.assertIn((atr, BAT.leadsTo, BAT.Overdischarge), reasoner.graph)


if __name__ == "__main__":
    unittest.main(verbosity=2)
