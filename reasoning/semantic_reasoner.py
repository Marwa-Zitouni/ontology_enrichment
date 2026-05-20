"""
Module 3: Semantic Reasoning Engine.

Uses RDFLib to load the battery ontology and execute SPARQL queries
for causal relationship validation and failure pattern matching.
"""

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from datetime import datetime
import os

# Namespace definitions
BAT = Namespace("http://example.org/battery-ontology#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
SSN = Namespace("http://www.w3.org/ns/ssn/")
TIME = Namespace("http://www.w3.org/2006/time#")
GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")


class SemanticReasoner:
    """Stream reasoning component that queries the battery ontology."""

    def __init__(self, ontology_path=None):
        self.graph = Graph()

        # Bind common prefixes
        self.graph.bind("bat", BAT)
        self.graph.bind("sosa", SOSA)
        self.graph.bind("ssn", SSN)
        self.graph.bind("time", TIME)
        self.graph.bind("geo", GEO)
        self.graph.bind("owl", OWL)

        # Load ontology
        if ontology_path is None:
            ontology_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "battery_ontology.ttl"
            )
        if os.path.exists(ontology_path):
            self.graph.parse(ontology_path, format="turtle")
            print(f"Loaded ontology: {len(self.graph)} triples")
        else:
            print(f"WARNING: Ontology file not found at {ontology_path}")

    # SPARQL Queries for Causal Relationship Discovery

    def get_all_failure_modes(self):
        """Query all known failure modes from the ontology."""
        query = """
        SELECT ?mode ?label WHERE {
            ?mode rdfs:subClassOf bat:FailureMode .
            OPTIONAL { ?mode rdfs:label ?label . }
        }
        """
        results = self.graph.query(query, initNs={"bat": BAT})
        return [(str(row.mode), str(row.label)) for row in results]

    def get_causal_chains(self):
        """Query all causal chains (leadsTo relationships) between failure modes."""
        query = """
        SELECT ?from ?fromLabel ?to ?toLabel WHERE {
            ?from bat:leadsTo ?to .
            OPTIONAL { ?from rdfs:label ?fromLabel . }
            OPTIONAL { ?to rdfs:label ?toLabel . }
        }
        """
        results = self.graph.query(query, initNs={"bat": BAT})
        return [
            {
                "from": str(row[0]),
                "from_label": str(row[1]) if row[1] else "",
                "to": str(row[2]),
                "to_label": str(row[3]) if row[3] else "",
            }
            for row in results
        ]

    # Ontology Existence Checking — Detect Duplicates Before Enrichment
    

    def failure_mode_exists_by_uri(self, uri: str) -> bool:
        """
        Check if a given URI is already declared as rdfs:subClassOf bat:FailureMode
        in the ontology graph.

        Uses RDFLib's O(1) triple lookup: (uri, rdfs:subClassOf, bat:FailureMode) in graph

        Args:
            uri: Full URI string, e.g. "http://example.org/battery-ontology#Overheating"

        Returns:
            True if the URI is a known FailureMode, False otherwise
        """
        try:
            return (URIRef(uri), RDFS.subClassOf, BAT.FailureMode) in self.graph
        except Exception:
            return False

    def failure_mode_exists_by_label(self, label: str) -> bool:
        """
        Case-insensitive SPARQL ASK query to check if any rdfs:subClassOf bat:FailureMode
        node carries an rdfs:label that matches the provided label.

        Args:
            label: Label string, e.g. "Overheating"

        Returns:
            True if a matching FailureMode label exists, False otherwise
        """
        query = """
        ASK {
            ?mode rdfs:subClassOf bat:FailureMode .
            ?mode rdfs:label ?lbl .
            FILTER(LCASE(STR(?lbl)) = LCASE(STR(?target)))
        }
        """
        try:
            result = self.graph.query(
                query,
                initNs={"bat": BAT},
                initBindings={"target": Literal(label)}
            )
            return bool(result.askAnswer)
        except Exception:
            return False

    def causal_relation_exists(self, from_uri: str, to_uri: str) -> bool:
        """
        Check if a bat:leadsTo relationship between two failure modes already exists.

        Uses RDFLib's O(1) triple lookup: (from_uri, bat:leadsTo, to_uri) in graph

        Args:
            from_uri: Source URI, e.g. "http://example.org/battery-ontology#Overheating"
            to_uri: Target URI, e.g. "http://example.org/battery-ontology#ThermalRunaway"

        Returns:
            True if the causal relationship already exists, False otherwise
        """
        try:
            return (URIRef(from_uri), BAT.leadsTo, URIRef(to_uri)) in self.graph
        except Exception:
            return False

    def filter_new_triples(self, triples: list) -> tuple:
        """
        Pre-filter a list of (subject, predicate, object) triples before insertion.

        Separates the input list into:
        - new_triples: triples that do NOT already exist in the graph
        - duplicate_triples: triples that already exist (and would be silently ignored)

        Uses the same coercion logic as add_new_triples() to ensure consistency.

        Args:
            triples: List of (subject, predicate, object) tuples.
                     Each element may be a full URI, prefixed term, or plain label.

        Returns:
            (new_triples, duplicate_triples): Two lists of triples
        """
        new_triples = []
        duplicate_triples = []

        for s, p, o in triples:
            try:
                # Apply same coercion logic as add_new_triples()
                s_node = self._coerce_uri(s)
                p_node = self._coerce_uri(p)
                o_node = self._coerce_object(o)

                # Skip invalid coercions (None values)
                if s_node is None or p_node is None or o_node is None:
                    new_triples.append((s, p, o))  # Let add_new_triples() handle the skip
                    continue

                # Check if triple already exists in the graph
                if (s_node, p_node, o_node) in self.graph:
                    duplicate_triples.append((s, p, o))
                else:
                    new_triples.append((s, p, o))
            except Exception:
                # On coercion error, treat as new and let add_new_triples() report it
                new_triples.append((s, p, o))

        return new_triples, duplicate_triples

    def get_causal_rules(self):
        """Query all causal rules mapping observation patterns to failure modes."""
        query = """
        SELECT ?rule ?label ?condition ?failure ?failureLabel WHERE {
            ?rule a bat:CausalRule .
            ?rule bat:hasCondition ?condition .
            ?rule bat:impliesFailure ?failure .
            OPTIONAL { ?rule rdfs:label ?label . }
            OPTIONAL { ?failure rdfs:label ?failureLabel . }
        }
        """
        results = self.graph.query(query, initNs={"bat": BAT})
        return [
            {
                "rule": str(row[0]),
                "label": str(row[1]) if row[1] else "",
                "condition": str(row[2]),
                "failure": str(row[3]),
                "failure_label": str(row[4]) if row[4] else "",
            }
            for row in results
        ]

    def get_observations_for_cell(self, cell_uri):
        """Query all observations for a specific cell."""
        query = """
        SELECT ?obs ?type ?sensor ?time ?value WHERE {
            ?obs sosa:hasFeatureOfInterest ?cell .
            ?obs a ?type .
            ?obs sosa:madeBySensor ?sensor .
            ?obs sosa:resultTime ?time .
            ?obs sosa:hasSimpleResult ?value .
            FILTER(?type != sosa:Observation)
        }
        ORDER BY ?time
        """
        results = self.graph.query(
            query,
            initNs={"bat": BAT, "sosa": SOSA},
            initBindings={"cell": URIRef(cell_uri)},
        )
        return [
            {
                "observation": str(row[0]),
                "type": str(row[1]),
                "sensor": str(row[2]),
                "time": str(row[3]),
                "value": float(row[4]),
            }
            for row in results
        ]

    # Anomaly Validation via Semantic Cross-Validation

    def classify_observation_patterns(self, observations):
        """
        Given a list of observation dicts, detect patterns like:
        - temperature_rise, current_increase, voltage_spike, etc.

        Returns: set of detected pattern strings
        """
        patterns = set()

        # Group observations by type
        temps, currents, voltages = [], [], []
        for obs in observations:
            obs_type = obs.get("type", "")
            value = obs.get("value", 0.0)
            if "Temperature" in obs_type:
                temps.append(value)
            elif "Current" in obs_type:
                currents.append(value)
            elif "Voltage" in obs_type:
                voltages.append(value)

        # Detect patterns based on thresholds
        if len(temps) >= 2 and (temps[-1] - temps[0]) > 15:
            patterns.add("temperature_rise")
        if len(temps) >= 2 and (temps[-1] - temps[0]) > 40:
            patterns.add("rapid_temperature_spike")
        if len(currents) >= 2 and (currents[-1] - currents[0]) > 3:
            patterns.add("current_increase")
        if len(currents) >= 2 and max(currents) > 7:
            patterns.add("current_spike")
        if len(voltages) >= 2 and max(voltages) > 4.2:
            patterns.add("voltage_spike")
        if len(voltages) >= 2 and min(voltages) < 2.5:
            patterns.add("voltage_drop")

        return patterns

    def validate_anomaly_semantically(self, detected_patterns):
        """
        Cross-validate detected anomaly patterns against ontology causal rules.

        Args:
            detected_patterns: set of pattern strings (e.g., {"temperature_rise", "current_increase"})

        Returns:
            list of matching causal rules with their implied failure modes
        """
        rules = self.get_causal_rules()
        matches = []

        for rule in rules:
            # Parse condition (e.g., "temperature_rise AND current_increase")
            condition_str = rule["condition"]
            conditions = [c.strip() for c in condition_str.split("AND")]

            # Check if all conditions are met
            if all(cond in detected_patterns for cond in conditions):
                matches.append({
                    "rule": rule["label"],
                    "matched_conditions": conditions,
                    "implied_failure": rule["failure_label"],
                    "failure_uri": rule["failure"],
                })

        return matches

    def check_causal_pathway(self, failure_mode_uri):
        """
        Check if a failure mode has known downstream consequences.
        E.g., does Overheating lead to Thermal Runaway?
        """
        query = """
        SELECT ?downstream ?label WHERE {
            ?start bat:leadsTo+ ?downstream .
            OPTIONAL { ?downstream rdfs:label ?label . }
        }
        """
        results = self.graph.query(
            query,
            initNs={"bat": BAT},
            initBindings={"start": URIRef(failure_mode_uri)},
        )
        return [
            {"uri": str(row[0]), "label": str(row[1]) if row[1] else ""}
            for row in results
        ]

    # Ontology Enrichment: Add new triples

    def add_anomaly_instance(self, anomaly_id, anomaly_score, failure_mode_uri=None,
                             timestamp=None, validated=True):
        """Add a detected anomaly instance to the ontology."""
        anomaly_uri = BAT[f"anomaly_{anomaly_id}"]
        anomaly_class = BAT.ValidatedAnomaly if validated else BAT.RejectedAnomaly

        self.graph.add((anomaly_uri, RDF.type, anomaly_class))
        self.graph.add((anomaly_uri, BAT.anomalyScore, Literal(anomaly_score, datatype=XSD.float)))

        if failure_mode_uri:
            self.graph.add((anomaly_uri, BAT.hasCause, URIRef(failure_mode_uri)))

        if timestamp:
            self.graph.add((anomaly_uri, SOSA.resultTime,
                          Literal(timestamp, datatype=XSD.dateTime)))

        return anomaly_uri

    def add_new_triples(self, triples):
        """Add a list of (subject, predicate, object) triples to the graph.

        Each element may be a full URI ("http://..."), a prefixed term
        ("bat:Overheating"), or a plain label from an LLM ("Thermal Runaway").
        Plain labels are normalized into the BAT namespace so the resulting
        graph remains serializable.
        """
        added = 0
        skipped = 0
        for s, p, o in triples:
            try:
                s_node = self._coerce_uri(s)
                p_node = self._coerce_uri(p)
                o_node = self._coerce_object(o)
                if s_node is None or p_node is None or o_node is None:
                    skipped += 1
                    continue
                self.graph.add((s_node, p_node, o_node))
                added += 1
            except Exception as e:
                print(f"  skipping invalid triple {(s, p, o)}: {e}")
                skipped += 1
        msg = f"Added {added} new triples"
        if skipped:
            msg += f" (skipped {skipped} invalid)"
        print(f"{msg}. Total: {len(self.graph)}")

    @staticmethod
    def _slugify(term):
        """CamelCase a free-form label into a URI-safe local name."""
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in term)
        return "".join(w for w in cleaned.title().split())

    @classmethod
    def _coerce_uri(cls, value):
        """Turn an arbitrary string into a valid URIRef within the BAT namespace."""
        if value is None:
            return None
        if isinstance(value, URIRef):
            # Re-validate: existing URIRefs may still contain invalid chars
            text = str(value)
        else:
            text = str(value).strip()
        if not text:
            return None
        # Full URI => slugify only the fragment/path tail if it has whitespace
        if text.startswith("http://") or text.startswith("https://"):
            if "#" in text:
                base, frag = text.rsplit("#", 1)
                if any(c.isspace() for c in frag) or not frag:
                    frag = cls._slugify(frag)
                    if not frag:
                        return None
                    return URIRef(f"{base}#{frag}")
                return URIRef(text)
            # path-style: slugify last segment if it has whitespace
            if any(c.isspace() for c in text):
                head, tail = text.rsplit("/", 1)
                tail = cls._slugify(tail)
                if not tail:
                    return None
                return URIRef(f"{head}/{tail}")
            return URIRef(text)
        if text.startswith("bat:"):
            local = cls._slugify(text.split(":", 1)[1])
            return BAT[local] if local else None
        # Plain term — slugify into CamelCase and drop into BAT namespace
        slug = cls._slugify(text)
        if not slug:
            return None
        return BAT[slug]

    @classmethod
    def _coerce_object(cls, value):
        """Objects can be URIs or literals. Prefer URI when it clearly is one."""
        if value is None:
            return None
        if isinstance(value, (URIRef, Literal)):
            return value
        s = str(value).strip()
        if not s:
            return Literal("")
        if s.startswith("http://") or s.startswith("https://") or s.startswith("bat:"):
            return cls._coerce_uri(s)
        return Literal(s)

    def save_ontology(self, output_path):
        """Serialize the graph to a Turtle file."""
        self.graph.serialize(destination=output_path, format="turtle")
        print(f"Ontology saved to {output_path} ({len(self.graph)} triples)")

    # Example SPARQL Demonstrations
    

    def demo_queries(self):
        """Run and display example SPARQL queries."""
        print("\n" + "="*70)
        print("SPARQL QUERY DEMONSTRATIONS")
        print("="*70)

        # 1. All failure modes
        print("\n--- Query 1: All Known Failure Modes ---")
        for uri, label in self.get_all_failure_modes():
            print(f"  {label}: {uri}")

        # 2. Causal chains
        print("\n--- Query 2: Causal Chains (leadsTo) ---")
        for chain in self.get_causal_chains():
            print(f"  {chain['from_label']} => {chain['to_label']}")

        # 3. Causal rules
        print("\n--- Query 3: Causal Rules ---")
        for rule in self.get_causal_rules():
            print(f"  IF [{rule['condition']}] => {rule['failure_label']}")

        # 4. Observations for cell_001
        print("\n--- Query 4: Observations for Cell 001 ---")
        cell_uri = str(BAT.cell_001)
        obs = self.get_observations_for_cell(cell_uri)
        for o in obs:
            print(f"  [{o['time']}] {o['type'].split('#')[-1]}: {o['value']}")

        # 5. Cross-validation example
        print("\n--- Query 5: Semantic Cross-Validation ---")
        patterns = {"temperature_rise", "current_increase"}
        matches = self.validate_anomaly_semantically(patterns)
        print(f"  Detected patterns: {patterns}")
        for m in matches:
            print(f"  * Rule matched: {m['rule']} => {m['implied_failure']}")

        print("="*70)


if __name__ == "__main__":
    reasoner = SemanticReasoner()
    reasoner.demo_queries()
