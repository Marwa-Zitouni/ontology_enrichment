"""
Module 5: LLM-Based Ontology Enrichment.

"""

import json
import re
from typing import Optional


NS = "http://example.org/battery-ontology#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
RDFS_SUBCLASSOF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


# Available models for enrichment
ENRICHMENT_MODELS = [
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    # Lightweight models
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
]


# Stage-5 deterministic fallback metadata

FALLBACK_SUBTYPE_LABELS = {
    "AbnormalTemperatureRise": "Abnormal Temperature Rise",
    "SensorNoiseBurst":        "Sensor Noise Burst",
    "MissingSensorSegment":    "Missing Sensor Segment",
    "SuddenVoltageDrop":       "Sudden Voltage Drop",
    "CurrentSpike":            "Current Spike",
    "ThermalDrift":            "Thermal Drift",
    "CapacityInconsistency":   "Capacity Inconsistency",
}

FALLBACK_SUBTYPE_COMMENTS = {
    "AbnormalTemperatureRise":
        "A fine-grained anomaly subtype characterized by an abnormal "
        "increase in cell temperature beyond expected operating bounds.",
    "SensorNoiseBurst":
        "A fine-grained anomaly subtype characterized by a transient burst "
        "of high-frequency noise on a sensor channel, typically indicating "
        "a loose harness contact or instrumentation glitch.",
    "MissingSensorSegment":
        "A fine-grained anomaly subtype characterized by a contiguous gap "
        "in telemetry from a sensor channel, typically indicating a sensor "
        "dropout or BMS communication loss.",
    "SuddenVoltageDrop":
        "A fine-grained anomaly subtype characterized by a step-like drop "
        "in cell terminal voltage.",
    "CurrentSpike":
        "A fine-grained anomaly subtype characterized by a single-sample "
        "current impulse beyond normal cycling bounds.",
    "ThermalDrift":
        "A fine-grained anomaly subtype characterized by a slow monotonic "
        "drift in measured temperature.",
    "CapacityInconsistency":
        "A fine-grained anomaly subtype characterized by a sudden drop in "
        "discharge-capacity inconsistent with prior cycle history.",
}


def _humanize(camel: str) -> str:
    """CamelCase -> 'Camel Case'."""
    if not camel:
        return ""
    out = []
    for i, ch in enumerate(camel):
        if ch.isupper() and i > 0:
            out.append(" ")
        out.append(ch)
    return "".join(out)


def build_fallback_proposal(missing_subtype: str,
                            broad_failure_mode: str = "Overdischarge",
                            patterns=None,
                            anomaly_id: str = "",
                            ) -> dict:
    
    if not missing_subtype:
        return {
            "new_anomaly_types": [],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [],
            "fallback_used": True,
            "fallback_reason": "no missing_subtype provided",
        }

    label = FALLBACK_SUBTYPE_LABELS.get(missing_subtype, _humanize(missing_subtype))
    comment = FALLBACK_SUBTYPE_COMMENTS.get(
        missing_subtype,
        f"A fine-grained anomaly subtype: {label}.",
    )
    broad_local = broad_failure_mode.replace(" ", "") if broad_failure_mode else "FailureMode"

    return {
        "new_anomaly_types": [{
            "name": missing_subtype,
            "label": label,
            "description": comment,
            "parent_class": "FailureMode",
        }],
        "new_causal_relationships": [{
            "from": f"{NS}{missing_subtype}",
            "to": f"{NS}{broad_local}",
            "description": f"{label} can lead to {broad_failure_mode}.",
        }],
        "new_properties": [],
        "rdf_triples": [
            {"subject": f"{NS}{missing_subtype}",
             "predicate": RDF_TYPE,
             "object": OWL_CLASS},
            {"subject": f"{NS}{missing_subtype}",
             "predicate": RDFS_SUBCLASSOF,
             "object": f"{NS}FailureMode"},
            {"subject": f"{NS}{missing_subtype}",
             "predicate": RDFS_LABEL,
             "object": label},
            {"subject": f"{NS}{missing_subtype}",
             "predicate": RDFS_COMMENT,
             "object": comment},
            {"subject": f"{NS}{missing_subtype}",
             "predicate": f"{NS}leadsTo",
             "object": f"{NS}{broad_local}"},
        ],
        "fallback_used": True,
        "fallback_reason": (
            f"deterministic fallback for missing subtype '{missing_subtype}' "
            f"(broad failure mode '{broad_failure_mode}' already exists)"
        ),
        "anomaly_id": anomaly_id,
        "patterns": list(patterns or []),
    }


def build_enrichment_prompt(anomaly_info, semantic_context):
   
    broad = (anomaly_info.get("failure_mode") or "FailureMode")
    missing_subtype = anomaly_info.get("subtype_proposal") or ""
    patterns = anomaly_info.get("patterns") or []
    obs = anomaly_info.get("observations") or {}
    a_id = obs.get("anomaly_id", "?")
    channel = obs.get("type", "")
    existing_modes = semantic_context.get("existing_failure_modes", [])

    label_human = FALLBACK_SUBTYPE_LABELS.get(
        missing_subtype, _humanize(missing_subtype)) if missing_subtype else ""

    constraint_block = ""
    if missing_subtype:
        constraint_block = (
            f"\n## CRITICAL CONSTRAINTS\n"
            f"- The matched broad failure mode is `{broad}` and it ALREADY "
            f"EXISTS in the ontology.\n"
            f"- The missing fine-grained subtype is `{missing_subtype}` "
            f"(human label: \"{label_human}\").\n"
            f"- You MUST propose `{missing_subtype}` as the new class.\n"
            f"- You MUST NOT propose `{broad}` as a new class — it is "
            f"already declared.\n"
            f"- The new class MUST be `rdfs:subClassOf` `FailureMode`.\n"
            f"- The new class MUST be linked to `{broad}` via "
            f"`bat:leadsTo`.\n"
        )

    prompt = f"""You are an expert in lithium-ion battery safety and ontology engineering.

A time-series anomaly has been detected. The framework has matched it to a
broad failure mode that already exists in the ontology, but the
fine-grained injected subtype is missing. Your job is to propose the
missing subtype as a new OWL class with a `leadsTo` edge to the broad mode.

## Detected Anomaly
- Anomaly ID: {a_id}
- Channel/Type: {channel}
- Anomaly Score: {anomaly_info.get('score', 'N/A')}
- Observed Patterns: {patterns}
- Broad (matched) Failure Mode: {broad}
- Missing Fine-Grained Subtype: {missing_subtype or '(not specified)'}

## Already-Known Failure Modes (DO NOT PROPOSE AGAIN)
{existing_modes}
{constraint_block}
## OUTPUT FORMAT
Return STRICT JSON ONLY. Do NOT wrap in Markdown fences. Do NOT include
JavaScript-style `//` comments. Do NOT add trailing prose. Do NOT include
trailing commas. The JSON MUST conform exactly to this schema:

{{
  "new_classes": [
    {{
      "class_name": "{missing_subtype or 'NewSubtype'}",
      "parent_class": "FailureMode",
      "label": "{label_human or 'Readable Label'}",
      "comment": "<one-sentence description of the subtype>",
      "leads_to": "{broad}"
    }}
  ],
  "new_relationships": [
    {{
      "subject": "{missing_subtype or 'NewSubtype'}",
      "predicate": "leadsTo",
      "object": "{broad}"
    }}
  ],
  "new_datatype_properties": [],
  "new_object_properties": [],
  "property_assertions": []
}}
"""
    return prompt


# Free-discovery Stage-5 prompt

def build_free_discovery_prompt(anomaly_info: dict,
                                semantic_context: dict) -> str:
    """Build the free-discovery Stage-5 prompt.

    anomaly_info is expected to carry, in addition to the constrained-mode
    fields (score / patterns / failure_mode / observations):
      - signal_statistics: dict of per-channel summary stats (real units)
      - segment_description: short prose about the anomalous segment
      - confidence: normalised confidence in [0, 1]
      - reconstruction_error: raw LSTM-AE recon error
      - channels: list of channel names actually carrying the anomaly
    Any missing field is rendered as 'n/a'.

    semantic_context is expected to carry:
      - existing_classes:           list of all local class labels
      - existing_failure_modes:     list of FailureMode subclass labels
      - existing_object_properties: list of bat: object-property local names
      - existing_datatype_properties: list of bat: datatype-property local names
      - known_rules:                list of human-readable rule strings
    """
    obs = anomaly_info.get("observations") or {}

    def _fmt_list(xs):
        if not xs:
            return "(none)"
        return ", ".join(sorted(str(x) for x in xs))

    candidate_id = obs.get("candidate_id", obs.get("anomaly_id", "?"))
    anomaly_id = obs.get("anomaly_id", "?") or "(no ground-truth ID)"
    cell_id = obs.get("cell_id", "?")
    cycle = obs.get("cycle", "?")
    recon_err = anomaly_info.get("reconstruction_error", "n/a")
    confidence = anomaly_info.get("confidence", anomaly_info.get("score", "n/a"))
    channels = anomaly_info.get("channels", []) or []
    patterns = anomaly_info.get("patterns") or []
    stage4_fm = anomaly_info.get("failure_mode") or "(no semantic match)"
    stage4_status = anomaly_info.get("stage4_status", "n/a")
    signal_stats = anomaly_info.get("signal_statistics") or {}
    segment_desc = anomaly_info.get("segment_description") or "(not provided)"

    existing_classes = semantic_context.get("existing_classes", [])
    existing_fms = semantic_context.get("existing_failure_modes", [])
    existing_obj_props = semantic_context.get("existing_object_properties",
                                              ["leadsTo"])
    existing_data_props = semantic_context.get("existing_datatype_properties",
                                               [])
    known_rules = semantic_context.get("known_rules", [])

    # Render signal stats compactly for the prompt
    if isinstance(signal_stats, dict) and signal_stats:
        stats_lines = []
        for ch, ch_stats in signal_stats.items():
            if isinstance(ch_stats, dict):
                kvs = ", ".join(f"{k}={v}" for k, v in ch_stats.items())
                stats_lines.append(f"  - {ch}: {kvs}")
            else:
                stats_lines.append(f"  - {ch}: {ch_stats}")
        stats_block = "\n".join(stats_lines)
    else:
        stats_block = "  (not provided)"

    prompt = f"""You are an expert in lithium-ion battery safety, anomaly diagnosis, and ontology engineering.

A time-series anomaly has been detected in a lithium-ion battery dataset. The existing ontology may not contain the most appropriate fine-grained concept needed to describe this anomaly.

Your task is to propose possible ontology enrichments based only on the anomaly evidence and the existing ontology vocabulary.

Do NOT assume that a new class is always required.
Do NOT duplicate an existing class.
Do NOT invent relations if an existing relation is sufficient.
Do NOT simply repeat the broad failure mode already matched by the rule engine.

## Anomaly Evidence

- Candidate ID: {candidate_id}
- Anomaly ID, if available: {anomaly_id}
- Cell: {cell_id}
- Cycle: {cycle}
- Reconstruction error: {recon_err}
- Normalised confidence: {confidence}
- Affected channel(s): {_fmt_list(channels)}
- Observed patterns: {_fmt_list(patterns)}
- Stage-4 semantic match: {stage4_fm}
- Stage-4 status: {stage4_status}
- Signal statistics:
{stats_block}
- Short description of anomalous segment: {segment_desc}

## Existing Ontology Vocabulary

Existing classes:
{_fmt_list(existing_classes)}

Existing subclasses of FailureMode:
{_fmt_list(existing_fms)}

Existing object properties:
{_fmt_list(existing_obj_props)}

Existing datatype properties:
{_fmt_list(existing_data_props)}

Known semantic rules:
{_fmt_list(known_rules)}

## Task

Based on the anomaly evidence, propose one of the following:

1. No ontology update is needed because the anomaly is already covered by existing classes.
2. One or more new fine-grained classes should be added.
3. One or more new relations between existing or proposed classes should be added.
4. The evidence is too weak or ambiguous, so the proposal should be rejected or sent to human review.

For each proposed class:
- use a valid PascalCase class name;
- select an existing ontology class as parent;
- provide a human-readable label;
- provide a one-sentence rdfs:comment;
- cite the evidence that supports the proposal;
- give a confidence score between 0 and 1;
- explain why the class is not a duplicate of an existing class.

For each proposed relation:
- use only an existing object property;
- ensure the subject and object are existing classes or proposed classes;
- give a confidence score between 0 and 1;
- justify the relation using the anomaly evidence.

## Output Format

Return STRICT JSON ONLY.
Do NOT use Markdown fences.
Do NOT include comments.
Do NOT include trailing commas.
Do NOT add explanatory text outside the JSON.

Use exactly this schema:

{{
  "update_needed": true,
  "decision": "add|skip_existing|reject_uncertain|human_review",
  "candidate_new_classes": [
    {{
      "class_name": "<PascalCaseClassName>",
      "parent_class": "<ExistingOntologyClass>",
      "label": "<Human-readable label>",
      "comment": "<one-sentence ontology comment>",
      "evidence": ["<evidence item 1>", "<evidence item 2>"],
      "confidence": 0.0,
      "justification": "<why this class is needed and why existing classes are insufficient>"
    }}
  ],
  "candidate_new_relationships": [
    {{
      "subject": "<CandidateOrExistingClass>",
      "predicate": "<ExistingObjectProperty>",
      "object": "<CandidateOrExistingClass>",
      "evidence": ["<evidence item 1>"],
      "confidence": 0.0,
      "justification": "<why this relation is supported>"
    }}
  ],
  "duplicate_risk": [
    {{
      "candidate": "<CandidateClassName>",
      "similar_existing_class": "<ExistingClassName>",
      "risk": "low|medium|high",
      "reason": "<why this may or may not be a duplicate>"
    }}
  ],
  "rejected_candidates": [
    {{
      "class_name": "<RejectedClassName>",
      "reason": "<why this candidate should not be inserted>"
    }}
  ],
  "human_review_note": "<short note explaining what a human reviewer should verify>"
}}
"""
    return prompt


# JSON extraction
def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` and ``` ... ``` fences."""
    text = re.sub(r"^\s*```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Also strip any inline fences mid-text
    text = text.replace("```json", "").replace("```JSON", "").replace("```", "")
    return text


def _strip_js_comments(text: str) -> str:
    """Remove `// …` and `/* … */` style comments that some LLMs emit."""
    # Block comments
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    # Line comments — careful not to touch URLs (http://...). Only strip
    # `//` that follows whitespace or a JSON token, not after a colon
    # immediately preceded by `http` / `https`.
    out = []
    i = 0
    n = len(text)
    in_str = False
    str_quote = ""
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == str_quote:
                in_str = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_quote = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            # Skip until end of line
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before `}` or `]` (illegal in strict JSON)."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the substring corresponding to the first balanced {...} block."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    str_quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if ch == "\\":
                continue
            if ch == str_quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            str_quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def robust_parse_json(text: str) -> dict:
    """Parse possibly-malformed JSON from an LLM response.

    Steps:
      1. Strip Markdown fences.
      2. Remove `//` and `/* */` comments.
      3. Remove trailing commas.
      4. Extract the first balanced JSON object.
      5. Try `json.loads`. If that fails, retry once after stripping
         non-printable characters and unquoted single-quoted strings.

    Returns:
        dict — parsed JSON (possibly empty); on hard failure returns
        `{"_parse_error": "<reason>", "_raw": "<truncated raw text>"}`.
    """
    if isinstance(text, dict):
        text = text.get("content", json.dumps(text))
    if not isinstance(text, str):
        text = str(text)

    cleaned = _strip_markdown_fences(text)
    cleaned = _strip_js_comments(cleaned)
    cleaned = _strip_trailing_commas(cleaned)

    block = _extract_first_json_object(cleaned)
    if block is None:
        return {"_parse_error": "no_json_object_found",
                "_raw": text[:2000]}

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        repaired = _strip_trailing_commas(block)
        # Convert single-quoted keys/strings to double quotes (heuristic)
        repaired_q = re.sub(r"'([^'\\]*)'", r'"\1"', repaired)
        try:
            return json.loads(repaired_q)
        except json.JSONDecodeError as e:
            return {"_parse_error": f"json_decode_error: {e}",
                    "_raw": text[:2000]}


# Schema normalization — accept both legacy and new schema
def normalize_proposal_schema(parsed: dict, broad: str = "Overdischarge") -> dict:
    """Normalize either the new strict schema (new_classes/new_relationships/...)
    or the legacy schema (new_anomaly_types/new_causal_relationships/...) into a
    single internal representation used by the rest of the pipeline.

    Internal representation keys:
      - new_anomaly_types
      - new_causal_relationships
      - new_properties
      - rdf_triples
    """
    if not isinstance(parsed, dict):
        return {"new_anomaly_types": [], "new_causal_relationships": [],
                "new_properties": [], "rdf_triples": []}

    # If legacy keys already present, pass through (still normalize triples).
    if "new_anomaly_types" in parsed or "new_causal_relationships" in parsed:
        return {
            "new_anomaly_types": parsed.get("new_anomaly_types", []) or [],
            "new_causal_relationships": parsed.get("new_causal_relationships", []) or [],
            "new_properties": parsed.get("new_properties", []) or [],
            "rdf_triples": parsed.get("rdf_triples", []) or [],
            "_parse_error": parsed.get("_parse_error"),
        }

    # Strict schema -> legacy
    out_types = []
    out_rels = []
    out_triples = []

    for c in parsed.get("new_classes", []) or []:
        name = c.get("class_name") or c.get("name") or ""
        if not name:
            continue
        label = c.get("label") or _humanize(name)
        parent = c.get("parent_class") or "FailureMode"
        comment = c.get("comment") or c.get("description") or ""
        out_types.append({
            "name": name,
            "label": label,
            "description": comment,
            "parent_class": parent,
        })
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDF_TYPE,
                            "object": OWL_CLASS})
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDFS_SUBCLASSOF,
                            "object": f"{NS}{parent}"})
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDFS_LABEL,
                            "object": label})
        if comment:
            out_triples.append({"subject": f"{NS}{name}",
                                "predicate": RDFS_COMMENT,
                                "object": comment})
        leads_to = c.get("leads_to")
        if leads_to:
            target = leads_to if leads_to.startswith("http") \
                else f"{NS}{leads_to.replace(' ', '')}"
            out_rels.append({
                "from": f"{NS}{name}",
                "to": target,
                "description": f"{label} can lead to {leads_to}.",
            })
            out_triples.append({"subject": f"{NS}{name}",
                                "predicate": f"{NS}leadsTo",
                                "object": target})

    for r in parsed.get("new_relationships", []) or []:
        subj = r.get("subject") or r.get("from") or ""
        pred = r.get("predicate") or "leadsTo"
        obj = r.get("object") or r.get("to") or ""
        if not subj or not obj:
            continue
        s_uri = subj if subj.startswith("http") else f"{NS}{subj.replace(' ', '')}"
        o_uri = obj if obj.startswith("http") else f"{NS}{obj.replace(' ', '')}"
        p_uri = pred if pred.startswith("http") else f"{NS}{pred}"
        out_rels.append({
            "from": s_uri, "to": o_uri,
            "description": r.get("description", ""),
        })
        out_triples.append({"subject": s_uri, "predicate": p_uri, "object": o_uri})

    return {
        "new_anomaly_types": out_types,
        "new_causal_relationships": out_rels,
        "new_properties": (parsed.get("new_datatype_properties", []) or []) +
                          (parsed.get("new_object_properties", []) or []),
        "rdf_triples": out_triples,
        "_parse_error": parsed.get("_parse_error"),
    }


def normalize_free_discovery_schema(parsed: dict) -> dict:
    """Normalise the free-discovery JSON response into the internal
    representation used by the rest of the pipeline, while preserving the
    free-discovery-specific fields (decision, confidence, evidence,
    justification, duplicate_risk, rejected_candidates, human_review_note).

    The internal representation keys (new_anomaly_types,
    new_causal_relationships, new_properties, rdf_triples) are populated
    so existing dedup + validator + apply_approved_updates work unchanged.

    If decision is "skip_existing" or "reject_uncertain", no classes /
    relationships / triples are emitted — the caller is expected to honour
    the decision and skip Stage 6 / 7.
    """
    if not isinstance(parsed, dict):
        return {
            "new_anomaly_types": [],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [],
            "free_discovery": {
                "decision": "parse_error",
                "update_needed": False,
                "rejected_candidates": [],
                "duplicate_risk": [],
                "human_review_note": "",
                "candidate_new_classes": [],
                "candidate_new_relationships": [],
            },
            "_parse_error": "non_dict_parse_result",
        }

    decision = (parsed.get("decision") or "").strip().lower() or "add"
    update_needed = bool(parsed.get("update_needed", decision == "add"))

    cand_classes = parsed.get("candidate_new_classes", []) or []
    cand_rels = parsed.get("candidate_new_relationships", []) or []
    duplicate_risk = parsed.get("duplicate_risk", []) or []
    rejected = parsed.get("rejected_candidates", []) or []
    human_note = parsed.get("human_review_note", "") or ""

    fd_block = {
        "decision": decision,
        "update_needed": update_needed,
        "candidate_new_classes": cand_classes,
        "candidate_new_relationships": cand_rels,
        "duplicate_risk": duplicate_risk,
        "rejected_candidates": rejected,
        "human_review_note": human_note,
    }

    # Honour the decision: if not "add", emit nothing for Stages 6/7.
    if decision != "add" or not update_needed:
        return {
            "new_anomaly_types": [],
            "new_causal_relationships": [],
            "new_properties": [],
            "rdf_triples": [],
            "free_discovery": fd_block,
            "_parse_error": parsed.get("_parse_error"),
        }

    out_types = []
    out_rels = []
    out_triples = []

    for c in cand_classes:
        if not isinstance(c, dict):
            continue
        name = (c.get("class_name") or c.get("name") or "").strip()
        if not name:
            continue
        label = c.get("label") or _humanize(name)
        parent = c.get("parent_class") or "FailureMode"
        comment = c.get("comment") or c.get("description") or ""
        out_types.append({
            "name": name,
            "label": label,
            "description": comment,
            "parent_class": parent,
            # free-discovery extras kept on the item for downstream reporting
            "confidence": c.get("confidence"),
            "evidence": c.get("evidence", []) or [],
            "justification": c.get("justification", ""),
        })
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDF_TYPE,
                            "object": OWL_CLASS})
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDFS_SUBCLASSOF,
                            "object": f"{NS}{parent.replace(' ', '')}"})
        out_triples.append({"subject": f"{NS}{name}",
                            "predicate": RDFS_LABEL,
                            "object": label})
        if comment:
            out_triples.append({"subject": f"{NS}{name}",
                                "predicate": RDFS_COMMENT,
                                "object": comment})

    for r in cand_rels:
        if not isinstance(r, dict):
            continue
        subj = (r.get("subject") or r.get("from") or "").strip()
        pred = (r.get("predicate") or "leadsTo").strip()
        obj = (r.get("object") or r.get("to") or "").strip()
        if not subj or not obj:
            continue
        s_uri = subj if subj.startswith("http") else f"{NS}{subj.replace(' ', '')}"
        o_uri = obj if obj.startswith("http") else f"{NS}{obj.replace(' ', '')}"
        p_uri = pred if pred.startswith("http") else f"{NS}{pred}"
        out_rels.append({
            "from": s_uri, "to": o_uri,
            "description": r.get("justification", ""),
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence", []) or [],
        })
        out_triples.append({"subject": s_uri, "predicate": p_uri, "object": o_uri})

    return {
        "new_anomaly_types": out_types,
        "new_causal_relationships": out_rels,
        "new_properties": [],
        "rdf_triples": out_triples,
        "free_discovery": fd_block,
        "_parse_error": parsed.get("_parse_error"),
    }


class LLMOntologyEnricher:
    """
    Uses a HuggingFace LLM to propose ontology enrichments
    based on detected anomalies and semantic context.
    """

    def __init__(self, model_name=None, device="auto", use_mock=False,
                 prompt_mode="constrained"):
        """
        Args:
            model_name: HuggingFace model ID (from ENRICHMENT_MODELS)
            device: "auto", "cpu", or "cuda"
            use_mock: if True, use mock responses instead of loading model
            prompt_mode: "constrained" (default; existing behaviour — the
                prompt names the missing subtype) or "free_discovery"
                (the LLM is given evidence + existing vocabulary and must
                decide whether and what to enrich).
        """
        if prompt_mode not in ("constrained", "free_discovery"):
            raise ValueError(f"prompt_mode must be one of "
                             f"'constrained' / 'free_discovery', "
                             f"got {prompt_mode!r}")
        self.model_name = model_name or ENRICHMENT_MODELS[0]
        self.use_mock = use_mock
        self.prompt_mode = prompt_mode
        self.pipeline = None

        if not use_mock:
            self._load_model(device)

    def _load_model(self, device):
        """Load the HuggingFace text-generation pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            print(f"Loading model: {self.model_name} ...")
            self.pipeline = hf_pipeline(
                "text-generation",
                model=self.model_name,
                device_map=device,
                dtype="auto",
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
            print(f"Model loaded: {self.model_name}")
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {e}")
            print("Falling back to mock mode.")
            self.use_mock = True

    def generate_enrichment(self, anomaly_info, semantic_context):
        """
        Generate ontology enrichment proposals.

        Args:
            anomaly_info: dict with keys: score, patterns, failure_mode, observations
            semantic_context: dict with keys: matched_rules, causal_chains, failure_modes

        Returns:
            dict with proposed new ontology elements (legacy schema with
            new_anomaly_types / new_causal_relationships / new_properties /
            rdf_triples).
        """
        if self.prompt_mode == "free_discovery":
            prompt = build_free_discovery_prompt(anomaly_info, semantic_context)
        else:
            prompt = build_enrichment_prompt(anomaly_info, semantic_context)

        if self.use_mock:
            return self._mock_response(anomaly_info)

        # Call the LLM
        messages = [{"role": "user", "content": prompt}]
        output = self.pipeline(messages, return_full_text=False)
        response_text = output[0]["generated_text"]

        return self._parse_llm_response(response_text, prompt=prompt)

    def _parse_llm_response(self, response_text, prompt=None):
        """Robust parse: handle Markdown fences, // comments, trailing commas,
        and trailing prose. Dispatches on self.prompt_mode to the appropriate
        schema normaliser. On hard failure, returns a structured parse_error
        dict so the caller can activate the fallback.
        """
        if isinstance(response_text, dict):
            response_text = response_text.get("content", str(response_text))

        parsed = robust_parse_json(response_text)
        if self.prompt_mode == "free_discovery":
            normalized = normalize_free_discovery_schema(parsed)
        else:
            normalized = normalize_proposal_schema(parsed)

        # Always preserve the raw text + prompt for audit, not only on errors.
        normalized["raw_response"] = response_text
        if prompt is not None:
            normalized["prompt"] = prompt
        if parsed.get("_parse_error"):
            normalized["_parse_error"] = parsed["_parse_error"]
        return normalized

    def _mock_response(self, anomaly_info):
        """Generate a mock enrichment response for testing without GPU."""
        patterns = anomaly_info.get("patterns", [])
        failure = anomaly_info.get("failure_mode", "Unknown")

        mock = {
            "new_anomaly_types": [
                {
                    "name": "RapidDegradation",
                    "label": "Rapid Degradation",
                    "description": "Accelerated capacity loss due to sustained thermal stress",
                    "parent_class": "FailureMode",
                }
            ],
            "new_causal_relationships": [
                {
                    "from": f"{NS}Overheating",
                    "to": f"{NS}RapidDegradation",
                    "description": "Sustained overheating causes accelerated capacity degradation",
                }
            ],
            "new_properties": [
                {
                    "name": "degradationRate",
                    "domain": f"{NS}RapidDegradation",
                    "range": "xsd:float",
                    "description": "Rate of capacity loss in %/cycle",
                }
            ],
            "rdf_triples": [
                {"subject": f"{NS}RapidDegradation",
                 "predicate": RDF_TYPE,
                 "object": OWL_CLASS},
                {"subject": f"{NS}RapidDegradation",
                 "predicate": RDFS_SUBCLASSOF,
                 "object": f"{NS}FailureMode"},
                {"subject": f"{NS}RapidDegradation",
                 "predicate": RDFS_LABEL,
                 "object": "Rapid Degradation"},
                {"subject": f"{NS}Overheating",
                 "predicate": f"{NS}leadsTo",
                 "object": f"{NS}RapidDegradation"},
            ],
        }
        return mock


# Module 6: Human-in-the-Loop Validation

class HumanValidator:
    """
    Simulates a domain expert reviewing proposed ontology updates.
    In a real system, this would be a UI or API endpoint.
    """

    def __init__(self, auto_approve=False):
        """
        Args:
            auto_approve: if True, automatically approve all proposals (for testing)
        """
        self.auto_approve = auto_approve
        self.review_log = []

    def review_proposal(self, proposal, interactive=True):
        """
        Present proposed ontology updates to a domain expert for review.

        Args:
            proposal: dict from LLMOntologyEnricher.generate_enrichment()
            interactive: if True, prompt for user input

        Returns:
            dict with accepted/rejected items
        """
        review_result = {
            "accepted_types": [],
            "accepted_relationships": [],
            "accepted_triples": [],
            "rejected_items": [],
            "reviewer_notes": "",
        }

        print("\n" + "="*70)
        print("DOMAIN EXPERT REVIEW — Proposed Ontology Updates")
        print("="*70)

        # Review new anomaly types
        for item in proposal.get("new_anomaly_types", []):
            print(f"\n  [NEW TYPE] {item['name']}: {item.get('description', '')}")
            if self._get_approval(f"Accept '{item['name']}'?", interactive):
                review_result["accepted_types"].append(item)
            else:
                review_result["rejected_items"].append(("type", item["name"]))

        # Review new causal relationships
        for item in proposal.get("new_causal_relationships", []):
            print(f"\n  [NEW RELATION] {item['from']} => {item['to']}: {item.get('description', '')}")
            if self._get_approval(f"Accept this relationship?", interactive):
                review_result["accepted_relationships"].append(item)
            else:
                review_result["rejected_items"].append(("relationship", item.get("description", "")))

        # Review RDF triples
        triples = proposal.get("rdf_triples", [])
        if triples:
            print(f"\n  [RDF TRIPLES] {len(triples)} proposed triples")
            for t in triples:
                s = t["subject"].split("#")[-1] if "#" in t["subject"] else t["subject"]
                p = t["predicate"].split("#")[-1] if "#" in t["predicate"] else t["predicate"]
                o = str(t["object"]).split("#")[-1] if "#" in str(t["object"]) else t["object"]
                print(f"    ({s}, {p}, {o})")

            if self._get_approval("Accept all triples?", interactive):
                review_result["accepted_triples"] = triples
            else:
                review_result["rejected_items"].append(("triples", f"{len(triples)} triples"))

        # Summary
        n_accepted = (len(review_result["accepted_types"]) +
                      len(review_result["accepted_relationships"]) +
                      len(review_result["accepted_triples"]))
        n_rejected = len(review_result["rejected_items"])
        print(f"\n  Summary: {n_accepted} accepted, {n_rejected} rejected")
        print("="*70)

        self.review_log.append(review_result)
        return review_result

    def _get_approval(self, question, interactive):
        """Get approval from expert (or auto-approve)."""
        if self.auto_approve:
            print(f"    [APPROVED] {question}")
            return True

        if not interactive:
            print(f"    [AUTO] Non-interactive mode: auto-approved")
            return True

        while True:
            response = input(f"    {question} [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            print("    Please answer 'y' or 'n'.")



# Ontology Existence Filtering — Remove Duplicates Before Human Review

def filter_proposals_by_ontology(proposal, reasoner):
    """
    Remove from LLM proposal any items that already exist in the ontology.

    Checks:
    - new_anomaly_types: removed if label already exists (via failure_mode_exists_by_label)
    - new_causal_relationships: removed if (from, to) pair already exists
    - rdf_triples: keep all triples whose SUBJECT corresponds to a class
      that survived filtering. Triples that target an already-existing
      duplicate-only class are dropped (otherwise we would re-emit
      already-existing class declarations).

    Args:
        proposal: dict from LLMOntologyEnricher.generate_enrichment() with keys:
                  new_anomaly_types, new_causal_relationships, new_properties, rdf_triples
        reasoner: SemanticReasoner instance

    Returns:
        (filtered_proposal, removed_summary):
        - filtered_proposal: cleaned version of proposal with duplicates removed
        - removed_summary: dict with "types" and "relationships" lists of what was filtered
    """
    import copy

    filtered = copy.deepcopy(proposal)
    removed = {"types": [], "relationships": []}

    # Filter new_anomaly_types — track which class names were dropped as duplicates
    kept_types = []
    dropped_class_names = set()
    for item in filtered.get("new_anomaly_types", []):
        label = item.get("label") or item.get("name", "")
        name = item.get("name") or label
        if label and reasoner.failure_mode_exists_by_label(label):
            removed["types"].append(label)
            dropped_class_names.add(name)
            print(f"  [PRE-FILTER] Skipping already-existing type: '{label}'")
        else:
            kept_types.append(item)
    filtered["new_anomaly_types"] = kept_types

    # Filter new_causal_relationships
    kept_rels = []
    for item in filtered.get("new_causal_relationships", []):
        from_uri = item.get("from", "")
        to_uri = item.get("to", "")
        if from_uri and to_uri and reasoner.causal_relation_exists(from_uri, to_uri):
            removed["relationships"].append(f"{from_uri} => {to_uri}")
            from_label = from_uri.split("#")[-1] if "#" in from_uri else from_uri
            to_label = to_uri.split("#")[-1] if "#" in to_uri else to_uri
            print(f"  [PRE-FILTER] Skipping already-existing relation: {from_label} => {to_label}")
        else:
            kept_rels.append(item)
    filtered["new_causal_relationships"] = kept_rels

    # Drop any rdf_triples whose subject corresponds to a dropped class AND
    # whose predicate is rdf:type / rdfs:subClassOf / rdfs:label / rdfs:comment.
    # Relations on dropped classes are also pruned to avoid re-asserting on
    # already-existing class identifiers when those identifiers came from
    # a duplicate proposal.
    if dropped_class_names:
        kept_triples = []
        for t in filtered.get("rdf_triples", []):
            subj = str(t.get("subject", ""))
            local = subj.split("#")[-1] if "#" in subj else subj
            if local in dropped_class_names:
                continue
            kept_triples.append(t)
        filtered["rdf_triples"] = kept_triples

    return filtered, removed


# Apply approved updates — serialize types, relationships AND triples
def _to_class_triples(item: dict) -> list:
    """Convert an accepted_type item into the minimal class-declaration
    triple set: rdf:type owl:Class, rdfs:subClassOf <parent>, rdfs:label.
    """
    name = item.get("name") or item.get("label", "")
    if not name:
        return []
    label = item.get("label") or _humanize(name)
    parent = item.get("parent_class") or "FailureMode"
    parent_uri = parent if parent.startswith("http") else f"{NS}{parent.replace(' ', '')}"
    name_uri = f"{NS}{name.replace(' ', '')}"
    triples = [
        (name_uri, RDF_TYPE, OWL_CLASS),
        (name_uri, RDFS_SUBCLASSOF, parent_uri),
        (name_uri, RDFS_LABEL, label),
    ]
    desc = item.get("description")
    if desc:
        triples.append((name_uri, RDFS_COMMENT, desc))
    return triples


def _to_relationship_triples(item: dict) -> list:
    """Convert an accepted_relationship item into a leadsTo triple."""
    s = item.get("from") or item.get("subject", "")
    o = item.get("to") or item.get("object", "")
    if not s or not o:
        return []
    pred = item.get("predicate", "leadsTo")
    s_uri = s if s.startswith("http") else f"{NS}{s.replace(' ', '')}"
    o_uri = o if o.startswith("http") else f"{NS}{o.replace(' ', '')}"
    p_uri = pred if pred.startswith("http") else f"{NS}{pred}"
    return [(s_uri, p_uri, o_uri)]


def apply_approved_updates(reasoner, review_result):
    """
    Apply approved ontology updates to the semantic reasoner's graph.

    Serializes ALL three accepted buckets:
      - accepted_types        -> (Class, subClassOf, label[, comment]) triples
      - accepted_relationships -> (from, leadsTo, to) triples
      - accepted_triples       -> as-is

    Subject classes from accepted_types are inserted BEFORE accepted_relationships
    so that a leadsTo edge whose subject is a freshly-proposed class still
    resolves to a declared class in the resulting graph.

    Pre-filters everything against the existing graph so duplicates are
    counted, not re-inserted.

    Returns:
        Number of new triples actually added (duplicates not counted).
    """
    raw_triples = []

    # 1. accepted_types -> class declarations FIRST
    for item in review_result.get("accepted_types", []):
        raw_triples.extend(_to_class_triples(item))

    # 2. accepted_relationships -> leadsTo edges
    for item in review_result.get("accepted_relationships", []):
        raw_triples.extend(_to_relationship_triples(item))

    # 3. accepted_triples -> as-is
    for triple in review_result.get("accepted_triples", []):
        raw_triples.append((
            triple["subject"],
            triple["predicate"],
            triple["object"],
        ))

    if not raw_triples:
        return 0

    # Pre-filter: remove triples already in the graph
    new_triples, duplicates = reasoner.filter_new_triples(raw_triples)

    if duplicates:
        print(f"  [DEDUP] Skipped {len(duplicates)} triples already in ontology:")
        for s, p, o in duplicates:
            s_frag = str(s).split("#")[-1] if "#" in str(s) else s
            p_frag = str(p).split("#")[-1] if "#" in str(p) else p
            o_frag = str(o).split("#")[-1] if "#" in str(o) else o
            print(f"    (already exists) ({s_frag}, {p_frag}, {o_frag})")

    if new_triples:
        reasoner.add_new_triples(new_triples)
        print(f"Applied {len(new_triples)} new triples to ontology "
              f"({len(duplicates)} duplicates skipped).")
    else:
        print("  All approved triples already exist in ontology. Nothing added.")

    return len(new_triples)
