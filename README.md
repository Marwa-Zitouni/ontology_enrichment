## Overview

The framework is a closed-loop pipeline with seven stages:

1. **Time-series input** voltage, current, temperature windows from a battery management system.
2. **Anomaly detection**  unsupervised bidirectional LSTM autoencoder trained on normal cycles.
3. **False-positive filtering**  confidence, temporal persistence, and ontology-based semantic checks.
4. **Semantic reasoning**  SPARQL + OWL validation over the battery ontology (confirmed / rejected / enrichment-triggering).
5. **LLM-assisted enrichment**  lightweight LLMs (Qwen 2.5 1.5B, Phi-3-mini, SmolLM2-1.7B) propose new classes and relations as strict JSON.
6. **Human validation**  expert reviews proposals before integration.
7. **Controlled ontology update**  duplicate filtering and consistency checks before merging into the TBox.

## Repository Structure

```
.
├── data/                                  # MIT batch 1 loader + battery ontology (.ttl)
│   ├── battery_ontology.ttl
│   ├── mit_batch1_final/
│   └── mit_data.py
├── models/                                # LSTM autoencoder used as the detector
│   ├── autoencoders.py
│   └── lstm_autoencoder.py
├── filtering/                             # Stage 3 false-positive filtering
│   └── false_positive_filter.py
├── reasoning/                             # Stage 4 SPARQL + OWL reasoning
│   └── semantic_reasoner.py
├── enrichment/                            # Stage 5 LLM proposal generation
│   ├── lightweight_llms.py
│   └── llm_enrichment.py
├── experiments/
│   └── conference_validation_mit/         # Conference case study
│       ├── ontology/
│       ├── regime_A_conservative/
│       ├── regime_B_permissive/
│       ├── results_llm/
│       ├── results_llm_free_discovery/
│       ├── run_conference_experiment.py
│       ├── run_llm_sweep.py
│       └── run_llm_sweep_free_discovery.py
├── tests/
├── pipeline.py                            # End-to-end pipeline entry point
├── run.py
└── requirements.txt
```
## Data

The case study uses a controlled subset of the MIT lithium-ion cycling dataset (Severson et al., 2019): batch 1, 8 cells, 10 cycles per cell, yielding N = 80 windows. The processed subset is shipped under `data/mit_batch1_final/`.

Since the dataset has no event-level fault labels, seven anomaly archetypes (A1–A7) are injected with controlled operators across voltage, current, temperature, and capacity channels (see Table 2 of the paper).
