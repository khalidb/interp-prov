# An Interpretive Provenance Model for Scientific Reasoning: A Computational Biology Study

**Authors:** Khalid Belhajjame (LAMSADE, Université Paris-Dauphine, PSL Research University)

---

## Overview

This repository contains all code, data, prompts, gold standards, and LaTeX source for the paper:

> *An Interpretive Provenance Model for Scientific Reasoning: A Computational Biology Study*

The paper proposes an **Interpretive Provenance Model** that captures the reasoning chain connecting analytical results to scientific conclusions, grounded in W3C PROV. We distinguish two forms of interpretive provenance:

- **Prospective interpretive provenance**: an LLM agent interprets experimental results and generates a structured provenance record of its reasoning.
- **Retrospective interpretive provenance**: the reasoning chain is reconstructed from the Results and Discussion sections of a published paper.

We validate the model empirically across three published computational biology papers, using LLaMA 3 8B and Mistral 7B as interpretation agents under three analytical stances and five repetitions per configuration, yielding 120 provenance-instrumented runs.

---

## Repository Structure

```
project/
│
├── paper/                         # LaTeX source files
│   ├── main_paper.tex             # Master file (IEEE format)
│   ├── introduction.tex
│   ├── background.tex             # Related work
│   ├── model.tex                  # The Interpretive Provenance Model
│   ├── methods.tex                # Study design and evaluation framework
│   ├── results.tex                # Empirical findings
│   ├── conclusion.tex             # Discussion and concluding remarks
│   └── references.bib             # Bibliography
│
├── model/
│   └── interpretive_provenance_model.puml   # PlantUML diagram source
│
├── gold_standards/                # Manually constructed reference records
│   ├── gold_standard_CS1.json     # Kim & Lee (2022) — strong signal
│   ├── gold_standard_CS2.json     # Cui et al. (2021) — borderline signal
│   └── gold_standard_CS3.json     # Li et al. (2020) — interpretation-dependent
│
├── prompts/                       # Prompt templates
│   └── prompt_architecture.md     # Full prompt design and JSON schemas
│
├── outputs/                       # LLM outputs (120 JSON files)
│   ├── CS1_prospective_conservative_llama3_rep1.json
│   ├── CS1_prospective_conservative_llama3_rep2.json
│   └── ...                        # All 120 run outputs
│
├── evaluation/                    # Evaluation scripts and results
│   ├── evaluate.py                # Automated evaluation (EC1, EC3, EC5, EC6, EC7)
│   ├── human_eval_sheet.py        # Human evaluation sheet generator
│   ├── summarise_results.py       # Results summary printer
│   ├── evaluation_results.json    # Full evaluation results per run
│   ├── evaluation_summary.csv     # Summary table (all criteria, all runs)
│   └── human_eval_sheet_completed.json   # Completed human evaluation (EC2, EC4)
│
└── run_experiments.py             # Experiment runner script
```

---

## Case Studies

| ID  | Paper | Signal type | In eScience version |
|-----|-------|-------------|---------------------|
| CS1 | Kim & Lee (2022), *Journal of Personalized Medicine* — 20 breast cancer biomarker candidates, AUC ≥ 0.9, TCGA-BRCA | Strong | Yes |
| CS2 | Cui et al. (2021), *Human Genomics* — RNA-seq reproducibility, DEG overlap < 40% | Borderline | Yes |
| CS3 | Li et al. (2020), *PeerJ* — BRCA1/2-mutant breast cancer, GSEA + DEG + PPI + survival → 5 hub genes | Interpretation-dependent | Yes |

---

## Experimental Setup

- **LLM agents:** LLaMA 3 8B (`llama3:latest`, Q4_0) and Mistral 7B (`mistral:latest`, Q4_K_M), served locally via Ollama
- **Decoding:** greedy (temperature = 0)
- **Analytical stances (prospective mode):** conservative, discovery-oriented, evidence-only
- **Retrospective mode:** agent provided with HumanReasoningTrace and asked to reconstruct faithful provenance
- **Repetitions:** 5 per configuration
- **Total runs:** 90 prospective + 30 retrospective = 120

---

## Evaluation Criteria

| Criterion | Description | Method |
|-----------|-------------|--------|
| EC1 | Evidence coverage (Jaccard overlap) | Automatic |
| EC2 | Claim equivalence to gold standard | Human |
| EC3 | Claim scope calibration | Automatic |
| EC4 | Rationale alignment with human reasoning trace | Human |
| EC5 | Uncertainty calibration | Automatic |
| EC6 | Overclaim detection | Automatic |
| EC7 | Cross-agent and cross-stance consistency | Automatic |

---

## Reproducing the Experiments

### Prerequisites

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama pull mistral

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install requests
```

### Run experiments

```bash
python3 run_experiments.py
```

Outputs are saved to `outputs/`. The script is idempotent: re-running skips completed outputs.

### Run evaluation

```bash
# Automated evaluation (EC1, EC3, EC5, EC6, EC7)
python3 evaluation/evaluate.py

# Generate human evaluation sheet (EC2, EC4)
python3 evaluation/human_eval_sheet.py

# Print full summary
python3 evaluation/summarise_results.py
```

---

## Key Findings

- Retrospective mode substantially improves evidence coverage (Jaccard: 0.33→1.0 for CS2, 0.26→0.86 for CS3) and claim equivalence (EC2 = 1.0 in all retrospective configurations).
- Both agents systematically overclaim in retrospective mode for CS2: every retrospective run produces a `validated` claim for a paper whose gold standard scope is `exploratory`, despite correctly following the human reasoning steps.
- Prospective mode fails to reach the biomarker candidate conclusion for CS3 in conservative and evidence-only stances: agents anchor on the GSEA pathway finding without progressing to the convergent multi-evidence argument.
- The analytical stance reliably affects claim scope direction (EC7 cross-stance = 1.0 universally) but does not prevent scope inflation in borderline papers.
- Mistral achieves perfect rationale alignment (EC4 = 2.0) for CS2 in retrospective mode, the only perfect alignment in the dataset.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{belhajjame2026interpretive,
  author    = {Belhajjame, Khalid},
  title     = {An Interpretive Provenance Model for Scientific Reasoning:
               A Computational Biology Study},
  booktitle = {Proceedings of the IEEE International Conference on
               eScience (eScience 2026)},
  year      = {2026}
}
```

---

## Licence

Code: MIT Licence  
Paper: © 2026 IEEE. Personal use permitted. For other uses contact IEEE.

---

## Contact

Khalid Belhajjame  
LAMSADE, Université Paris-Dauphine, PSL Research University  
khalid.belhajjame@dauphine.fr
