"""
Experiment runner for:
"An Interpretive Provenance Model for Scientific Reasoning:
 A Computational Biology Study"

Runs 120 provenance-instrumented interpretations:
- 90 prospective runs: 3 papers x 3 stances x 2 agents x 5 repetitions
- 30 retrospective runs: 3 papers x 2 agents x 5 repetitions

Outputs are saved as JSON files in ./outputs/
"""

import json
import os
import time
import requests
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = "./outputs"
GOLD_STANDARD_DIR = "./gold_standards"

MODELS = {
    "llama3": "llama3:latest",
    "mistral": "mistral:latest"
}

PAPERS = ["CS1", "CS2", "CS3"]

PROSPECTIVE_STANCES = ["conservative", "discovery-oriented", "evidence-only"]

N_REPETITIONS = 5

# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE INPUTS (structured JSON per paper)
# Load from gold standard files
# ─────────────────────────────────────────────────────────────────────────────

def load_gold_standard(paper_id):
    path = os.path.join(GOLD_STANDARD_DIR, f"gold_standard_{paper_id}.json")
    with open(path, "r") as f:
        return json.load(f)

def build_evidence_input(gold):
    """Extract the evidence JSON block to present to agents."""
    return {
        "analyticalContext": gold["analyticalContext"],
        "evidenceItems": gold["evidenceItems"]
    }

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FORMAT = """{
  "ScientificClaim": {
    "claimText": "...",
    "claimScope": "exploratory|candidate|validated",
    "confidence": 0.0
  },
  "EvidenceItemsUsed": ["EI-XXX-01", "EI-XXX-02"],
  "RationaleArtifact": {
    "rationaleText": "...",
    "keyEvidenceSummary": "...",
    "uncertainties": ["...", "..."]
  }
}"""

PROSPECTIVE_PROMPTS = {
    "conservative": """You are a scientific analyst. You have been given the analytical results from a published study.

Using ONLY the evidence items provided below, produce a structured scientific interpretation.
Be cautious in your claims. State uncertainties explicitly. Do not make clinical claims.
Do not extrapolate beyond what the evidence directly supports.

Evidence:
{evidence}

Respond ONLY in the following JSON format, with no preamble or explanation:
{output_format}""",

    "discovery-oriented": """You are a scientific analyst identifying novel findings.
You have been given analytical results from a published study.

Using the evidence items provided below, identify the most scientifically significant findings
and their potential implications for the field.

Evidence:
{evidence}

Respond ONLY in the following JSON format, with no preamble or explanation:
{output_format}""",

    "evidence-only": """You are a scientific analyst. Produce a minimal scientific interpretation
based strictly on the numerical evidence provided.
Do not add any biological context, prior knowledge, or information beyond what is in the evidence items.

Evidence:
{evidence}

Respond ONLY in the following JSON format, with no preamble or explanation:
{output_format}"""
}

RETROSPECTIVE_PROMPT = """You are a scientific analyst. You have been given the analytical results
from a published study, together with the human reasoning trace that documents how the
original authors moved from evidence to conclusion.

Your task is to reconstruct a provenance record that is faithful to the human reasoning
documented in the trace. Use only evidence items referenced in the trace.
Produce a claim consistent with the published conclusion.

Evidence:
{evidence}

Human Reasoning Trace:
{human_trace}

Respond ONLY in the following JSON format, with no preamble or explanation:
{output_format}"""

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA API CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(model_name, prompt, max_retries=3):
    """Call Ollama API with greedy decoding (temperature=0)."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 2048
        }
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# JSON PARSING WITH FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_output(raw_text):
    """Attempt to parse JSON from model output, with cleanup and truncation recovery."""
    if raw_text is None:
        return None
    # Strip markdown code fences if present
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    # Strip any preamble before the first {
    start = raw_text.find("{")
    if start == -1:
        return None
    candidate = raw_text[start:]
    # Try parsing as-is first
    try:
        end = candidate.rfind("}") + 1
        return json.loads(candidate[:end])
    except json.JSONDecodeError:
        pass
    # Try to recover truncated JSON by adding missing closing braces
    open_braces = candidate.count("{") - candidate.count("}")
    if open_braces > 0:
        candidate_fixed = candidate + ("}" * open_braces)
        try:
            return json.loads(candidate_fixed)
        except json.JSONDecodeError:
            pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILE NAMING
# ─────────────────────────────────────────────────────────────────────────────

def output_path(paper, mode, stance, agent_key, rep):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if mode == "prospective":
        filename = f"{paper}_{mode}_{stance}_{agent_key}_rep{rep}.json"
    else:
        filename = f"{paper}_{mode}_{agent_key}_rep{rep}.json"
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single(paper_id, mode, stance, agent_key, rep, gold, evidence_input):
    out_path = output_path(paper_id, mode, stance, agent_key, rep)

    if os.path.exists(out_path):
        print(f"  Skipping (already exists): {os.path.basename(out_path)}")
        return True

    model_name = MODELS[agent_key]
    evidence_str = json.dumps(evidence_input, indent=2)

    if mode == "prospective":
        prompt = PROSPECTIVE_PROMPTS[stance].format(
            evidence=evidence_str,
            output_format=OUTPUT_FORMAT
        )
    else:
        human_trace_str = json.dumps(gold["humanReasoningTrace"], indent=2)
        prompt = RETROSPECTIVE_PROMPT.format(
            evidence=evidence_str,
            human_trace=human_trace_str,
            output_format=OUTPUT_FORMAT
        )

    print(f"  Running: {os.path.basename(out_path)}")
    start_time = datetime.utcnow().isoformat()
    raw = call_ollama(model_name, prompt)
    end_time = datetime.utcnow().isoformat()

    parsed = parse_json_output(raw)

    result = {
        "metadata": {
            "paper": paper_id,
            "mode": mode,
            "analyticalStance": stance if mode == "prospective" else "N/A (retrospective)",
            "agentKey": agent_key,
            "modelName": model_name,
            "repetition": rep,
            "reconstructionMode": mode,
            "startTime": start_time,
            "endTime": end_time
        },
        "rawOutput": raw,
        "parsedOutput": parsed,
        "parseSuccess": parsed is not None
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "OK" if parsed is not None else "PARSE FAILED"
    print(f"    -> {status}")
    return parsed is not None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    successes = 0
    failures = []

    for paper_id in PAPERS:
        print(f"\n{'='*60}")
        print(f"Paper: {paper_id}")
        print(f"{'='*60}")

        gold = load_gold_standard(paper_id)
        evidence_input = build_evidence_input(gold)

        # Prospective runs: 3 stances x 2 agents x 5 reps = 30 per paper
        for stance in PROSPECTIVE_STANCES:
            for agent_key in MODELS:
                for rep in range(1, N_REPETITIONS + 1):
                    total += 1
                    ok = run_single(
                        paper_id, "prospective", stance,
                        agent_key, rep, gold, evidence_input
                    )
                    if ok:
                        successes += 1
                    else:
                        failures.append(f"{paper_id}_prospective_{stance}_{agent_key}_rep{rep}")
                    time.sleep(1)  # brief pause between calls

        # Retrospective runs: 2 agents x 5 reps = 10 per paper
        for agent_key in MODELS:
            for rep in range(1, N_REPETITIONS + 1):
                total += 1
                ok = run_single(
                    paper_id, "retrospective", None,
                    agent_key, rep, gold, evidence_input
                )
                if ok:
                    successes += 1
                else:
                    failures.append(f"{paper_id}_retrospective_{agent_key}_rep{rep}")
                time.sleep(1)

    print(f"\n{'='*60}")
    print(f"COMPLETED: {successes}/{total} runs successful")
    if failures:
        print(f"FAILED RUNS ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    print(f"Outputs saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
