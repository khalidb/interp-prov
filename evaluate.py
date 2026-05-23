"""
Automated evaluation script for:
"An Interpretive Provenance Model for Scientific Reasoning:
 A Computational Biology Study"

Evaluates EC1, EC3, EC5, EC6, EC7 automatically.
EC2 and EC4 require human evaluation (see human_eval_sheet.py).

Outputs:
- evaluation_results.json: full results per run
- evaluation_summary.csv: summary table for analysis
"""

import json
import os
import csv
from pathlib import Path

OUTPUT_DIR = "./outputs"
GOLD_DIR = "./gold_standards"
RESULTS_FILE = "./evaluation_results.json"
SUMMARY_FILE = "./evaluation_summary.csv"

# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED CLAIM SCOPES PER PAPER (from methods section)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_SCOPE = {
    "CS1": "candidate",
    "CS2": "exploratory",
    "CS3": "candidate"
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED UNCERTAINTIES PER PAPER (keywords that should appear)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_UNCERTAINTY_KEYWORDS = {
    "CS1": ["validation", "clinical", "generali"],
    "CS2": ["sample size", "generaliz", "TCGA", "replicate"],
    "CS3": ["validation", "experimental", "sample size", "somatic", "germline"]
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_gold(paper_id):
    with open(os.path.join(GOLD_DIR, f"gold_standard_{paper_id}.json")) as f:
        return json.load(f)

def load_output(filepath):
    with open(filepath) as f:
        return json.load(f)

def get_gold_evidence_ids(gold):
    return set(e["evidenceId"] for e in gold["evidenceItems"])

# ─────────────────────────────────────────────────────────────────────────────
# EC1 — Evidence Coverage (Jaccard overlap)
# ─────────────────────────────────────────────────────────────────────────────

def eval_ec1(parsed_output, gold_evidence_ids):
    """Jaccard overlap between agent evidence selection and gold standard."""
    if parsed_output is None:
        return {"score": None, "detail": "parse failed"}

    agent_ids = set(parsed_output.get("EvidenceItemsUsed", []))
    if not agent_ids and not gold_evidence_ids:
        return {"score": 1.0, "detail": "both empty"}

    intersection = len(agent_ids & gold_evidence_ids)
    union = len(agent_ids | gold_evidence_ids)
    jaccard = intersection / union if union > 0 else 0.0

    return {
        "score": round(jaccard, 3),
        "agentSelected": sorted(agent_ids),
        "goldStandard": sorted(gold_evidence_ids),
        "intersection": sorted(agent_ids & gold_evidence_ids),
        "missing": sorted(gold_evidence_ids - agent_ids),
        "extra": sorted(agent_ids - gold_evidence_ids)
    }

# ─────────────────────────────────────────────────────────────────────────────
# EC3 — Claim Scope Calibration
# ─────────────────────────────────────────────────────────────────────────────

def eval_ec3(parsed_output, paper_id):
    """Whether agent correctly scoped the claim."""
    if parsed_output is None:
        return {"score": None, "detail": "parse failed"}

    claim = parsed_output.get("ScientificClaim", {})
    agent_scope = claim.get("claimScope", "").lower().strip()
    expected_scope = EXPECTED_SCOPE[paper_id]

    # Score: 1.0 = correct, 0.5 = adjacent (e.g. candidate vs exploratory), 0.0 = far off
    scope_order = ["exploratory", "candidate", "validated"]
    if agent_scope == expected_scope:
        score = 1.0
    elif agent_scope in scope_order and expected_scope in scope_order:
        distance = abs(scope_order.index(agent_scope) - scope_order.index(expected_scope))
        score = 0.5 if distance == 1 else 0.0
    else:
        score = 0.0

    return {
        "score": score,
        "agentScope": agent_scope,
        "expectedScope": expected_scope,
        "correct": agent_scope == expected_scope
    }

# ─────────────────────────────────────────────────────────────────────────────
# EC5 — Uncertainty Calibration
# ─────────────────────────────────────────────────────────────────────────────

def eval_ec5(parsed_output, paper_id):
    """Whether agent stated appropriate uncertainties."""
    if parsed_output is None:
        return {"score": None, "detail": "parse failed"}

    rationale = parsed_output.get("RationaleArtifact", {})
    uncertainties = rationale.get("uncertainties", [])

    if not uncertainties:
        return {"score": 0.0, "detail": "no uncertainties stated", "found": []}

    # Check how many expected keywords appear across all uncertainty statements
    all_uncertainty_text = " ".join(uncertainties).lower()
    keywords = EXPECTED_UNCERTAINTY_KEYWORDS[paper_id]
    found = [kw for kw in keywords if kw.lower() in all_uncertainty_text]
    score = round(len(found) / len(keywords), 3)

    return {
        "score": score,
        "keywordsFound": found,
        "keywordsMissed": [kw for kw in keywords if kw not in found],
        "uncertaintiesStated": uncertainties
    }

# ─────────────────────────────────────────────────────────────────────────────
# EC6 — Overclaim Detection
# ─────────────────────────────────────────────────────────────────────────────

def eval_ec6(parsed_output, paper_id):
    """Whether agent made claims stronger than evidence supports."""
    if parsed_output is None:
        return {"score": None, "detail": "parse failed"}

    claim = parsed_output.get("ScientificClaim", {})
    agent_scope = claim.get("claimScope", "").lower().strip()
    expected_scope = EXPECTED_SCOPE[paper_id]

    scope_order = ["exploratory", "candidate", "validated"]

    overclaim = False
    overclaim_degree = 0

    if agent_scope in scope_order and expected_scope in scope_order:
        agent_idx = scope_order.index(agent_scope)
        expected_idx = scope_order.index(expected_scope)
        if agent_idx > expected_idx:
            overclaim = True
            overclaim_degree = agent_idx - expected_idx

    # Also check confidence: flag if confidence > 0.9 for exploratory papers
    confidence = claim.get("confidence", 0.0)
    confidence_overclaim = (expected_scope == "exploratory" and confidence > 0.9)

    # Score: 1.0 = no overclaim, 0.0 = overclaim detected
    score = 0.0 if (overclaim or confidence_overclaim) else 1.0

    return {
        "score": score,
        "overclaim": overclaim or confidence_overclaim,
        "agentScope": agent_scope,
        "expectedScope": expected_scope,
        "overclamDegree": overclaim_degree,
        "agentConfidence": confidence,
        "confidenceOverclaim": confidence_overclaim
    }

# ─────────────────────────────────────────────────────────────────────────────
# EC7 — Cross-Agent and Cross-Stance Consistency
# (computed after all runs are evaluated)
# ─────────────────────────────────────────────────────────────────────────────

def eval_ec7(all_results):
    """
    Cross-agent consistency: do LLaMA and Mistral reach equivalent
    claim scopes under the same stance?
    Cross-stance consistency: does conservative produce more cautious
    claims than discovery-oriented?
    """
    ec7 = {}

    for paper in ["CS1", "CS2", "CS3"]:
        ec7[paper] = {
            "crossAgent": {},
            "crossStance": {}
        }

        # Cross-agent consistency per stance
        for stance in ["conservative", "discovery-oriented", "evidence-only"]:
            llama_scopes = []
            mistral_scopes = []
            for rep in range(1, 6):
                key_llama = f"{paper}_prospective_{stance}_llama3_rep{rep}"
                key_mistral = f"{paper}_prospective_{stance}_mistral_rep{rep}"
                if key_llama in all_results and key_mistral in all_results:
                    ls = all_results[key_llama].get("ec3", {}).get("agentScope")
                    ms = all_results[key_mistral].get("ec3", {}).get("agentScope")
                    if ls:
                        llama_scopes.append(ls)
                    if ms:
                        mistral_scopes.append(ms)

            # Agreement rate: how often do both agents produce the same scope
            agreements = sum(1 for l, m in zip(llama_scopes, mistral_scopes) if l == m)
            total = len(list(zip(llama_scopes, mistral_scopes)))
            agreement_rate = round(agreements / total, 3) if total > 0 else None

            ec7[paper]["crossAgent"][stance] = {
                "agreementRate": agreement_rate,
                "llamaScopes": llama_scopes,
                "mistralScopes": mistral_scopes
            }

        # Cross-stance: conservative vs discovery-oriented
        scope_order = ["exploratory", "candidate", "validated"]
        for agent in ["llama3", "mistral"]:
            cons_scopes = []
            disc_scopes = []
            for rep in range(1, 6):
                key_cons = f"{paper}_prospective_conservative_{agent}_rep{rep}"
                key_disc = f"{paper}_prospective_discovery-oriented_{agent}_rep{rep}"
                if key_cons in all_results and key_disc in all_results:
                    cs = all_results[key_cons].get("ec3", {}).get("agentScope")
                    ds = all_results[key_disc].get("ec3", {}).get("agentScope")
                    if cs:
                        cons_scopes.append(cs)
                    if ds:
                        disc_scopes.append(ds)

            # Does conservative produce lower or equal scope than discovery-oriented?
            stance_consistent = 0
            total = len(list(zip(cons_scopes, disc_scopes)))
            for cs, ds in zip(cons_scopes, disc_scopes):
                if cs in scope_order and ds in scope_order:
                    if scope_order.index(cs) <= scope_order.index(ds):
                        stance_consistent += 1

            ec7[paper]["crossStance"][agent] = {
                "conservativeMoreCautious": round(stance_consistent / total, 3) if total > 0 else None,
                "conservativeScopes": cons_scopes,
                "discoveryScopes": disc_scopes
            }

    return ec7

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    output_files = sorted(Path(OUTPUT_DIR).glob("*.json"))
    print(f"Found {len(output_files)} output files.")

    all_results = {}
    rows = []  # for CSV summary

    for filepath in output_files:
        run_id = filepath.stem
        data = load_output(filepath)
        meta = data["metadata"]
        parsed = data.get("parsedOutput")

        paper = meta["paper"]
        gold = load_gold(paper)
        gold_evidence_ids = get_gold_evidence_ids(gold)

        ec1 = eval_ec1(parsed, gold_evidence_ids)
        ec3 = eval_ec3(parsed, paper)
        ec5 = eval_ec5(parsed, paper)
        ec6 = eval_ec6(parsed, paper)

        all_results[run_id] = {
            "metadata": meta,
            "ec1": ec1,
            "ec3": ec3,
            "ec5": ec5,
            "ec6": ec6
        }

        rows.append({
            "runId": run_id,
            "paper": paper,
            "mode": meta["mode"],
            "analyticalStance": meta.get("analyticalStance", "N/A"),
            "agent": meta["agentKey"],
            "repetition": meta["repetition"],
            "ec1_jaccard": ec1.get("score"),
            "ec1_missing": "|".join(ec1.get("missing", [])),
            "ec1_extra": "|".join(ec1.get("extra", [])),
            "ec3_score": ec3.get("score"),
            "ec3_agentScope": ec3.get("agentScope"),
            "ec3_correct": ec3.get("correct"),
            "ec5_score": ec5.get("score"),
            "ec5_keywordsFound": "|".join(ec5.get("keywordsFound", [])),
            "ec6_score": ec6.get("score"),
            "ec6_overclaim": ec6.get("overclaim"),
            "ec6_overclamDegree": ec6.get("overclamDegree"),
            "agentConfidence": parsed.get("ScientificClaim", {}).get("confidence") if parsed else None,
            "agentClaimText": parsed.get("ScientificClaim", {}).get("claimText", "")[:100] if parsed else ""
        })

        print(f"  {run_id}: EC1={ec1.get('score'):.3f} EC3={ec3.get('score')} EC5={ec5.get('score'):.3f} EC6={'OK' if ec6.get('score')==1.0 else 'OVERCLAIM'}")

    # EC7: cross-agent and cross-stance consistency
    print("\nComputing EC7 (cross-agent and cross-stance consistency)...")
    ec7_results = eval_ec7(all_results)

    # Save full results
    final = {
        "perRun": all_results,
        "ec7": ec7_results
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_FILE}")

    # Save CSV summary
    if rows:
        with open(SUMMARY_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV saved to: {SUMMARY_FILE}")

    # Print EC7 summary
    print("\n── EC7: Cross-Agent Consistency ──────────────────────────")
    for paper in ["CS1", "CS2", "CS3"]:
        print(f"\n{paper}:")
        for stance, v in ec7_results[paper]["crossAgent"].items():
            print(f"  {stance}: agreement rate = {v['agreementRate']}")
        print(f"  Cross-stance (conservative more cautious):")
        for agent, v in ec7_results[paper]["crossStance"].items():
            print(f"    {agent}: {v['conservativeMoreCautious']}")

if __name__ == "__main__":
    main()
