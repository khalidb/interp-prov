"""
Human evaluation sheet generator for EC2 (claim equivalence)
and EC4 (rationale alignment).

Generates a structured JSON file with all comparisons
that require human assessment, ready to be filled in.

Usage:
  python3 human_eval_sheet.py          # generates the sheet
  python3 human_eval_sheet.py --check  # checks completion status
"""

import json
import os
import sys
from pathlib import Path

OUTPUT_DIR = "./outputs"
GOLD_DIR = "./gold_standards"
SHEET_FILE = "./human_eval_sheet.json"

def load_gold(paper_id):
    with open(os.path.join(GOLD_DIR, f"gold_standard_{paper_id}.json")) as f:
        return json.load(f)

def load_output(filepath):
    with open(filepath) as f:
        return json.load(f)

def generate_sheet():
    output_files = sorted(Path(OUTPUT_DIR).glob("*.json"))
    entries = []

    for filepath in output_files:
        run_id = filepath.stem
        data = load_output(filepath)
        meta = data["metadata"]
        parsed = data.get("parsedOutput")

        if parsed is None:
            continue

        paper = meta["paper"]
        gold = load_gold(paper)

        gold_claim = gold["scientificClaim"]["claimText"]
        gold_rationale = gold["rationaleArtifact"]["rationaleText"]
        gold_trace_steps = [
            s["stepDescription"]
            for s in gold["humanReasoningTrace"]["steps"]
        ]

        agent_claim = parsed.get("ScientificClaim", {}).get("claimText", "")
        agent_rationale = parsed.get("RationaleArtifact", {}).get("rationaleText", "")

        entries.append({
            "runId": run_id,
            "paper": paper,
            "mode": meta["mode"],
            "analyticalStance": meta.get("analyticalStance", "N/A"),
            "agent": meta["agentKey"],
            "repetition": meta["repetition"],

            "EC2": {
                "criterion": "Semantic equivalence of agent claim to gold standard claim",
                "goldClaim": gold_claim,
                "agentClaim": agent_claim,
                "score": None,
                "score_options": [
                    "2 = fully equivalent (same core assertion)",
                    "1 = partially equivalent (overlapping but different scope or emphasis)",
                    "0 = not equivalent (different assertion)"
                ],
                "notes": ""
            },

            "EC4": {
                "criterion": "Alignment of agent rationale with human reasoning trace",
                "goldRationaleSteps": gold_trace_steps,
                "agentRationale": agent_rationale,
                "score": None,
                "score_options": [
                    "2 = closely aligned (agent captures same inferential steps)",
                    "1 = partially aligned (some steps present, some missing or divergent)",
                    "0 = not aligned (different reasoning path)"
                ],
                "notes": ""
            }
        })

    with open(SHEET_FILE, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"Generated human evaluation sheet with {len(entries)} entries.")
    print(f"Saved to: {SHEET_FILE}")
    print(f"\nFor each entry, fill in:")
    print(f"  EC2.score: 0, 1, or 2")
    print(f"  EC4.score: 0, 1, or 2")
    print(f"  notes: any relevant observations")

def check_completion():
    if not os.path.exists(SHEET_FILE):
        print("Sheet not found. Run without --check first.")
        return

    with open(SHEET_FILE) as f:
        entries = json.load(f)

    total = len(entries)
    ec2_done = sum(1 for e in entries if e["EC2"]["score"] is not None)
    ec4_done = sum(1 for e in entries if e["EC4"]["score"] is not None)

    print(f"Total entries: {total}")
    print(f"EC2 completed: {ec2_done}/{total}")
    print(f"EC4 completed: {ec4_done}/{total}")

    incomplete = [e["runId"] for e in entries
                  if e["EC2"]["score"] is None or e["EC4"]["score"] is None]
    if incomplete:
        print(f"\nIncomplete ({len(incomplete)}):")
        for r in incomplete[:10]:
            print(f"  - {r}")
        if len(incomplete) > 10:
            print(f"  ... and {len(incomplete)-10} more")

if __name__ == "__main__":
    if "--check" in sys.argv:
        check_completion()
    else:
        generate_sheet()
