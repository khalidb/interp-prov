"""
Results summary script.
Reads evaluation_results.json and human_eval_sheet.json
and prints a readable summary for analysis.

Usage: python3 summarise_results.py
"""

import json
import os
from collections import defaultdict

RESULTS_FILE = "./evaluation_results.json"
SHEET_FILE = "./human_eval_sheet.json"
SUMMARY_FILE = "./evaluation_summary.csv"

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None

def print_table(headers, rows, col_width=12):
    fmt = " | ".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt.format(*[str(v) if v is not None else "-" for v in row]))

def main():
    results = load_json(RESULTS_FILE)
    sheet = load_json(SHEET_FILE)

    if results is None:
        print("evaluation_results.json not found. Run evaluate.py first.")
        return

    per_run = results["perRun"]
    ec7 = results.get("ec7", {})

    # ── Merge human eval scores if available ────────────────────────────────
    human_scores = {}
    if sheet:
        for entry in sheet:
            human_scores[entry["runId"]] = {
                "ec2": entry["EC2"]["score"],
                "ec4": entry["EC4"]["score"]
            }

    # ── Build flat records ───────────────────────────────────────────────────
    records = []
    for run_id, r in per_run.items():
        meta = r["metadata"]
        rec = {
            "runId": run_id,
            "paper": meta["paper"],
            "mode": meta["mode"],
            "stance": meta.get("analyticalStance", "N/A"),
            "agent": meta["agentKey"],
            "rep": meta["repetition"],
            "ec1": r["ec1"].get("score"),
            "ec3": r["ec3"].get("score"),
            "ec5": r["ec5"].get("score"),
            "ec6": r["ec6"].get("score"),
            "ec2": human_scores.get(run_id, {}).get("ec2"),
            "ec4": human_scores.get(run_id, {}).get("ec4"),
            "agentScope": r["ec3"].get("agentScope"),
            "overclaim": r["ec6"].get("overclaim"),
            "confidence": None
        }
        records.append(rec)

    # ── EC1: Evidence coverage by paper and mode ─────────────────────────────
    print("\n══ EC1: Evidence Coverage (Jaccard) ══════════════════════════════")
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["paper"], rec["mode"], rec["agent"])].append(rec["ec1"])

    rows = []
    for (paper, mode, agent), scores in sorted(groups.items()):
        rows.append([paper, mode, agent, mean(scores)])
    print_table(["Paper", "Mode", "Agent", "Mean Jaccard"], rows)

    # ── EC3: Claim scope calibration ─────────────────────────────────────────
    print("\n══ EC3: Claim Scope Calibration ══════════════════════════════════")
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["paper"], rec["mode"], rec["agent"])].append(rec["ec3"])
    rows = []
    for (paper, mode, agent), scores in sorted(groups.items()):
        rows.append([paper, mode, agent, mean(scores)])
    print_table(["Paper", "Mode", "Agent", "Mean Score"], rows)

    # ── EC3: Scope distribution ───────────────────────────────────────────────
    print("\n── Scope distribution (prospective) ──")
    scope_counts = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if rec["mode"] == "prospective" and rec["agentScope"]:
            scope_counts[(rec["paper"], rec["agent"])][rec["agentScope"]] += 1
    for (paper, agent), counts in sorted(scope_counts.items()):
        print(f"  {paper} {agent}: {dict(counts)}")

    # ── EC5: Uncertainty calibration ─────────────────────────────────────────
    print("\n══ EC5: Uncertainty Calibration ══════════════════════════════════")
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["paper"], rec["mode"], rec["agent"])].append(rec["ec5"])
    rows = []
    for (paper, mode, agent), scores in sorted(groups.items()):
        rows.append([paper, mode, agent, mean(scores)])
    print_table(["Paper", "Mode", "Agent", "Mean Score"], rows)

    # ── EC6: Overclaim detection ──────────────────────────────────────────────
    print("\n══ EC6: Overclaim Detection ══════════════════════════════════════")
    groups = defaultdict(lambda: {"overclaim": 0, "total": 0})
    for rec in records:
        key = (rec["paper"], rec["mode"], rec["agent"])
        groups[key]["total"] += 1
        if rec["overclaim"]:
            groups[key]["overclaim"] += 1
    rows = []
    for (paper, mode, agent), v in sorted(groups.items()):
        rate = round(v["overclaim"] / v["total"], 3) if v["total"] > 0 else None
        rows.append([paper, mode, agent, v["overclaim"], v["total"], rate])
    print_table(["Paper", "Mode", "Agent", "Overclaims", "Total", "Rate"], rows)

    # ── EC2 and EC4 (human eval) ──────────────────────────────────────────────
    if any(r["ec2"] is not None for r in records):
        print("\n══ EC2: Claim Equivalence (human) ════════════════════════════════")
        groups = defaultdict(list)
        for rec in records:
            if rec["ec2"] is not None:
                groups[(rec["paper"], rec["mode"], rec["agent"])].append(rec["ec2"])
        rows = []
        for (paper, mode, agent), scores in sorted(groups.items()):
            rows.append([paper, mode, agent, mean(scores)])
        print_table(["Paper", "Mode", "Agent", "Mean Score"], rows)

    if any(r["ec4"] is not None for r in records):
        print("\n══ EC4: Rationale Alignment (human) ══════════════════════════════")
        groups = defaultdict(list)
        for rec in records:
            if rec["ec4"] is not None:
                groups[(rec["paper"], rec["mode"], rec["agent"])].append(rec["ec4"])
        rows = []
        for (paper, mode, agent), scores in sorted(groups.items()):
            rows.append([paper, mode, agent, mean(scores)])
        print_table(["Paper", "Mode", "Agent", "Mean Score"], rows)
    else:
        print("\n══ EC2/EC4: Human evaluation not yet complete ════════════════════")

    # ── EC7: Cross-agent consistency ──────────────────────────────────────────
    if ec7:
        print("\n══ EC7: Cross-Agent Consistency ══════════════════════════════════")
        for paper in ["CS1", "CS2", "CS3"]:
            print(f"\n  {paper}:")
            for stance, v in ec7[paper]["crossAgent"].items():
                print(f"    {stance}: agreement rate = {v['agreementRate']}")
            print(f"  Cross-stance (conservative more cautious than discovery-oriented):")
            for agent, v in ec7[paper]["crossStance"].items():
                print(f"    {agent}: {v['conservativeMoreCautious']}")

    print("\nDone.")

if __name__ == "__main__":
    main()
