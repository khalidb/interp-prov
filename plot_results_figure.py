"""
Regenerates results_figure.pdf / results_figure.png for the eScience 2026
paper ("An Interpretive Provenance Model for Scientific Reasoning").

Inputs:
    evaluation_summary.csv   -- EC1 (ec1_jaccard), EC3 (ec3_score), EC6 (ec6_score)
    human_eval_sheet.json    -- EC2 (human-scored claim equivalence)

Aggregation (matches the values in the paper's Results section):
    - Prospective mode: mean over all 3 analytical stances x 5 repetitions (15 runs)
    - Retrospective mode: mean over 5 repetitions
    - EC6 is plotted as the TRUE OVERCLAIM RATE (1 - ec6_score), so that
      0 = no runs overclaimed (good) and 1 = every run overclaimed (bad),
      matching the "lower is better" framing used throughout results.tex.
      (ec6_score itself is defined in evaluate.py as 1.0 = no overclaim,
      0.0 = overclaim detected -- i.e. the inverse of what we want to plot.)

Font embedding: pdf.fonttype and ps.fonttype are set to 42 (TrueType) so
that text in the exported PDF is NOT embedded as Type 3 fonts, which
IEEE PDF eXpress / Xplore reject. Verify with:
    pdffonts results_figure.pdf
No line in the output should say "Type 3".
"""

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType, not Type 3
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "evaluation_summary.csv"
HEVAL_PATH = "human_eval_sheet.json"

PAPERS = ["CS1", "CS2", "CS3"]
CONDITIONS = [
    ("prospective", "llama3", "Pro-L3"),
    ("prospective", "mistral", "Pro-M7"),
    ("retrospective", "llama3", "Ret-L3"),
    ("retrospective", "mistral", "Ret-M7"),
]
COLORS = {
    "Pro-L3": "#4C72B0",
    "Pro-M7": "#55A88C",
    "Ret-L3": "#E07B1A",
    "Ret-M7": "#F2B366",
}
LEGEND_LABELS = {
    "Pro-L3": "Prospective LLaMA 3",
    "Pro-M7": "Prospective Mistral",
    "Ret-L3": "Retrospective LLaMA 3",
    "Ret-M7": "Retrospective Mistral",
}


def load_metrics():
    rows = list(csv.DictReader(open(CSV_PATH)))
    heval = json.load(open(HEVAL_PATH))

    def agg_csv(field):
        out = defaultdict(list)
        for r in rows:
            key = (r["paper"], r["mode"], r["agent"])
            out[key].append(float(r[field]))
        return {k: sum(v) / len(v) for k, v in out.items()}

    ec1 = agg_csv("ec1_jaccard")
    ec3 = agg_csv("ec3_score")
    ec6_score = agg_csv("ec6_score")
    ec6 = {k: 1.0 - v for k, v in ec6_score.items()}  # true overclaim rate

    ec2_raw = defaultdict(list)
    for e in heval:
        key = (e["paper"], e["mode"], e["agent"])
        ec2_raw[key].append(e["EC2"]["score"])
    ec2 = {k: sum(v) / len(v) for k, v in ec2_raw.items()}

    return ec1, ec2, ec3, ec6


def main():
    ec1, ec2, ec3, ec6 = load_metrics()

    metrics = [
        ("EC1", "Evidence coverage", ec1, False),
        ("EC2", "Claim equivalence", ec2, False),
        ("EC3", "Scope calibration", ec3, False),
        ("EC6", "Overclaim rate", ec6, True),  # True = "lower is better"
    ]

    fig, axes = plt.subplots(
        len(PAPERS), len(metrics), figsize=(13, 9), sharey=True
    )

    x = np.arange(len(CONDITIONS))
    bar_labels = [c[2] for c in CONDITIONS]

    for row, paper in enumerate(PAPERS):
        for col, (code, title, data, lower_better) in enumerate(metrics):
            ax = axes[row, col]
            heights = [data[(paper, mode, agent)] for mode, agent, _ in CONDITIONS]
            colors = [COLORS[lbl] for lbl in bar_labels]
            ax.bar(x, heights, color=colors, width=0.6)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, rotation=35, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row == 0:
                header = f"{code}\n{title}"
                ax.set_title(header, fontsize=11, fontweight="bold", pad=12)
                if lower_better:
                    ax.text(
                        0.5, 1.30, "(\u2193 better)",
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=9, color="firebrick",
                    )
            if col == 0:
                ax.set_ylabel(paper, fontsize=12, fontweight="bold", rotation=0,
                               labelpad=28, va="center")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[lbl]) for lbl in bar_labels]
    labels = [LEGEND_LABELS[lbl] for lbl in bar_labels]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig("results_figure.pdf", bbox_inches="tight")
    fig.savefig("results_figure.png", dpi=150, bbox_inches="tight")
    print("Wrote results_figure.pdf and results_figure.png")


if __name__ == "__main__":
    main()
