"""Analyze judged SelfAware responses.

H5: Models dodge on unanswerable questions more than they refuse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llm_client import MODEL_TIER  # noqa: E402

JUDGED_DIR = ROOT / "results" / "judged_selfaware"
RAW_DIR = ROOT / "results" / "raw" / "selfaware"
OUT_DIR = ROOT / "results" / "analysis"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ("DIRECT_ANSWER", "HEDGED_ATTEMPT", "LOOPHOLE_DODGE", "REFUSAL")
COLORS = {
    "DIRECT_ANSWER": "#2E7D32",
    "HEDGED_ATTEMPT": "#9CCC65",
    "LOOPHOLE_DODGE": "#EF6C00",
    "REFUSAL": "#1565C0",
}


def load() -> pd.DataFrame:
    rows = []
    for path in sorted(JUDGED_DIR.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["tier"] = df["model_short"].map(MODEL_TIER)
    return df


def main():
    df = load()
    if df.empty:
        print("No SelfAware judged data; exiting.")
        return
    summary = df.groupby(["model_short", "answerable", "judge_label"]).size().unstack(fill_value=0)
    for col in LABELS:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[list(LABELS)]
    summary["total"] = summary.sum(axis=1)
    for col in LABELS:
        summary[f"{col}_pct"] = summary[col] / summary["total"]
    summary.to_csv(OUT_DIR / "selfaware_breakdown.csv")
    print(f"selfaware breakdown -> {OUT_DIR / 'selfaware_breakdown.csv'}")

    # H5: For unanswerable items, compare DODGE rate vs REFUSAL rate
    rows = []
    for model, grp in df.groupby("model_short"):
        sub_un = grp[~grp["answerable"]]
        sub_an = grp[grp["answerable"]]
        rows.append(
            {
                "model": model,
                "tier": MODEL_TIER[model],
                "n_unanswerable": len(sub_un),
                "n_answerable": len(sub_an),
                "unans_direct": (sub_un["judge_label"] == "DIRECT_ANSWER").mean(),
                "unans_dodge": (sub_un["judge_label"] == "LOOPHOLE_DODGE").mean(),
                "unans_refusal": (sub_un["judge_label"] == "REFUSAL").mean(),
                "unans_hedged": (sub_un["judge_label"] == "HEDGED_ATTEMPT").mean(),
                "ans_direct": (sub_an["judge_label"] == "DIRECT_ANSWER").mean(),
                "ans_dodge": (sub_an["judge_label"] == "LOOPHOLE_DODGE").mean(),
                "ans_refusal": (sub_an["judge_label"] == "REFUSAL").mean(),
                "ans_hedged": (sub_an["judge_label"] == "HEDGED_ATTEMPT").mean(),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "selfaware_summary.csv", index=False)
    print(summary_df.round(3).to_string(index=False))

    # χ² 2×4
    from scipy.stats import chi2_contingency

    chi_rows = []
    for model, grp in df.groupby("model_short"):
        table = grp.groupby(["answerable", "judge_label"]).size().unstack(fill_value=0)
        for col in LABELS:
            if col not in table.columns:
                table[col] = 0
        table = table[list(LABELS)]
        try:
            chi2, p, dof, _ = chi2_contingency(table.values)
            n = table.values.sum()
            cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if n else 0.0
        except Exception as e:
            chi2, p, cramers_v = float("nan"), 1.0, float("nan")
        chi_rows.append(
            {"model": model, "tier": MODEL_TIER[model], "chi2": chi2, "p": p, "cramers_v": cramers_v}
        )
    pd.DataFrame(chi_rows).to_csv(OUT_DIR / "selfaware_chi2.csv", index=False)
    print("chi2 ->", OUT_DIR / "selfaware_chi2.csv")

    # Figure: stacked bar
    models = sorted(df["model_short"].unique(), key=lambda m: ["frontier", "large_open", "small_open", "reasoning"].index(MODEL_TIER[m]) if MODEL_TIER[m] in ["frontier", "large_open", "small_open", "reasoning"] else 99)
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub_un = (df[(df["model_short"] == m) & (~df["answerable"])]["judge_label"].value_counts(normalize=True) * 100)
        sub_an = (df[(df["model_short"] == m) & (df["answerable"])]["judge_label"].value_counts(normalize=True) * 100)
        # Reindex first, then fillna, in correct order
        ans = pd.DataFrame({"answerable": sub_an, "unanswerable": sub_un})
        ans = ans.reindex(list(LABELS)).fillna(0)
        bottom = np.zeros(2)
        for lbl in LABELS:
            v = ans.loc[lbl, ["answerable", "unanswerable"]].astype(float).values
            ax.bar([0, 1], v, bottom=bottom, color=COLORS[lbl], label=lbl, edgecolor="white", linewidth=0.5)
            bottom += v
        ax.set_title(f"{m}\n({MODEL_TIER[m]})", fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["answerable", "unanswerable"], rotation=0)
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.set_ylabel("% of responses")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Judge label")
    plt.suptitle("SelfAware: response category by model and answerability", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "selfaware_stacked_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"selfaware figure -> {FIG_DIR / 'selfaware_stacked_bar.png'}")


if __name__ == "__main__":
    main()
