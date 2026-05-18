"""Analysis: load judged KCL outputs, compute metrics, run statistical tests, save figures."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from llm_client import MODEL_TIER  # noqa: E402

JUDGED_DIR = ROOT / "results" / "judged"
OUT_DIR = ROOT / "results" / "analysis"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ("DIRECT_ANSWER", "HEDGED_ATTEMPT", "LOOPHOLE_DODGE", "REFUSAL")
COLORS = {
    "DIRECT_ANSWER": "#2E7D32",  # green
    "HEDGED_ATTEMPT": "#9CCC65",  # light green
    "LOOPHOLE_DODGE": "#EF6C00",  # orange
    "REFUSAL": "#1565C0",         # blue
}


def load_judged() -> pd.DataFrame:
    rows = []
    for path in sorted(JUDGED_DIR.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["tier"] = df["model_short"].map(MODEL_TIER)
    df["unsatisfactory"] = (
        (df["judge_label"] == "LOOPHOLE_DODGE")
        | ((df["judge_label"] == "DIRECT_ANSWER") & (df["correct_verdict"] == "WRONG"))
    )
    df["direct_correct"] = (df["judge_label"] == "DIRECT_ANSWER") & (df["correct_verdict"] == "CORRECT")
    df["direct_wrong"] = (df["judge_label"] == "DIRECT_ANSWER") & (df["correct_verdict"] == "WRONG")
    df["dodge"] = df["judge_label"] == "LOOPHOLE_DODGE"
    df["refusal"] = df["judge_label"] == "REFUSAL"
    return df


def label_breakdown(df: pd.DataFrame, output_path: Path):
    pivot = (
        df.groupby(["model_short", "knowability", "judge_label"]).size().unstack(fill_value=0)
    )
    for col in LABEL_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[list(LABEL_ORDER)]
    pivot["total"] = pivot.sum(axis=1)
    for col in LABEL_ORDER:
        pivot[f"{col}_pct"] = pivot[col] / pivot["total"]
    pivot.to_csv(output_path)
    print(f"label breakdown -> {output_path}")
    return pivot


def fine_grained_breakdown(df: pd.DataFrame, output_path: Path):
    """Split DIRECT_ANSWER into CORRECT vs WRONG."""
    df = df.copy()

    def lbl(row):
        if row["judge_label"] == "DIRECT_ANSWER":
            if row["correct_verdict"] == "CORRECT":
                return "DIRECT_CORRECT"
            elif row["correct_verdict"] == "WRONG":
                return "DIRECT_WRONG"
            else:
                return "DIRECT_UNCLEAR"
        if row["judge_label"] == "HEDGED_ATTEMPT":
            if row["correct_verdict"] == "CORRECT":
                return "HEDGED_CORRECT"
            elif row["correct_verdict"] == "WRONG":
                return "HEDGED_WRONG"
            else:
                return "HEDGED_UNCLEAR"
        return row["judge_label"]

    df["fine_label"] = df.apply(lbl, axis=1)
    pivot = df.groupby(["model_short", "knowability", "fine_label"]).size().unstack(fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot.to_csv(output_path)
    print(f"fine-grained breakdown -> {output_path}")
    return pivot


def selective_interpretation_index(df: pd.DataFrame, output_path: Path):
    """SII = P(LOOPHOLE_DODGE | unknowable) - P(LOOPHOLE_DODGE | knowable), per model."""
    rows = []
    for model, grp in df.groupby("model_short"):
        sub_un = grp[grp["knowability"] == "unknowable"]
        sub_kn = grp[grp["knowability"] == "knowable"]
        p_un = sub_un["dodge"].mean()
        p_kn = sub_kn["dodge"].mean()
        # also unsatisfactory rate
        u_un = sub_un["unsatisfactory"].mean()
        u_kn = sub_kn["unsatisfactory"].mean()
        rows.append(
            {
                "model": model,
                "tier": MODEL_TIER[model],
                "dodge_rate_knowable": p_kn,
                "dodge_rate_unknowable": p_un,
                "SII_dodge": p_un - p_kn,
                "unsat_rate_knowable": u_kn,
                "unsat_rate_unknowable": u_un,
                "SII_unsat": u_un - u_kn,
                "n_knowable": len(sub_kn),
                "n_unknowable": len(sub_un),
            }
        )
    sii = pd.DataFrame(rows).sort_values("SII_unsat", ascending=False)
    sii.to_csv(output_path, index=False)
    print(f"SII table -> {output_path}")
    return sii


def mcnemar_per_pair(df: pd.DataFrame, outcome_col: str, output_path: Path):
    """For each model, McNemar paired test on per-template paired outcomes.

    We treat a (template, seed) pair as the analysis unit. Convert each item to
    binary {outcome on knowable, outcome on unknowable} per (template, seed) and
    aggregate counts of discordant pairs.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    results = []
    for model, grp in df.groupby("model_short"):
        # pivot to (template_id, seed) rows
        wide = grp.pivot_table(
            index=["template_id", "seed"], columns="knowability", values=outcome_col, aggfunc="first"
        )
        if wide.shape[0] == 0:
            continue
        wide = wide.dropna()
        # convert to bool
        kn = wide["knowable"].astype(bool)
        un = wide["unknowable"].astype(bool)
        b = int(((~kn) & un).sum())   # knowable=0, unknowable=1
        c = int((kn & (~un)).sum())   # knowable=1, unknowable=0
        a = int((kn & un).sum())
        d = int(((~kn) & (~un)).sum())
        table = [[a, b], [c, d]]
        if (b + c) >= 1:
            res = mcnemar(table, exact=True)
            stat = res.statistic
            p = res.pvalue
        else:
            stat, p = float("nan"), 1.0
        # effect: proportion difference
        n = a + b + c + d
        diff = (b - c) / n if n else 0.0
        results.append(
            {
                "model": model,
                "tier": MODEL_TIER[model],
                "outcome": outcome_col,
                "n": n,
                "a_both_pos": a,
                "b_only_unknowable_pos": b,
                "c_only_knowable_pos": c,
                "d_both_neg": d,
                "rate_knowable": (a + c) / n if n else 0.0,
                "rate_unknowable": (a + b) / n if n else 0.0,
                "diff": diff,
                "mcnemar_stat": stat,
                "p_value": p,
            }
        )
    out = pd.DataFrame(results)
    # Bonferroni correction across models
    out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    out.to_csv(output_path, index=False)
    print(f"McNemar [{outcome_col}] -> {output_path}")
    return out


def stacked_bar_per_model(df: pd.DataFrame, output_path: Path):
    """One stacked bar per (model, knowability) showing label distribution."""
    summary = df.groupby(["model_short", "knowability", "judge_label"]).size().unstack(fill_value=0)
    for col in LABEL_ORDER:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[list(LABEL_ORDER)]
    summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100

    models = sorted(df["model_short"].unique(), key=lambda m: ["frontier", "large_open", "small_open", "reasoning"].index(MODEL_TIER[m]) if MODEL_TIER[m] in ["frontier", "large_open", "small_open", "reasoning"] else 99)
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub = summary_pct.loc[m]
        # Make sure knowable then unknowable order
        sub = sub.reindex(["knowable", "unknowable"])
        bottom = np.zeros(2)
        x = np.arange(2)
        for lbl in LABEL_ORDER:
            vals = sub[lbl].values
            ax.bar(x, vals, bottom=bottom, label=lbl, color=COLORS[lbl], edgecolor="white", linewidth=0.5)
            bottom += vals
        ax.set_title(f"{m}\n({MODEL_TIER[m]})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["knowable", "unknowable"], rotation=0)
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.set_ylabel("% of responses")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Judge label")
    plt.suptitle("KCL-Pairs: response category by model and knowability", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"stacked bar -> {output_path}")


def figure_sii(sii: pd.DataFrame, output_path: Path):
    sii = sii.copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(sii))
    ax.bar(x - width / 2, sii["dodge_rate_knowable"], width, label="knowable: dodge", color="#FFB300")
    ax.bar(x + width / 2, sii["dodge_rate_unknowable"], width, label="unknowable: dodge", color="#EF6C00")
    for i, row in enumerate(sii.itertuples()):
        ax.text(i, max(row.dodge_rate_knowable, row.dodge_rate_unknowable) + 0.02, f"ΔSII={row.SII_dodge:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.model}\n({r.tier})" for r in sii.itertuples()], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("LOOPHOLE_DODGE rate")
    ax.set_title("Selective interpretation: dodge rate by knowability")
    ax.set_ylim(0, max(0.5, sii[["dodge_rate_knowable", "dodge_rate_unknowable"]].max().max() * 1.3))
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"SII figure -> {output_path}")


def figure_unsat_sii(sii: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    x = np.arange(len(sii))
    ax.bar(x - width / 2, sii["unsat_rate_knowable"], width, label="knowable: unsatisfactory", color="#90A4AE")
    ax.bar(x + width / 2, sii["unsat_rate_unknowable"], width, label="unknowable: unsatisfactory", color="#37474F")
    for i, row in enumerate(sii.itertuples()):
        ax.text(i, max(row.unsat_rate_knowable, row.unsat_rate_unknowable) + 0.02, f"Δ={row.SII_unsat:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.model}\n({r.tier})" for r in sii.itertuples()], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Unsatisfactory-response rate (DODGE + DIRECT_WRONG)")
    ax.set_title("Total unsatisfactory rate by knowability")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"unsat figure -> {output_path}")


def figure_fine(fine: pd.DataFrame, output_path: Path):
    # Build a stacked bar of fine-grained labels
    fine = fine.copy()
    cols = [c for c in [
        "DIRECT_CORRECT", "DIRECT_WRONG", "DIRECT_UNCLEAR",
        "HEDGED_CORRECT", "HEDGED_WRONG", "HEDGED_UNCLEAR",
        "LOOPHOLE_DODGE", "REFUSAL",
    ] if c in fine.columns]
    pct = fine[cols].div(fine[cols].sum(axis=1), axis=0) * 100
    fine_cols_colors = {
        "DIRECT_CORRECT": "#1B5E20",
        "DIRECT_WRONG": "#D32F2F",
        "DIRECT_UNCLEAR": "#9E9E9E",
        "HEDGED_CORRECT": "#7CB342",
        "HEDGED_WRONG": "#F4511E",
        "HEDGED_UNCLEAR": "#BDBDBD",
        "LOOPHOLE_DODGE": "#FB8C00",
        "REFUSAL": "#1976D2",
    }
    models = sorted({i[0] for i in pct.index}, key=lambda m: ["frontier", "large_open", "small_open", "reasoning"].index(MODEL_TIER[m]) if MODEL_TIER[m] in ["frontier", "large_open", "small_open", "reasoning"] else 99)
    fig, axes = plt.subplots(1, len(models), figsize=(2.8 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        sub = pct.loc[m].reindex(["knowable", "unknowable"])
        bottom = np.zeros(2)
        x = np.arange(2)
        for col in cols:
            v = sub[col].values
            ax.bar(x, v, bottom=bottom, label=col, color=fine_cols_colors[col], edgecolor="white", linewidth=0.5)
            bottom += v
        ax.set_title(f"{m}\n({MODEL_TIER[m]})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["knowable", "unknowable"])
        ax.set_ylim(0, 100)
        if ax is axes[0]:
            ax.set_ylabel("% of responses")
    handles, labels = axes[-1].get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq.append((h, l))
            seen.add(l)
    axes[-1].legend(*zip(*uniq), loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="Fine label")
    plt.suptitle("KCL-Pairs: fine-grained response categories (DIRECT × correctness)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"fine figure -> {output_path}")


def intent_vs_behavior(df: pd.DataFrame, output_path: Path):
    """H4: are models predicting the intended interpretation correctly even when they dodge?"""
    rows = []
    for model, grp in df.groupby("model_short"):
        # restrict to unknowable for the strategic-exploitation test
        sub = grp[grp["knowability"] == "unknowable"]
        n = len(sub)
        intent_acc = sub["intent_correct"].mean()
        dodge_rate = sub["dodge"].mean()
        unsat_rate = sub["unsatisfactory"].mean()
        rows.append(
            {
                "model": model,
                "tier": MODEL_TIER[model],
                "n": n,
                "intent_acc_unknowable": intent_acc,
                "dodge_rate_unknowable": dodge_rate,
                "unsat_rate_unknowable": unsat_rate,
                "strategic_gap_dodge": intent_acc - (1 - dodge_rate),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    print(f"intent vs behavior -> {output_path}")
    return out


def summary_text(df: pd.DataFrame, sii: pd.DataFrame, mcnemar_unsat: pd.DataFrame, mcnemar_dodge: pd.DataFrame, intent: pd.DataFrame) -> str:
    lines = []
    lines.append("=== KCL-Pairs analysis summary ===")
    lines.append(f"Items: {df.shape[0]}; models: {df['model_short'].nunique()}; seeds: {df['seed'].nunique()}")
    lines.append("")
    lines.append("Selective interpretation index (SII):")
    for r in sii.itertuples():
        lines.append(
            f"  {r.model:18s} ({r.tier:10s}) "
            f"dodge: kn {r.dodge_rate_knowable:.2f} → un {r.dodge_rate_unknowable:.2f}  "
            f"(Δ={r.SII_dodge:+.3f}) | "
            f"unsat: kn {r.unsat_rate_knowable:.2f} → un {r.unsat_rate_unknowable:.2f} (Δ={r.SII_unsat:+.3f})"
        )
    lines.append("")
    lines.append("McNemar paired test [unsatisfactory] (knowable vs unknowable):")
    for r in mcnemar_unsat.itertuples():
        lines.append(
            f"  {r.model:18s} ({r.tier:10s}) "
            f"rate_kn={r.rate_knowable:.2f} rate_un={r.rate_unknowable:.2f} diff={r.diff:+.3f} "
            f"p={r.p_value:.4g} (Bonf={r.p_bonferroni:.4g})"
        )
    lines.append("")
    lines.append("McNemar paired test [dodge_only]:")
    for r in mcnemar_dodge.itertuples():
        lines.append(
            f"  {r.model:18s} ({r.tier:10s}) "
            f"rate_kn={r.rate_knowable:.2f} rate_un={r.rate_unknowable:.2f} diff={r.diff:+.3f} "
            f"p={r.p_value:.4g} (Bonf={r.p_bonferroni:.4g})"
        )
    lines.append("")
    lines.append("Intent-MC accuracy on unknowable items (H4 — does the model SAY the right interpretation even while dodging?):")
    for r in intent.itertuples():
        lines.append(f"  {r.model:18s} intent_acc={r.intent_acc_unknowable:.3f} dodge={r.dodge_rate_unknowable:.3f}")
    return "\n".join(lines)


def main():
    df = load_judged()
    if df.empty:
        print("No judged data; exiting.")
        return
    label_breakdown(df, OUT_DIR / "label_breakdown.csv")
    fine = fine_grained_breakdown(df, OUT_DIR / "fine_breakdown.csv")
    sii = selective_interpretation_index(df, OUT_DIR / "selective_interpretation_index.csv")
    mc_unsat = mcnemar_per_pair(df, "unsatisfactory", OUT_DIR / "mcnemar_unsat.csv")
    mc_dodge = mcnemar_per_pair(df, "dodge", OUT_DIR / "mcnemar_dodge.csv")
    intent = intent_vs_behavior(df, OUT_DIR / "intent_vs_behavior.csv")

    stacked_bar_per_model(df, FIG_DIR / "kcl_stacked_bar.png")
    figure_sii(sii, FIG_DIR / "kcl_sii_dodge.png")
    figure_unsat_sii(sii, FIG_DIR / "kcl_sii_unsat.png")
    figure_fine(fine, FIG_DIR / "kcl_fine_grained.png")

    txt = summary_text(df, sii, mc_unsat, mc_dodge, intent)
    (OUT_DIR / "summary.txt").write_text(txt)
    print()
    print(txt)


if __name__ == "__main__":
    main()
