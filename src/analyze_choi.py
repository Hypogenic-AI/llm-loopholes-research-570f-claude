"""Analyze the Choi-style goal-conflict (scalar) replication."""
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

RAW_DIR = ROOT / "results" / "raw" / "choi"
OUT_DIR = ROOT / "results" / "analysis"
FIG_DIR = ROOT / "figures"


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(RAW_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
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
        print("No Choi data; exiting")
        return
    # By (model, job, amount)
    summary = (
        df.groupby(["model_short", "job", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "choi_label_counts.csv", index=False)

    # Loophole rates: per (model, job)
    rates = (
        df.assign(is_loop=(df["label"] == "loophole"))
        .groupby(["model_short", "job"])
        .agg(loophole_rate=("is_loop", "mean"), n=("is_loop", "size"))
        .reset_index()
    )
    rates["tier"] = rates["model_short"].map(MODEL_TIER)
    rates.to_csv(OUT_DIR / "choi_loophole_rates.csv", index=False)
    print(rates.round(3).to_string(index=False))

    # Plot
    pivot = rates.pivot(index="model_short", columns="job", values="loophole_rate")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, color={"scalar": "#EF6C00", "scalar_max": "#9CCC65"})
    ax.set_ylabel("Loophole rate (give 1 item)")
    ax.set_title("Goal-conflict (scalar) loophole rate by model and prompt variant")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "choi_loophole_rates.png", dpi=150)
    plt.close()
    print(f"choi figure -> {FIG_DIR / 'choi_loophole_rates.png'}")


if __name__ == "__main__":
    main()
