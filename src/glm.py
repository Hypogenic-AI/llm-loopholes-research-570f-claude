"""Mixed-effects logistic regression: loophole ~ knowability + tier + (1|template).

statsmodels has a binomial GLM with cluster-robust standard errors, which is a
serviceable substitute for a full mixed-effects model on a project this size.
We model:
    unsatisfactory ~ knowability + tier
    cluster=template_id
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import MODEL_TIER  # noqa: E402

import statsmodels.api as sm
import statsmodels.formula.api as smf

JUDGED_DIR = ROOT / "results" / "judged"
OUT_DIR = ROOT / "results" / "analysis"


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(JUDGED_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["tier"] = df["model_short"].map(MODEL_TIER)
    df["unsatisfactory"] = (
        (df["judge_label"] == "LOOPHOLE_DODGE")
        | ((df["judge_label"] == "DIRECT_ANSWER") & (df["correct_verdict"] == "WRONG"))
    )
    df["dodge"] = df["judge_label"] == "LOOPHOLE_DODGE"
    df["unknowable"] = df["knowability"] == "unknowable"
    return df


def fit_glm(df: pd.DataFrame, outcome: str):
    # Predict outcome from knowability + tier with template-clustered SEs
    d = df.copy()
    d["y"] = d[outcome].astype(int)
    # Treat 'small_open' as reference
    d["tier_cat"] = pd.Categorical(d["tier"], categories=["small_open", "large_open", "frontier", "reasoning"])
    md = smf.glm(
        "y ~ C(unknowable) + C(tier_cat)",
        data=d,
        family=sm.families.Binomial(),
    )
    # cluster-robust
    res = md.fit(cov_type="cluster", cov_kwds={"groups": d["template_id"]})
    return res


def main():
    df = load()
    if df.empty:
        print("no data")
        return
    print("== unsatisfactory ~ knowability + tier ==")
    res_u = fit_glm(df, "unsatisfactory")
    print(res_u.summary())

    print("\n== dodge ~ knowability + tier ==")
    res_d = fit_glm(df, "dodge")
    print(res_d.summary())

    # Save odds ratios
    rows = []
    for name, res in [("unsatisfactory", res_u), ("dodge", res_d)]:
        for term in res.params.index:
            rows.append(
                {
                    "outcome": name,
                    "term": term,
                    "coef": res.params[term],
                    "OR": float(np.exp(res.params[term])),
                    "se": res.bse[term],
                    "z": res.tvalues[term],
                    "p": res.pvalues[term],
                    "ci_lo": float(np.exp(res.conf_int().loc[term, 0])),
                    "ci_hi": float(np.exp(res.conf_int().loc[term, 1])),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "glm_odds_ratios.csv", index=False)
    print(f"OR table -> {OUT_DIR / 'glm_odds_ratios.csv'}")


if __name__ == "__main__":
    main()
