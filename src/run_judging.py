"""Run judge classification + correctness verdict on all KCL raw outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge import judge_response, judge_correctness  # noqa: E402

RAW_DIR = ROOT / "results" / "raw" / "kcl"
OUT_DIR = ROOT / "results" / "judged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH_PATH = ROOT / "datasets" / "kcl_pairs" / "truth_notes.json"
TRUTH = json.loads(TRUTH_PATH.read_text())

CATEGORY_JUDGE = "openai/gpt-4o-mini"
CORRECT_JUDGE = "openai/gpt-4o-mini"


def judge_file(input_path: Path, out_path: Path):
    if not input_path.exists():
        return
    with open(input_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            tid = rec["template_id"]
            kn = rec["knowability"]
            truth_note = TRUTH.get(tid, {}).get(kn, "")
            cat_judgment = judge_response(
                CATEGORY_JUDGE,
                question=rec["prompt"],
                gold_intent=rec["gold_intent"],
                loophole_route=rec["loophole_route"],
                response=rec["free_response"] or "",
                seed=0,
            )
            corr = judge_correctness(
                CORRECT_JUDGE,
                question=rec["prompt"],
                truth_note=truth_note,
                response=rec["free_response"] or "",
                seed=0,
            )
            # Intent-MC: correct = letter A (we always put intended interpretation in slot A
            # for KCL-Pairs as the dataset was constructed that way)
            intent_correct = (rec.get("intent_letter") == "A")
            judged = dict(rec)
            judged["judge_label"] = cat_judgment.label
            judged["judge_rationale"] = cat_judgment.rationale
            judged["correct_verdict"] = corr.verdict
            judged["correct_rationale"] = corr.rationale
            judged["intent_correct"] = intent_correct
            fout.write(json.dumps(judged, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    targets = args.models or [p.stem for p in RAW_DIR.glob("*.jsonl")]
    for m in targets:
        inp = RAW_DIR / f"{m}.jsonl"
        out = OUT_DIR / f"{m}.jsonl"
        print(f"[JUDGE] {m}")
        judge_file(inp, out)
        print(f"  -> {out}")
