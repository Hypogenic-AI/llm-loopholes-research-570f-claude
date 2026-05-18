"""Faster judging via ThreadPoolExecutor."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def judge_one(rec):
    tid = rec["template_id"]
    kn = rec["knowability"]
    truth_note = TRUTH.get(tid, {}).get(kn, "")
    cat = judge_response(
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
    intent_correct = (rec.get("intent_letter") == "A")
    out = dict(rec)
    out["judge_label"] = cat.label
    out["judge_rationale"] = cat.rationale
    out["correct_verdict"] = corr.verdict
    out["correct_rationale"] = corr.rationale
    out["intent_correct"] = intent_correct
    return out


def judge_file(input_path: Path, out_path: Path, max_workers=12):
    if not input_path.exists():
        return
    records = [json.loads(l) for l in input_path.read_text().splitlines() if l.strip()]
    print(f"  judging {len(records)} records w/ {max_workers} threads")
    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2idx = {ex.submit(judge_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {**records[i], "judge_label": "LOOPHOLE_DODGE", "judge_rationale": f"ERR {e}", "correct_verdict": "UNCLEAR", "correct_rationale": str(e), "intent_correct": False}
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(records)}")
    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    for m in args.models:
        inp = RAW_DIR / f"{m}.jsonl"
        out = OUT_DIR / f"{m}.jsonl"
        print(f"[JUDGE-P] {m}")
        judge_file(inp, out, max_workers=args.workers)
