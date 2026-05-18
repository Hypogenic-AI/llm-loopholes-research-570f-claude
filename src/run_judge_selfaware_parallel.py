"""Parallel SelfAware judging using ThreadPoolExecutor."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from judge import judge_response  # noqa: E402
from judge_selfaware import gold_intent, loophole_route  # noqa: E402

RAW_DIR = ROOT / "results" / "raw" / "selfaware"
OUT_DIR = ROOT / "results" / "judged_selfaware"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CATEGORY_JUDGE = "openai/gpt-4o-mini"


def judge_one(rec):
    j = judge_response(
        CATEGORY_JUDGE,
        question=rec["question"],
        gold_intent=gold_intent(rec),
        loophole_route=loophole_route(rec),
        response=rec["response"] or "",
        seed=0,
    )
    out = dict(rec)
    out["judge_label"] = j.label
    out["judge_rationale"] = j.rationale
    return out


def judge_file(inp: Path, out: Path, workers=12):
    records = [json.loads(l) for l in inp.read_text().splitlines() if l.strip()]
    print(f"  {len(records)} records w/ {workers} threads")
    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2idx = {ex.submit(judge_one, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {**records[i], "judge_label": "LOOPHOLE_DODGE", "judge_rationale": f"ERR {e}"}
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(records)}")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    for m in args.models:
        inp = RAW_DIR / f"{m}.jsonl"
        out = OUT_DIR / f"{m}.jsonl"
        if not inp.exists() or inp.stat().st_size == 0:
            print(f"[SKIP] {m}: no input")
            continue
        print(f"[JUDGE-SA-P] {m}")
        judge_file(inp, out, workers=args.workers)
