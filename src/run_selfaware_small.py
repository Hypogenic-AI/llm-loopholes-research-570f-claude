"""Smaller SelfAware probe: 60 items (30 answerable + 30 unanswerable), 1 seed, 3 models.

Used as supplementary external validity after the main KCL run.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import SUBJECT_MODELS, query  # noqa: E402
from datasets import load_from_disk  # noqa: E402

OUT_DIR = ROOT / "results" / "raw" / "selfaware"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_PATH = ROOT / "datasets" / "selfaware" / "sampled_60.json"


def build_sample(n_each: int = 30, seed: int = 42):
    if SAMPLE_PATH.exists():
        return json.loads(SAMPLE_PATH.read_text())
    ds = load_from_disk(str(ROOT / "datasets" / "selfaware" / "train"))
    items = [dict(r) for r in ds]
    rng = random.Random(seed)
    answerable = [x for x in items if x["answerable"]]
    unanswerable = [x for x in items if not x["answerable"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    sample = answerable[:n_each] + unanswerable[:n_each]
    SAMPLE_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    return sample


GEN_SYSTEM = "You are a helpful assistant. Answer the user's question concisely (1-3 sentences)."


def run_one_model(model_short: str, seeds=(0,)):
    sample = build_sample()
    model_id = SUBJECT_MODELS[model_short]
    out_file = OUT_DIR / f"{model_short}.jsonl"
    print(f"[SA-small] {model_short} -> {out_file} (n={len(sample)} × {len(seeds)} seeds)")
    gen_tokens = 600 if "r1" in model_short else 300
    with open(out_file, "w") as f:
        for seed in seeds:
            for it in sample:
                resp = query(
                    model_id,
                    it["question"],
                    system=GEN_SYSTEM,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=gen_tokens,
                    seed=seed,
                )
                rec = {
                    "model_short": model_short,
                    "model_id": model_id,
                    "question_id": it["question_id"],
                    "answerable": it["answerable"],
                    "source": it["source"],
                    "seed": seed,
                    "question": it["question"],
                    "gold_answers": it.get("answer", []),
                    "response": resp.content,
                    "finish_reason": resp.finish_reason,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            print(f"  seed={seed} done")
    return out_file


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt-4o-mini", "claude-3.5-haiku", "llama-3.1-8b"]
    build_sample()
    for m in models:
        run_one_model(m)
