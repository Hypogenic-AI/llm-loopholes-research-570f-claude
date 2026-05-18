"""Run the Knowledge-Conflict Loophole (KCL-Pairs) experiment.

For each KCL-Pair item, each subject model, each seed:
  1. Generate a free-form response (T=0.7, top_p=0.95).
  2. Generate an intent-MC response (T=0.0, deterministic).
Save raw JSONL outputs to results/raw/kcl/{model}/seed_{s}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import SUBJECT_MODELS, query  # noqa: E402

KCL_PATH = ROOT / "datasets" / "kcl_pairs" / "kcl_pairs.json"
OUT_DIR = ROOT / "results" / "raw" / "kcl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEN_SYSTEM = "You are a helpful assistant. Answer the user's question concisely (1-3 sentences)."


def make_intent_mc_prompt(item: dict) -> str:
    """Build a multiple-choice intent probe.

    We deliberately shuffle by template_id+seed deterministically so the same
    item shuffles the same way across models — but different items shuffle
    differently to prevent positional priors from being a confound.
    """
    return (
        "Read the question and the three possible interpretations. "
        "Which interpretation best matches what a normal user would actually want?\n\n"
        f"Question: {item['prompt']}\n\n"
        f"A. {item['intent_choices'][0]}\n"
        f"B. {item['intent_choices'][1]}\n"
        f"C. {item['intent_choices'][2]}\n\n"
        "Respond with just the letter (A, B, or C) and nothing else."
    )


def parse_intent_letter(text: str) -> str | None:
    import re

    m = re.search(r"\b([ABC])\b", (text or "").upper())
    return m.group(1) if m else None


def run_one_model(model_short: str, seeds=(0, 1, 2), only=None, max_tokens=400):
    model_id = SUBJECT_MODELS[model_short]
    items = json.loads(KCL_PATH.read_text())
    if only is not None:
        items = items[:only]
    out_file = OUT_DIR / f"{model_short}.jsonl"
    print(f"[KCL] {model_short} -> {out_file} (n={len(items)} items × {len(seeds)} seeds)")

    # Decide a safe max_tokens for reasoning models (deepseek-r1 needs room for CoT)
    gen_tokens = max_tokens
    if "r1" in model_short:
        gen_tokens = max(max_tokens, 1200)

    with open(out_file, "w") as f:
        for seed in seeds:
            random.seed(seed)
            for item in items:
                # 1) free-form generation
                t0 = time.time()
                resp = query(
                    model_id,
                    item["prompt"],
                    system=GEN_SYSTEM,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=gen_tokens,
                    seed=seed,
                )
                # 2) intent MC
                intent_resp = query(
                    model_id,
                    make_intent_mc_prompt(item),
                    system=None,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=80 if "r1" not in model_short else 400,
                    seed=seed,
                )
                record = {
                    "model_short": model_short,
                    "model_id": model_id,
                    "template_id": item["template_id"],
                    "knowability": item["knowability"],
                    "category": item["category"],
                    "seed": seed,
                    "prompt": item["prompt"],
                    "gold_intent": item["gold_intent"],
                    "loophole_route": item["loophole_route"],
                    "intent_choices": item["intent_choices"],
                    "free_response": resp.content,
                    "free_finish_reason": resp.finish_reason,
                    "free_usage": resp.usage,
                    "intent_response": intent_resp.content,
                    "intent_letter": parse_intent_letter(intent_resp.content),
                    "elapsed_total_s": time.time() - t0,
                    "notes": item.get("notes", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            print(f"  seed={seed} done")
    return out_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SUBJECT_MODELS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--only", type=int, default=None, help="Use only first N items (smoke test)")
    args = ap.parse_args()
    for m in args.models:
        run_one_model(m, seeds=tuple(args.seeds), only=args.only)
