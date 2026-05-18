"""Re-download all datasets used by this workspace.

Run from the repo root with the workspace venv active:
    source .venv/bin/activate && python datasets/download.py
"""
from __future__ import annotations

import os
import shutil
import urllib.request

from datasets import load_dataset

WS = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _ws(p: str) -> str:
    return os.path.join(WS, p)


def main() -> None:
    os.makedirs(_ws("datasets/truthfulqa"), exist_ok=True)
    os.makedirs(_ws("datasets/selfaware"), exist_ok=True)
    os.makedirs(_ws("datasets/sycophancy"), exist_ok=True)
    os.makedirs(_ws("datasets/loophole_scenarios"), exist_ok=True)

    truth_out = _ws("datasets/truthfulqa/generation_validation")
    if not os.path.exists(truth_out):
        load_dataset("truthfulqa/truthful_qa", "generation", split="validation").save_to_disk(truth_out)
        print("ok truthful_qa")

    sa_out = _ws("datasets/selfaware/train")
    if not os.path.exists(sa_out):
        load_dataset("JesusCrist/SelfAware", split="train").save_to_disk(sa_out)
        print("ok SelfAware")

    base = "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/"
    for name in [
        "sycophancy_on_political_typology_quiz.jsonl",
        "sycophancy_on_nlp_survey.jsonl",
        "sycophancy_on_philpapers2020.jsonl",
    ]:
        out = _ws(f"datasets/sycophancy/{name}")
        if os.path.exists(out) and os.path.getsize(out) > 1_000:
            continue
        with urllib.request.urlopen(base + name, timeout=60) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
        print(f"ok {name}")

    src = _ws("code/ambiguous-loophole-exploitation/scenario_prompts.json")
    dst = _ws("datasets/loophole_scenarios/choi2025_power_scenarios.json")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print("ok choi2025_power_scenarios")


if __name__ == "__main__":
    main()
