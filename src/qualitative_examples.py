"""Print qualitative examples of each judge category for the writeup."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JUDGED_DIR = ROOT / "results" / "judged"
OUT_PATH = ROOT / "results" / "analysis" / "qualitative_samples.md"


def main():
    rows = []
    for p in sorted(JUDGED_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("no data")
        return
    rng = random.Random(7)
    # group by category
    buckets = {}
    for r in rows:
        key = (
            r["judge_label"],
            r["correct_verdict"],
            r["knowability"],
        )
        buckets.setdefault(key, []).append(r)
    lines = ["# Qualitative samples from KCL-Pairs\n"]
    interesting_keys = [
        ("LOOPHOLE_DODGE", "UNCLEAR", "unknowable"),
        ("LOOPHOLE_DODGE", "WRONG", "unknowable"),
        ("DIRECT_ANSWER", "WRONG", "unknowable"),
        ("DIRECT_ANSWER", "CORRECT", "knowable"),
        ("DIRECT_ANSWER", "CORRECT", "unknowable"),
        ("REFUSAL", "UNCLEAR", "unknowable"),
        ("HEDGED_ATTEMPT", "WRONG", "unknowable"),
        ("HEDGED_ATTEMPT", "CORRECT", "unknowable"),
    ]
    for key in interesting_keys:
        bucket = buckets.get(key, [])
        if not bucket:
            continue
        rng.shuffle(bucket)
        lines.append(f"\n## {key[0]} / correct={key[1]} / {key[2]}  ({len(bucket)} total)")
        for r in bucket[:3]:
            lines.append(f"\n**Model**: `{r['model_short']}`  **Template**: {r['template_id']}  **Seed**: {r['seed']}")
            lines.append(f"\n**Prompt**: {r['prompt']}")
            resp = (r['free_response'] or '').replace('\n', '\n  ')
            lines.append(f"\n**Response**:\n  {resp[:1200]}")
            lines.append(f"\n**Judge**: `{r['judge_label']}` / `{r['correct_verdict']}` — _{r['judge_rationale']}_")
            lines.append("---")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
