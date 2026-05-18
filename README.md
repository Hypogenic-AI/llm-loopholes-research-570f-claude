# LLM and Loopholes

Do large language models take **knowledge-conflict loopholes** when they don't
know an answer but feel pressure to respond? This project tests the
Tomer-Ullman-inspired hypothesis using a new paired-prompt benchmark and five
LLMs.

## TL;DR

- We built **KCL-Pairs** — 40 items in 20 templates, each pairing a knowable
  question with a topic-matched unknowable one (e.g., Eiffel Tower height ↔
  Yangshan Lighthouse height).
- Five subject models (GPT-4o-mini, Claude-3.5-Haiku, Llama-3.1-70B/8B,
  DeepSeek-R1-distill-Llama-70B) × 3 seeds × free-form generation + intent-MC
  probe + judge.
- **Every model** shows a higher unsatisfactory-response rate on unknowable
  than on knowable items (Δ = +0.13 to +0.28; mixed-effects GLM odds ratio for
  knowability 3.24, 95% CI [1.21, 8.68], p = 0.020).
- But the mechanism is **confident hallucination, not Choi-style verbalized
  reinterpretation** — LOOPHOLE_DODGE rates are ≤5% in every model.
- Intent-MC accuracy on the same items is 98–100%: models *know* the intended
  interpretation, they just don't act on it.
- A complementary **SelfAware probe** inverts the picture: on genuinely
  unanswerable questions, all three tested models LOOPHOLE_DODGE at 87–97%
  rates with only 3–10% refusals — the direct verbal-dodge mode of the
  original hypothesis.
- A **Choi 2025 goal-conflict replication** on GPT-4o-mini matches the
  published pattern (scalar loophole 0.583, scalar_max 0.312).

The hypothesis is supported in spirit; the mechanism shifts depending on
question type. See `REPORT.md` for full results.

## File structure

```
.
├── REPORT.md                       # Primary deliverable
├── README.md                       # This file
├── planning.md                     # Pre-registered plan
├── literature_review.md            # Pre-gathered lit review
├── resources.md                    # Catalog of gathered resources
├── src/
│   ├── llm_client.py               # OpenRouter client + cache
│   ├── judge.py                    # Judge rubric (KCL)
│   ├── judge_selfaware.py          # Judge rubric (SelfAware)
│   ├── run_kcl.py                  # KCL generation runner
│   ├── run_kcl_one.py              # Per-model wrapper
│   ├── run_judging.py              # Serial judging
│   ├── run_judging_parallel.py     # Parallel (ThreadPool) judging
│   ├── run_choi.py                 # Choi 2025 scalar replication
│   ├── run_selfaware.py            # SelfAware 200-item runner
│   ├── run_selfaware_small.py      # SelfAware 60-item version
│   ├── run_judge_selfaware_parallel.py
│   ├── analyze.py                  # KCL analysis (McNemar, SII, figures)
│   ├── analyze_choi.py             # Choi analysis
│   ├── analyze_selfaware.py        # SelfAware analysis
│   ├── glm.py                      # Mixed-effects GLM
│   ├── qualitative_examples.py     # Sample CoT extracts
│   └── final_analysis.py           # Orchestrates all analyses
├── datasets/
│   ├── kcl_pairs/
│   │   ├── kcl_pairs.json          # The 40-item paired benchmark
│   │   ├── truth_notes.json        # Curated correctness notes
│   │   └── build.py                # Dataset construction script
│   ├── selfaware/                  # SelfAware dataset (HF)
│   ├── truthfulqa/                 # TruthfulQA (not used in main run)
│   ├── sycophancy/                 # Anthropic sycophancy (not used)
│   └── loophole_scenarios/         # Choi 108 power scenarios (not used)
├── code/
│   └── ambiguous-loophole-exploitation/   # Choi 2025 reference impl.
├── results/
│   ├── raw/                        # JSONL of all model outputs
│   │   ├── kcl/                    # 120 lines per model × 5 models
│   │   ├── choi/                   # gpt-4o-mini full + claude-haiku partial
│   │   └── selfaware/              # 60 lines per model × 3 models
│   ├── judged/                     # KCL responses + judge labels
│   ├── judged_selfaware/           # SelfAware responses + judge labels
│   ├── cache/                      # Per-call response cache (hash-keyed)
│   └── analysis/                   # Computed metrics, tables, summary.txt
├── figures/                        # PNGs used in REPORT.md
├── logs/                           # stdout/stderr from runs
└── papers/                         # 20 pre-downloaded reference papers
```

## Key result figures

- `figures/kcl_stacked_bar.png` — judge-label distribution per model × knowability
- `figures/kcl_fine_grained.png` — DIRECT_CORRECT vs DIRECT_WRONG vs DODGE vs REFUSAL
- `figures/kcl_sii_dodge.png` — selective interpretation index (dodge rate)
- `figures/kcl_sii_unsat.png` — unsatisfactory rate by knowability
- `figures/selfaware_stacked_bar.png` — SelfAware response categories
- `figures/choi_loophole_rates.png` — Choi 2025 replication

## How to reproduce

```bash
# 1. Setup
uv venv && source .venv/bin/activate
uv pip install openai tenacity datasets pandas numpy scipy matplotlib \
               seaborn statsmodels scikit-learn pyyaml tqdm

# 2. API key
export OPENROUTER_KEY=<your_key>

# 3. Build the dataset
python datasets/kcl_pairs/build.py

# 4. Run KCL generation (40 items × 3 seeds × 5 models = 600 + 600 calls)
python src/run_kcl.py --models gpt-4o-mini claude-3.5-haiku llama-3.1-70b \
                              llama-3.1-8b deepseek-r1-70b --seeds 0 1 2

# 5. Judge (parallel, ~2 min/model)
python src/run_judging_parallel.py --models gpt-4o-mini claude-3.5-haiku \
                                            llama-3.1-70b llama-3.1-8b \
                                            deepseek-r1-70b --workers 12

# 6. Companion experiments
python src/run_choi.py --models gpt-4o-mini --seeds 0 1 2 --trials 2
python src/run_selfaware_small.py gpt-4o-mini claude-3.5-haiku llama-3.1-8b
python src/run_judge_selfaware_parallel.py --models gpt-4o-mini \
                                                   claude-3.5-haiku \
                                                   llama-3.1-8b

# 7. Final analyses + figures
python src/final_analysis.py
```

All API calls are cached in `results/cache/`. Re-running with the same
parameters takes seconds.

## Key files to read first

1. **`REPORT.md`** — primary deliverable, all results.
2. **`planning.md`** — pre-registered hypotheses and methodology.
3. **`literature_review.md`** — context and prior work.
4. **`datasets/kcl_pairs/kcl_pairs.json`** — the benchmark itself.
5. **`results/analysis/qualitative_samples.md`** — illustrative responses.

## Cost / resource usage

- ~2,800 OpenRouter API calls (all cached after first run).
- Estimated cost ≤ $15.
- No local GPU required (the experiments are API-bound, not compute-bound).
- All experiments run on a single machine (parallelism is via Python threads
  for I/O-bound API calls).
