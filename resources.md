# Resources Catalog

## Summary

20 papers, 5 datasets, 1 primary code repository. All resources are organized under `papers/`, `datasets/`, and `code/`. Reproduction is via `pyproject.toml` (uv venv) plus `datasets/download.py` and a single git clone (`code/ambiguous-loophole-exploitation`).

## Papers

20 PDFs in `papers/`. Full table is in `papers/README.md`. Three are deep-read and summarized in `notes_primary_papers.md`:

| Key | Title | Year | arXiv | Cites |
|-----|-------|------|-------|-------|
| `loopholes` | Language Models Identify Ambiguities and Exploit Loopholes | 2025 | 2508.19546 | 2 |
| `loopholes_humans` | Comparing the Evaluation and Production of Loophole Behavior in Humans and LLMs | 2023 | EMNLP Findings | 1 |
| `subterfuge` | Sycophancy to Subterfuge: Reward Tampering | 2024 | 2406.10162 | 116 |
| `jailbroken` | Jailbroken: How Does LLM Safety Training Fail? | 2023 | 2307.02483 | 1786 |
| `past_tense` | Does Refusal Training Generalize to the Past Tense? | 2024 | 2407.11969 | 74 |
| `pragmatic` | Pragmatic Instruction Following via CLIPS | 2024 | 2402.17930 | 40 |
| `trace` | Is It Thinking or Cheating? TRACE for Implicit Reward Hacking | 2025 | 2510.01367 | 9 |
| `proxy_gaming` | Detecting Proxy Gaming via Evaluator Stress Tests | 2025 | 2507.05619 | 10 |
| `sirens_song` | Siren's Song: Hallucination Survey | 2023 | 2309.01219 | 990 |
| `honesty_survey` | A Survey on the Honesty of LLMs | 2024 | 2409.18786 | 23 |
| `dont_know` | Can AI Assistants Know What They Don't Know? | 2024 | 2401.13275 | 48 |
| `kn_boundary` | Teaching LLMs to Express Knowledge Boundary | 2024 | 2406.10881 | 31 |
| `trust_me_wrong` | Trust Me, I'm Wrong (CHOKE) | 2025 | 2502.12964 | 11 |
| `learn_refuse` | Learn to Refuse | 2023 | 2311.01041 | 41 |
| `curriculum` | Automatic Curriculum Expert Iteration (laziness vs. hallucination) | 2024 | 2410.07627 | 13 |
| `self_factuality` | Self-Alignment for Factuality | 2024 | 2402.09267 | 107 |
| `internal_lying` | Internal State Knows When It's Lying | 2023 | 2304.13734 | 638 |
| `llms_know_more` | LLMs Know More Than They Show | 2024 | 2410.02707 | 195 |
| `hidden_hallucination` | Do LLMs Know about Hallucination? Hidden States | 2024 | 2402.09733 | 59 |
| `sem_entropy` | Semantic Entropy Probes | 2024 | 2406.15927 | 185 |

Detailed annotations: `papers/README.md`. Resolution metadata + raw search results: `paper_search_results/_resolved.json`, `paper_search_results/_merged.json`.

## Datasets

5 datasets, total ~26 MB downloaded (data files git-ignored). Full table is in `datasets/README.md`.

| Name | Source | Size | Task | Local path |
|------|--------|------|------|------------|
| Choi 2025 power scenarios | github.com/esteng/ambiguous-loophole-exploitation | 108 prompts | 3-way MC: compliance/loophole/non-compliance | `datasets/loophole_scenarios/choi2025_power_scenarios.json` |
| Choi 2025 scalar+bracket templates | same repo | programmatic; ~hundreds per run | open-ended numerical answer + intent MC | generated in-place |
| TruthfulQA (generation) | HuggingFace `truthful_qa` | 817 q | open-ended truthfulness | `datasets/truthfulqa/generation_validation/` |
| SelfAware | HuggingFace `JesusCrist/SelfAware` | 3,369 q | answerable vs. unanswerable | `datasets/selfaware/train/` |
| Anthropic sycophancy evals (3 files) | github.com/anthropics/evals/sycophancy | ~25 MB | persona-conditioned 2-way MC | `datasets/sycophancy/*.jsonl` |

Reproduce all of the above with `source .venv/bin/activate && python datasets/download.py`.

## Code Repositories

| Name | URL | Purpose | Local path |
|------|-----|---------|------------|
| `ambiguous-loophole-exploitation` | github.com/esteng/ambiguous-loophole-exploitation | Choi 2025 reference impl. — templates for all loophole conditions + 108-prompt power dataset + pre-computed results for 14 models | `code/ambiguous-loophole-exploitation/` |

Murthy 2023's `github.com/skmur/LLLMs` is no longer reachable; the relevant stimuli persist via Bridgers 2025 → Choi 2025.

## Search strategy

- Used the `paper-finder` skill (`.claude/skills/paper-finder/scripts/find_papers.py --mode fast`) with 7 queries:
  - "LLM loopholes"
  - "LLM specification gaming reward hacking"
  - "LLM hallucination know what they know"
  - "LLM sycophancy reward gaming"
  - "LLM ambiguity pragmatic reasoning instruction"
  - "LLM jailbreak refusal evasion alignment"
  - Initial "large language models loopholes specification gaming" (fell back due to missing httpx; the others succeeded)
- Merged 7 result files (200 unique relevance≥2 papers) → ranked by relevance + citations → picked 20.
- Resolved arXiv IDs via the Semantic Scholar **batch** API (`graph/v1/paper/batch`) in a single request after individual queries hit rate limits.
- Downloaded all 20 PDFs from `arxiv.org/pdf/<id>` directly (one fell back to its ACL Anthology mirror).

## Selection criteria

- Direct loophole studies (3 papers) — must-have.
- Specification gaming / reward hacking (2 papers) — bridge to RL alignment literature.
- Knowledge-boundary / hallucination / honesty (8 papers) — directly relevant to the "lack sufficient knowledge but still want to answer" half of the hypothesis.
- Internal-state probes (4 papers) — mechanism for detecting deliberate overclaiming.
- Safety-training loopholes (2 papers) — concrete deployed-model examples (jailbroken, past-tense).
- Pragmatic ambiguity (1 paper, CLIPS) — comparator architecture.

## Challenges encountered

1. **Semantic Scholar individual queries rate-limited at >5 per minute.** Solved by switching to the `paper/batch` endpoint and lookup all 20 papers in one POST.
2. **arXiv search-by-title API rate-limited even at 3.5 s/request.** Same workaround.
3. **Murthy 2023's GitHub repo (`skmur/LLLMs`) is gone.** The stimuli persist downstream via Bridgers 2025 → Choi 2025, so no real loss.
4. **Three paper-finder queries timed out at 90 s.** Re-run individually with a 180 s timeout; all returned successfully.
5. **`uv add` initially tried to build a wheel because of an oversized `pyproject.toml`.** Simplified the pyproject to a bare project section and used `uv pip install` for direct adds. Final dependencies: `pypdf`, `requests`, `arxiv`, `httpx`, `datasets`.

## Gaps and workarounds

- **No published implementation of Denison 2024's curriculum.** Treat it as conceptual inspiration only; we cannot retrain at curriculum scale.
- **CHOKE benchmark (Simhi 2025) data was not publicly released as of search.** Adopt the methodology (paired-paraphrase prompts) and construct knowledge-conflict variants ourselves.
- **No open dataset specifically for *knowledge-conflict* loopholes.** This is the experimental gap our project should fill.

## Recommendations for experiment design

1. **Primary dataset.** Choi 2025 (108 power scenarios) for the goal-conflict slice. SelfAware + TruthfulQA for the knowledge-conflict slice.
2. **Baselines.**
   - Choi 2025 pre-computed runs: GPT-4o, Claude-3.7-Sonnet, Gemini-2.0-Flash, Llama-3.1-{8B,70B}, Qwen-2.5-{3,7,14,32,72}B, DeepSeek-R1-Distill-Qwen{1.5,7,14,32}B.
   - Anthropic helpful-only Claude-2 (Denison 2024 reports tampering = 0% in 100k trials).
2. **Metrics.** Loophole exploitation rate, intent-prediction accuracy, selective-interpretation index, refusal rate, hidden-state probe AUC (open weights only), semantic entropy.
3. **Code base.** Fork `code/ambiguous-loophole-exploitation` and add a new job `knowledge_loophole` that reuses the same `Prompt` / `args` / output-JSON conventions, so Choi-style plotting code applies directly.
