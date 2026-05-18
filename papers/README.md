# Downloaded Papers

20 papers (all PDFs in this directory). Listed by group. `key` matches the PDF filename `<key>.pdf`.

## Primary loophole papers (deep-read)

| Key | Title | Authors | Year | arXiv | Cites |
|-----|-------|---------|------|-------|-------|
| `loopholes` | Language Models Identify Ambiguities and Exploit Loopholes | Choi, Bansal, Stengel-Eskin | 2025 | 2508.19546 | 2 |
| `loopholes_humans` | Comparing the Evaluation and Production of Loophole Behavior in Humans and Large Language Models | Murthy, Parece, Bridgers, Qian, Ullman | 2023 (EMNLP Findings) | — | 1 |
| `subterfuge` | Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models | Denison, MacDiarmid, … Hubinger (Anthropic) | 2024 | 2406.10162 | 116 |

These three papers define the conceptual space:
- **`loopholes`** — the only paper directly testing whether modern LLMs *exploit* loopholes (scalar implicature, bracketing ambiguity, power scenarios). Includes a public dataset+code (`code/ambiguous-loophole-exploitation`).
- **`loopholes_humans`** — establishes the human baseline + the 36-scenario stimulus set + the trouble/upset/humor evaluation rubric. (Repo `github.com/skmur/LLLMs` is no longer accessible; the stimuli live in `code/ambiguous-loophole-exploitation/scenario_prompts.json`.)
- **`subterfuge`** — connects loopholes to the RL specification-gaming literature. Shows that sycophancy → rubric tampering → reward tampering generalize zero-shot.

## Goal-conflict / safety-training loopholes

| Key | Title | Authors | arXiv | Cites |
|-----|-------|---------|-------|-------|
| `jailbroken` | Jailbroken: How Does LLM Safety Training Fail? | Wei, Haghtalab, Steinhardt | 2307.02483 | 1786 |
| `past_tense` | Does Refusal Training in LLMs Generalize to the Past Tense? | Andriushchenko, Flammarion | 2407.11969 | 74 |
| `pragmatic` | Pragmatic Instruction Following and Goal Assistance via Cooperative Language-Guided Inverse Planning | Zhi-Xuan, Ying, Mansinghka, Tenenbaum | 2402.17930 | 40 |

## Reward / proxy gaming

| Key | Title | Authors | arXiv | Cites |
|-----|-------|---------|-------|-------|
| `trace` | Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort | X. Wang et al. | 2510.01367 | 9 |
| `proxy_gaming` | Detecting Proxy Gaming in RL and LLM Alignment via Evaluator Stress Tests | Shihab, Akter, Sharma | 2507.05619 | 10 |

## Knowledge-boundary / hallucination / "knowing what they don't know" (most directly relevant to our hypothesis's *knowledge-bounded* loopholes)

| Key | Title | Authors | arXiv | Cites |
|-----|-------|---------|-------|-------|
| `sirens_song` | Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models | Y. Zhang et al. | 2309.01219 | 990 |
| `honesty_survey` | A Survey on the Honesty of Large Language Models | S. Li et al. | 2409.18786 | 23 |
| `dont_know` | Can AI Assistants Know What They Don't Know? | Q. Cheng et al. | 2401.13275 | 48 |
| `kn_boundary` | Teaching Large Language Models to Express Knowledge Boundary from Their Own Signals | L. Chen et al. | 2406.10881 | 31 |
| `trust_me_wrong` | Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer (CHOKE) | Simhi et al. | 2502.12964 | 11 |
| `learn_refuse` | Learn to Refuse: Knowledge Scope Limitation and Refusal Mechanism | L. Cao | 2311.01041 | 41 |
| `curriculum` | Automatic Curriculum Expert Iteration for Reliable LLM Reasoning | Z. Zhao et al. | 2410.07627 | 13 |
| `self_factuality` | Self-Alignment for Factuality: Mitigating Hallucinations via Self-Evaluation | X. Zhang et al. | 2402.09267 | 107 |

## Internal-state / probe-based detection

| Key | Title | Authors | arXiv | Cites |
|-----|-------|---------|-------|-------|
| `internal_lying` | The Internal State of an LLM Knows When it's Lying | Azaria, Mitchell | 2304.13734 | 638 |
| `llms_know_more` | LLMs Know More Than They Show: Intrinsic Representations of Hallucinations | Orgad et al. | 2410.02707 | 195 |
| `hidden_hallucination` | Do LLMs Know about Hallucination? Investigation of Hidden States | Duan, Yang, Tam | 2402.09733 | 59 |
| `sem_entropy` | Semantic Entropy Probes: Robust and Cheap Hallucination Detection | Kossen et al. | 2406.15927 | 185 |

## PDF chunks for the three primary papers
The three deep-read papers were split into 3-page chunks under `papers/pages/` (manifest in `*_manifest.txt`). Other papers are kept whole; chunk them on demand with `python .claude/skills/paper-finder/scripts/pdf_chunker.py papers/<key>.pdf`.

## Resolving arXiv IDs / metadata
- `paper_search_results/_resolved.json` — all 20 papers with arXiv ID, DOI, venue, authors, abstract.
- `paper_search_results/_merged.json` — full ranked list of ~200 relevance≥2 papers from paper-finder.
- The download/resolve scripts (`download_papers.py`, `resolve_arxiv.py`) are kept at the workspace root for reproducibility.
