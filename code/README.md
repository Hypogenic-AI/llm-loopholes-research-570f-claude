# Cloned Repositories

## 1. `ambiguous-loophole-exploitation/` — Choi, Bansal, Stengel-Eskin (2025)

- **URL:** https://github.com/esteng/ambiguous-loophole-exploitation
- **License:** MIT
- **Purpose:** Reference implementation for the only paper that *directly* tests loophole exploitation by frontier LLMs. Defines and runs three loophole settings (scalar implicature, bracketing ambiguity, power scenarios).
- **Key entry points:**
  - `src/main.py` — orchestrator. CLI: `--job <name> --model <hf-or-api-id> --trials N --output_dir results/<run>`.
  - `src/ambiguity.py` — programmatic prompt generators for every condition (scalar, scalar_priced, scalar_max, scalar_intent, bracket_{tax,game,store}{,_unambig,_intent}, power_scenario).
  - `src/models/` — model wrappers.
  - `scenario_prompts.json` — 108 hand-written power-relationship scenarios (36 base stories × 3 power dynamics, from Bridgers et al. 2025).
- **Jobs supported (via `--job`):**
  - `scalar`, `scalar_priced`, `scalar_max`, `scalar_intent`
  - `bracket_{tax,game,store}` with `_unambig` and `_intent` variants
  - `power_scenario`
  - `evaluate` — post-hoc extraction/metrics
- **Pre-computed results:** `results/run_01/`, `run_02/`, `run_03/` contain Choi's own runs across all 14 models in the paper. These can be directly diffed against new runs for replication / regression checking.
- **Dependencies:** see `requirements.sh`. Uses Anthropic, OpenAI, Google, and HuggingFace Transformers backends.

### Suggested adaptation for this project
- The `scalar_max` variant *instructs* the model to "maximize the user's intent"; comparing `scalar` vs `scalar_max` rates is a clean test of whether explicit prompting alone removes loophole behavior.
- A new job `knowledge_loophole` could be added by following the pattern in `ambiguity.py`: present a question the model is unlikely to know, and offer an ambiguous escape interpretation (e.g., "name a Pulitzer-winning play from 1953" with the option to interpret "from 1953" as "published in or referencing 1953").

## 2. (Removed) `LLLMs` — Murthy et al. 2023

The original `github.com/skmur/LLLMs` repository is no longer accessible (Murthy's public repos do not include it as of 2026-05-18). The 36 stimulus scenarios were sourced from Bridgers et al. 2023 (`Cognition` 2025, Vol 261); Choi 2025 includes them in `code/ambiguous-loophole-exploitation/scenario_prompts.json` (extended to 108 with power-relationship variations). The human ratings from Murthy 2023's evaluation task are reported in the paper's Section 3 / Figure 2 but are not available as a downloadable file.

## Notes
- Both `subterfuge` (Denison 2024) and most knowledge-boundary papers do not release runnable training pipelines (the Anthropic curriculum is internal). We rely on their experimental designs as inspiration rather than direct re-use.
- Clone additional repos on demand:
  ```bash
  git clone --depth 1 https://github.com/<owner>/<repo>.git code/<repo>
  ```
