# Paper 2, Stage 1 — Judge Improvements (Phase 0-3)

[← Back to README](../README.md)

Quantifies how lenient and how stable the LLM judge is when grading plain classification predictions, then tests whether a different judge template fixes it. Standard workflow going forward uses the **thinking** variant (`RedHatAI/gemma-4-31B-it-FP8-dynamic` with `--enable_thinking`) as the checkpoint under test.

- **Phase 0 — generate the checkpoint** (with thinking enabled).
- **Phase 1 — leniency**: does the judge give more credit to the full free-text response than to just the resolved class name?
- **Phase 2 — self-consistency**: does the judge's verdict flip across repeat calls at higher temperature?
- **Phase 3 — template sensitivity**: compares 4 judge templates (`current` / `specificity` / `class_list` / `hybrid`) by re-running Phase 1 + 2 under each. `hybrid` is the winner and the pipeline-wide default — see below.

## Phase 0 — generate the checkpoint (thinking enabled)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skip_judge \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_thinking/
```

Gemma-4's own bundled chat template has no `enable_thinking` support — `--chat_template_path` (vendored at `src/config/tool_chat_template_gemma4.jinja`) is required alongside `--enable_thinking` for it to actually produce a `<|channel>thought...<channel|>` trace (Qwen models don't need this — their own template already supports `enable_thinking` directly). `--gen_max_tokens 4096` matters: at `1024`, a real thinking trace routinely truncates before reaching `<answer>`.

The checkpoint's `raw_response` column holds the fully untouched generation; `reasoning` holds the extracted thinking text; `prediction`/`extracted_answer` never see leftover channel markers.

**Splitting across multiple GPUs**: `--num_shards N --shard_index i` partitions the work into N independent single-GPU processes (each writes its own `ckpt_<model>_shard<i>ofN.csv`); `--merge_shards` combines them into the canonical `ckpt_<model>.csv` afterward. See `vision-baseline --help` for details. **Don't leave shard files sitting alongside the merged canonical file when pointing Phase 1-3 at that directory** — their `--ckpt` glob matches both and silently double-judges every row (fixed to exclude `*.bak.csv` but still matches shard files unless they're moved/deleted after merging).

## Baseline with rubric

An additional Phase 0 variant, not a replacement — same single-shot pipeline, no tournament/verifier machinery, just a different prompt. `--rubric` wraps the classification question through `verifier_prompts.yaml`'s `structured_templates` (the same model-facing, fill-in-the-blank criteria chain `verifier-cot`'s own k-candidate generation uses), asking the model to state each rubric field (affected species, symptom color/shape/texture, damage pattern, etc.) before its final `<answer>`. Same prompt shape as Paper 1's Appendix A "Manual CoT" condition, which found only a modest, inconsistent gain on its own (Δ=0.008 overall) — this reruns that idea on the current thinking-enabled model to see if it still holds.

Combines freely with `--enable_thinking` — they're independent (rubric changes what's asked, thinking changes whether the model can reason before answering):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skip_judge \
    --rubric \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_rubric_thinking/
```

Raise `--gen_max_tokens` regardless of `--enable_thinking` — the rubric prompt alone asks for several fields before the final answer, on top of any thinking trace (a printed `[rubric]` note fires if `--gen_max_tokens` is left at the flat 256 default).

## Phase 1 — leniency

```bash
uv run python main.py judge-leniency \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_leniency/vision_baseline_thinking/ \
    --label "Vision Baseline (thinking) — Gemma-4-31B-it-FP8"
```

Writes `leniency_gap_T<temp>.png`, `leniency_gap_by_temperature.png`, `disagreement_matrix_T<temp>.png`, `missing_answer_tag.txt`, `rechecked_T<temp>.csv`, `leniency_gap.json`.

## Phase 2 — self-consistency

```bash
uv run python main.py judge-consistency \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_consistency/vision_baseline_thinking/ \
    --label "Vision Baseline (thinking) — Gemma-4-31B-it-FP8" \
    --leniency_gap_json ablation/judge_leniency/vision_baseline_thinking/leniency_gap.json
```

Runs both conditions (full response / `<answer>`-tag-only) over Phase 1's same rows. `--leniency_gap_json` (optional) adds a leniency-vs-flip-rate correlation plot.

## Phase 3 — template sensitivity

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir outputs_final/judge_template_sensitivity/vision_baseline_thinking/ \
    --template hybrid \
    --skip_consistency
```

`--judge_model` defaults to `RedHatAI/gemma-4-31B-it-FP8-dynamic` and `--temperatures` defaults to `[0.7]`, but **judge thinking is off by default** (`--judge_enable_thinking`, plain `store_true`, matching every other thinking flag in this repo) — pass it explicitly, along with `--judge_max_tokens` (raise past the default `1024`) and `--judge_chat_template_path` (required alongside `--judge_enable_thinking` for a Gemma-4 judge, whose stock template has no thinking support at all):

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir outputs_final/judge_template_sensitivity/vision_baseline_thinking/ \
    --template hybrid \
    --skip_consistency \
    --judge_enable_thinking --judge_max_tokens 2048 \
    --judge_chat_template_path src/config/tool_chat_template_gemma4.jinja
```

Same three flags (`--judge_enable_thinking`/`--judge_max_tokens`/`--judge_chat_template_path`) work identically on `judge-leniency`, `judge-consistency`, and `vision-baseline`'s own built-in judge phase.

Every judged row carries `judge_raw_response` (the judge's completely untouched output) alongside the `<think>`-stripped `judge_reasoning`. The raw response is run through `extract_channel_thought` before JSON-verdict parsing, so stray `{`/`}` in the judge's own thinking prose can't corrupt the parse.

`--template {all,current,specificity,class_list,hybrid}` (default `all`) — run one variant per invocation against the **same** `--output_dir` and the cross-variant summary (`template_sensitivity.png`/`.tex`) backfills progressively as each finishes.

`hybrid` (`judge_classification_hybrid_template`, `src/config/prompts.yaml`) is the template that won this comparison — it credits a prediction when it **necessarily belongs to exactly one listed class** (`moldy` ⊂ `Bad`) while still rejecting a plausible-but-unproven cause guess for a generic symptom label. It's now `src.common.metrics.DEFAULT_JUDGE_TEMPLATE_KEY` — every judge call in the repo defaults to it unless a different template is explicitly requested (e.g. `--template_key judge_classification_template` for the original).

**`--skip_consistency`** skips Phase 2 (the most expensive part — `n_repeats` calls per row per condition per temperature) when only the leniency/accuracy comparison is needed; drops one variant's judge-call count from ~13x the dataset size to ~3x. Only cost: no `template_sensitivity_table.tex` "Flip rate" column, no `leniency_vs_flip_rate_*.png`.

**End-of-run outputs** (written once `current` and/or `hybrid` have data on disk):
- `judge_accuracy_f1.png` — judge accuracy + judge-F1 per variant, with CI.
- `leniency_and_false_positive_summary.png` (+ `_stats.json`) — leniency_gap and false-positive credit rate, read from `--summary_temperature` (default `0.7`; must be one of `--temperatures`).
- `oracle_validation_sample.csv` — up to 200 rows for manual adjudication, joining `hybrid`/`current` verdicts. Superseded by `judge-qa-export` below for a more deliberately-sampled version.

## Backfilling a checkpoint's own `judged_correct`

Once you've compared templates and decided which one's **full-response** verdict to trust, copy it into the checkpoint (useful for a `--skip_judge` checkpoint, which starts with blank `judged_correct`):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/ \
    --backfill_judge_results \
    --rechecked_csv outputs_final/judge_template_sensitivity/vision_baseline_thinking/hybrid/leniency/rechecked_T0.7.csv
```

Joins on `(agml_dataset, class_label, image_name)`; unmatched rows are left unchanged. Also auto-generates `task_f1_summary.png` (judged vs. strict F1 per task_type, with CI) in the same model directory.

**Comparing two templates directly** (e.g. before deciding which to backfill):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking/ \
    --compare_templates \
    --template_rechecked_csv \
        current=outputs_final/judge_template_sensitivity/vision_baseline_thinking/current/leniency/rechecked_T0.7.csv \
        hybrid=outputs_final/judge_template_sensitivity/vision_baseline_thinking/hybrid/leniency/rechecked_T0.7.csv
```

Writes `template_comparison.png` — judged F1 per task_type, one bar per template, read directly from each template's own `rechecked_T<temp>.csv` (deduped), so it works even when only one template is currently backfilled into the checkpoint.

## Judge QA review export

Builds a review-ready CSV for a human reviewer to hand-check judge verdicts, sampling three pools per `task_type`: `targeted_hybrid` (hybrid disagrees with a cheap extracted-answer-vs-ground-truth text check — calibrates hybrid's real failure rate), `disagreement` (current vs. hybrid give different verdicts — which template is actually right), `random_control` (unconditioned baseline). See `src/judge_qa_export.py`'s module docstring for the full pool logic.

```bash
uv run python main.py judge-qa-export --n_per_task 50 --seed 42
```

Writes to `dataset/judge_qa_review/judge_qa_review.csv` by default — `dataset/` (not `outputs_final/`, which is entirely gitignored) so the sampled review set is reproducible from the repo.

## Notes

- Phases 1-3 need a judge model loaded (`--judge_model`, default `RedHatAI/gemma-4-31B-it-FP8-dynamic`; thinking off by default, `--judge_enable_thinking` to turn it on) — GPU required.
- Every rate reported carries a bootstrap 95% CI (resampled from rows already on disk, no extra judge calls).
- `--temperatures` / `--n_repeats` control cost directly.
