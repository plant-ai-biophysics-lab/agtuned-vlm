# Paper 2, Stage 1 — Judge Improvements (Phase 0-3)

[← Back to README](../README.md)

Quantifies how lenient and how stable the LLM judge is when grading plain classification predictions (`vision-baseline` / `desc-aug-v2` / `self-cot` / `cot-yaml`), then tests whether a stricter judge template fixes it.

- **Phase 0 — regenerate the checkpoint**: re-run `vision-baseline` so its classification prompt is the tagged version (`assessment_classification_tagged`, `<answer></answer>` + `max_tokens=256`) that Phases 1-3 depend on for a low unresolved rate.
- **Phase 1 — leniency**: does the judge give more credit to the model's full free-text response than to just the class name it resolves to?
- **Phase 2 — self-consistency**: does the judge's verdict flip across repeat calls at higher temperature?
- **Phase 3 — template sensitivity**: does a stricter/specificity-focused template, or showing the judge the full class list, reduce leniency? Re-runs Phase 1 + 2 under each template variant.

## Phase 0 — regenerate the checkpoint

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --rerun
```

`--rerun` archives any existing checkpoint as `ckpt_<model>.<timestamp>.bak.csv` and starts fresh — required here, since resuming without it would just skip every already-checkpointed row and keep their old (pre-tagged-prompt) predictions. This also runs the judge phase, so the output is ready for Phase 1 directly: `outputs_final/vision_baseline/<safe_model_name>/` (for this model, `outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/`). No `--quantize` needed — this checkpoint's FP8 quantization is baked in and vLLM auto-detects it.

By default, this (and every other generation script in the repo) suppresses the model-being-evaluated's own extended thinking (`<think>`/`<thought>` tags) for the classification step — see `src/common/chat_prompt.py`'s module docstring for why. To let it through instead, add `--enable_thinking` (and raise `--gen_max_tokens`, default `256`, substantially — e.g. `1024`+ — since a thinking trace can consume the budget before the model reaches its `<answer>` tag):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skip_judge \
    --enable_thinking --gen_max_tokens 1024
```

This is a **different** flag from `--judge_enable_thinking` (below) — this one controls the model being evaluated's own generation call in Phase 0, that one controls the separate judge model's call in Phases 1-3. Whether Gemma-4's tokenizer actually exposes a controllable thinking mode via `enable_thinking=True`/`False` kwargs, or uses a different channel-based reasoning format entirely, is unverified without running it — check by diffing a `--enable_thinking` run's rendered prompt/raw prediction text against a default run's.

The checkpoint now has a `raw_response` column (right after `prediction`) holding the model's completely untouched generation output, before `<think>`/`<thought>` tags are stripped into `prediction`. This exists because thinking isn't guaranteed to show up as a tagged block at all — real Gemma-4 output has been observed as an unwrapped reasoning paragraph sitting directly before `<answer>`, which the tag-stripping regex doesn't touch and wouldn't distinguish from a normal answer. `raw_response` preserves exactly what the model said regardless of tagging convention, so it's the column to actually read when checking whether/how a model reasons — `prediction` alone can't be trusted to show or hide reasoning consistently across models. Blank for any pre-existing checkpoint row (backfilled automatically by `load_ckpt`'s schema-drift healing, same as `extracted_answer` was before it).

## Phase 1 — leniency

```bash
uv run python main.py judge-leniency \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_leniency/vision_baseline_gemma4_31b_fp8/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8"
```

Writes `leniency_gap_T<temp>.png` (one per swept temperature, default `0.1 0.5 1.0`), `leniency_gap_by_temperature.png`, `disagreement_matrix_T<temp>.png` (the raw disagreement counts behind `leniency_gap`'s single net rate — a 2x2 breakdown: both correct, both wrong, full-correct/resolved-wrong, full-wrong/resolved-correct), `missing_answer_tag.txt` (how many rows had no `<answer></answer>` tag at all — excluded from every temperature's comparison), `rechecked_T<temp>.csv`, and `leniency_gap.json` (Phase 2 reads this back for the correlation plot; also carries the disagreement counts for Phase 3's table).

## Phase 2 — self-consistency

```bash
uv run python main.py judge-consistency \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_consistency/vision_baseline_gemma4_31b_fp8/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8" \
    --leniency_gap_json ablation/judge_leniency/vision_baseline_gemma4_31b_fp8/leniency_gap.json
```

Runs BOTH conditions — full response and `<answer>`-tag-only — over the same row set Phase 1 uses, so `flip_rate_full` and `flip_rate_answer_only` are directly comparable (this doubles the judge-call budget vs. a single condition). `--leniency_gap_json` is optional but recommended — pass Phase 1's own output (same `--ckpt`) to also get a correlation scatter per condition. Writes `flip_rate_by_temperature_full.png`, `flip_rate_by_temperature_answer_only.png`, `flip_rate.json`, and (if `--leniency_gap_json` given) `leniency_vs_flip_rate_full.png` + `leniency_vs_flip_rate_answer_only.png` + their stats JSON files.

## Phase 3 — template sensitivity

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8" \
    --limit 500
```

Re-runs Phase 1 + 2 in full for each of 4 templates (`current` / `specificity` / `class_list` / `hybrid` — specificity-strict, +full-class-list, and Phase 4's hybrid respectively) — by far the most expensive step, so start with `--limit` on a small subsample before a full run. Writes each variant's complete Phase 1+2 output under `<output_dir>/<variant>/`, **including its own `disagreement_matrix_T<temp>.png`** (the full-response/resolved-class breakdown shown above, one per variant — not a single pooled figure), plus a top-level `template_sensitivity.png`, `length_bias.png`, and `template_sensitivity_table.tex` — a LaTeX summary table across variants x temperatures (leniency_gap, flip_rate, and both disagreement-direction counts/rates), ready to paste into the paper.

### Phase 4 — hybrid template (specificity + class list, membership not exact match)

`current`/`specificity`/`class_list` (see `src/config/prompts.yaml`) established that `class_list` beats `specificity` on both false-positive credit rate (12.0% vs. 18.0% on deterministic-mismatch rows) and false-negative cost (4.2% vs. 14.2% of deterministic-*correct* rows wrongly rejected) — but `class_list`'s own literal-string matching is too rigid for `quality_identification`, where the ground truth is a coarse catch-all (`Bad`, ripeness stages) and the model gives a specific descriptive term (`moldy`, `shriveled`). `hybrid` (`judge_classification_hybrid_template`) keeps `class_list`'s context and `specificity`'s generic-label guardrail, but explicitly permits crediting a non-exact-match prediction when it **necessarily belongs to exactly one listed class** (`moldy` ⊂ `Bad` — always true) while still rejecting a **plausible-cause guess for a generic symptom label** (`leaf crinckle` → `Yellow Mosaic Virus` — only one of many possible causes, so naming it is a guess, not a necessary membership relationship). This is the same leniency failure mode `current` had, so the distinction is drawn explicitly in the template text rather than left implicit.

Run it exactly like the other three variants — no new orchestration code, `TEMPLATE_VARIANTS` (`src/common/judge_template_sensitivity.py`) is a plain list this slots into:

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8/ \
    --template hybrid
```

Pointed at the same `--output_dir` that already has `current`/`specificity`/`class_list` output, the cross-invocation summary backfill automatically folds `hybrid` into `template_sensitivity.png`/`.tex` alongside the other three without re-running them.

### `hybrid` is now the pipeline-wide default judge template

`src.common.metrics.DEFAULT_JUDGE_TEMPLATE_KEY` (currently `judge_classification_hybrid_template`) is the single point of control for which template a judge call defaults to when nothing else is specified — every active judge call in the repo (`vision-baseline`'s own judge phase, `desc-aug-v2`/`self-cot`/`cot-yaml` via the same `run_judge_phase`, `majority_vote.py`, `cot_vision.py`, `cot_step_analysis.py`, `verifier_lmv.py`, and `judge-leniency`/`judge-consistency`'s `--template_key` CLI default) now reads this constant instead of hardcoding `judge_classification_template`. The original template is completely unchanged and still fully selectable by name (e.g. `--template_key judge_classification_template`, or `TEMPLATE_VARIANTS`' own `"current"` entry, which intentionally always means the literal original template regardless of this default).

Since `hybrid` has a `{class_list}` placeholder, every one of those call sites now also builds a per-row class list automatically via `src.common.dataset_config.class_list_extra_keys(rows, Path(config_path))` before judging — each of `majority_vote.py`/`cot_vision.py` gained a `--config` flag (default `src/config/datasets.yaml`) for this, matching every other classification script's existing convention. This is safe regardless of which template ends up active — a template without a `{class_list}` placeholder just ignores the unused value.

### End-of-run Phase 4 outputs

Three additional outputs are written to `--output_dir` once at least `current` and/or `hybrid` have data on disk (same "skip with a printed note if data's missing, never a hard error" pattern as `length_bias.png`):

- **`judge_accuracy_f1.png`** — judge accuracy + judge-scored macro-F1 (both with bootstrap 95% CI) per variant, reusing `judged_summary`/`bootstrap_judge_f1_ci` (`src/common/vision_eval.py`) — no new metric logic, just a new comparison chart.
- **`leniency_and_false_positive_summary.png`** (+ `_stats.json`) — two-panel: left panel is `leniency_gap` (overall judged-correct rate over all rows, same metric `leniency_gap_T<temp>.png` plots, collapsed to "overall"), right panel is the false-positive credit rate restricted to deterministically-wrong rows only (isolates the disagreement cases from the true-positive agreement that otherwise dilutes the all-rows average). Both panels read from `--summary_temperature` (float, default `0.7`) — **this must be one of the values in that variant's own `--temperatures` sweep**, or that variant is skipped from this chart with a printed note (e.g. pass `--temperatures 0.1 0.5 0.7 1.0` to populate the default `0.7`).
- **`oracle_validation_sample.csv`** — samples up to 200 rows (70% from disagreement cases, the rest a random sample for base-rate context, stratified by `task_type` where possible) joining `hybrid` and `current`'s verdicts on the same `(agml_dataset, class_label, image_name)`, for a human reviewer to manually adjudicate as an oracle. Needs both `hybrid` and `current` to have run in this `--output_dir`. Columns include both templates' `judged_correct`/`judge_reasoning` side by side, an `agreement` flag, and two blank columns (`human_verdict`, `human_notes`) for the reviewer to fill in.

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir outputs_final/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8/ \
    --temperatures 0.1 0.5 0.7 1.0 \
    --summary_temperature 0.7 \
    --template hybrid
```

### Running one template variant per job

`--template {all,current,specificity,class_list,hybrid}` (default `all`) runs just one variant per invocation instead of all four in one long-running job:

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8" \
    --template current
```

Run again with `--template specificity`, then `--template class_list`, then `--template hybrid`, against the **same** `--output_dir` — each invocation writes its own `<output_dir>/<variant>/` output and persists a `greedy_summary_rechecked.csv` there. The cross-variant summary (`template_sensitivity.png`/`.tex`, `length_bias.png`) re-scans `--output_dir` for every variant with completed output on disk each time it's written, not just the one variant just run — so the summary progressively fills in as each job finishes, without re-judging anything already done. `length_bias.png` needs the `current` variant specifically and is skipped (with a printed note) until that one has run.

## Judge thinking mode (experimental)

Every phase script (`judge-leniency`, `judge-consistency`, `judge-template-sensitivity`) accepts:

- `--judge_enable_thinking` — attempts `tokenizer.apply_chat_template(..., enable_thinking=True)` for the judge's own prompt. **Off by default**, which reproduces the exact original prompt rendering with no behavior change. This is a no-op (silently falls back to the plain call) if the judge tokenizer's chat template doesn't accept the `enable_thinking` kwarg at all.
- `--judge_max_tokens` — overrides the judge's default `max_tokens=1024` budget. Useful alongside `--judge_enable_thinking`, since a thinking trace can consume the normal budget before the judge ever reaches its final JSON verdict, truncating the response mid-thought.

```bash
uv run python main.py judge-leniency \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_leniency/vision_baseline_gemma4_31b_fp8_thinking/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8 (thinking)" \
    --limit 50 \
    --judge_enable_thinking --judge_max_tokens 2048
```

### Logging the judge's own thinking tokens

Every row the judge scores now also carries a `judge_raw_response` column (alongside the existing `judge_reasoning`, which is `<think>`-stripped and truncated to 600 chars) — the judge's **completely untouched** output text, same rationale as `vision_baseline.py`'s `raw_response`: a thinking judge's reasoning isn't guaranteed to show up as a tag `judge_reasoning`'s stripping regex would ever match, so it's the column to actually read when checking whether/how the judge reasoned.

- `judge-leniency`'s `rechecked_T<temp>.csv` carries both `judge_raw_response` (the `<answer>`-tag-only condition) and `judge_raw_response_full_trajectory` (the full-response condition) — same `_full_trajectory` suffix convention as `judged_correct`/`judge_reasoning`.
- `judge-template-sensitivity`'s `<variant>/greedy_summary_rechecked.csv` carries `judge_raw_response` for that variant's greedy-temperature verdict.
- Every `*_judge_log.jsonl` (written whenever a judge call logs at all — `judge-leniency` always, `judge-consistency` only with `--save_logs`) already had the judge's full raw response in its own `raw_response` field before this, independent of `judge_raw_response` — the CSV column is new, the JSONL log always had it.
- `judge-consistency`'s flip-rate CSVs/JSON don't carry per-repeat raw text (only the 0/1 verdict per repeat) — use `--save_logs` and read the per-repeat JSONL logs if you need the judge's actual reasoning text for a specific repeat.

`vision-baseline`'s own built-in judge phase (the one that runs unless `--skip_judge` is set) now also accepts `--judge_enable_thinking`/`--judge_max_tokens`, same meaning as above:

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_judge_thinking/ \
    --judge_enable_thinking --judge_max_tokens 2048
```

This is a separate pair of flags from `--enable_thinking`/`--gen_max_tokens` (Phase 0's own generation-side thinking flags, see above) — `--judge_enable_thinking` only affects the judge model's call inside `run_judge_phase`, not the classification model's generation call. Not yet wired up for `desc_augmented_vision_v2.py`/`self_cot.py`/`cot_yaml.py`, which share the same `run_judge_phase` function but don't expose these flags on their own CLIs yet — ask if you need it there too.

This is verified so far only via mocked tests (`enable_thinking=True` correctly changes the rendered prompt when the tokenizer accepts the kwarg, and falls back cleanly when it doesn't) — whether Gemma-4's tokenizer actually exposes a controllable thinking mode via this kwarg, or uses a different channel-based reasoning format entirely, is unverified without running it. To check: run a small `--limit` job with `--judge_enable_thinking` and inspect a `*_judge_log.jsonl`'s `rendered_prompt` field — if it differs from a non-thinking run's rendered prompt (e.g. shows a distinct reasoning/channel marker), the kwarg is doing something; if it's identical, the kwarg is a no-op for this tokenizer.

## Notes

- Phases 1-3 each need a judge model loaded (`--judge_model`, default `google/gemma-4-26B-A4B-it`; `--judge_quantize` for `4bit`/`8bit`/`awq`/`gptq`) — GPU required, not a local-CPU step.
- Every rate reported (leniency_gap, flip_rate, judged_correct_rate) carries a bootstrap 95% CI, shown as error bars on every figure — computed by resampling rows already on disk, no extra judge calls.
- `--temperatures` (default `0.1 0.5 1.0`) and `--n_repeats` (Phase 2/3, default `5`) control cost directly — fewer/lower temperatures and fewer repeats run faster.
- `--judge_enable_thinking` / `--judge_max_tokens` (all three phase scripts, off/unset by default) — see "Judge thinking mode" above.
- `--summary_temperature` (`judge-template-sensitivity` only, default `0.7`) controls the end-of-run `leniency_and_false_positive_summary.png` — see "End-of-run Phase 4 outputs" above.
