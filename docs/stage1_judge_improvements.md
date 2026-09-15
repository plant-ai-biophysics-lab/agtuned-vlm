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

Re-runs Phase 1 + 2 in full for each of 3 templates (`current` / `specificity` / `class_list` — specificity-strict and +full-class-list respectively) — by far the most expensive step, so start with `--limit` on a small subsample before a full run. Writes each variant's complete Phase 1+2 output under `<output_dir>/<variant>/`, **including its own `disagreement_matrix_T<temp>.png`** (the full-response/resolved-class breakdown shown above, one per variant — not a single pooled figure), plus a top-level `template_sensitivity.png`, `length_bias.png`, and `template_sensitivity_table.tex` — a LaTeX summary table across variants x temperatures (leniency_gap, flip_rate, and both disagreement-direction counts/rates), ready to paste into the paper.

### Running one template variant per job

`--template {all,current,specificity,class_list}` (default `all`) runs just one variant per invocation instead of all three in one long-running job:

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ \
    --output_dir ablation/judge_template_sensitivity/vision_baseline_gemma4_31b_fp8/ \
    --label "Vision Baseline — Gemma-4-31B-it-FP8" \
    --template current
```

Run again with `--template specificity`, then `--template class_list`, against the **same** `--output_dir` — each invocation writes its own `<output_dir>/<variant>/` output and persists a `greedy_summary_rechecked.csv` there. The cross-variant summary (`template_sensitivity.png`/`.tex`, `length_bias.png`) re-scans `--output_dir` for every variant with completed output on disk each time it's written, not just the one variant just run — so the summary progressively fills in as each job finishes, without re-judging anything already done. `length_bias.png` needs the `current` variant specifically and is skipped (with a printed note) until that one has run.

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

This is verified so far only via mocked tests (`enable_thinking=True` correctly changes the rendered prompt when the tokenizer accepts the kwarg, and falls back cleanly when it doesn't) — whether Gemma-4's tokenizer actually exposes a controllable thinking mode via this kwarg, or uses a different channel-based reasoning format entirely, is unverified without running it. To check: run a small `--limit` job with `--judge_enable_thinking` and inspect a `*_judge_log.jsonl`'s `rendered_prompt` field — if it differs from a non-thinking run's rendered prompt (e.g. shows a distinct reasoning/channel marker), the kwarg is doing something; if it's identical, the kwarg is a no-op for this tokenizer.

## Notes

- Phases 1-3 each need a judge model loaded (`--judge_model`, default `google/gemma-4-26B-A4B-it`; `--judge_quantize` for `4bit`/`8bit`/`awq`/`gptq`) — GPU required, not a local-CPU step.
- Every rate reported (leniency_gap, flip_rate, judged_correct_rate) carries a bootstrap 95% CI, shown as error bars on every figure — computed by resampling rows already on disk, no extra judge calls.
- `--temperatures` (default `0.1 0.5 1.0`) and `--n_repeats` (Phase 2/3, default `5`) control cost directly — fewer/lower temperatures and fewer repeats run faster.
- `--judge_enable_thinking` / `--judge_max_tokens` (all three phase scripts, off/unset by default) — see "Judge thinking mode" above.
