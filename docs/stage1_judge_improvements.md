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
    --model google/gemma-4-E4B-it \
    --rerun
```

`--rerun` archives any existing checkpoint as `ckpt_<model>.<timestamp>.bak.csv` and starts fresh — required here, since resuming without it would just skip every already-checkpointed row and keep their old (pre-tagged-prompt) predictions. This also runs the judge phase, so the output is ready for Phase 1 directly: `outputs_final/vision_baseline/<safe_model_name>/`.

## Phase 1 — leniency

```bash
uv run python main.py judge-leniency \
    --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \
    --output_dir ablation/judge_leniency/vision_baseline_e4b/ \
    --label "Vision Baseline — Gemma-4-E4B-it"
```

Writes `leniency_gap_T<temp>.png` (one per swept temperature, default `0.1 0.5 1.0`), `leniency_gap_by_temperature.png`, `resolve_failures.txt`, and `leniency_gap.json` (Phase 2 reads this back for the correlation plot).

## Phase 2 — self-consistency

```bash
uv run python main.py judge-consistency \
    --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \
    --output_dir ablation/judge_consistency/vision_baseline_e4b/ \
    --label "Vision Baseline — Gemma-4-E4B-it" \
    --leniency_gap_json ablation/judge_leniency/vision_baseline_e4b/leniency_gap.json
```

`--leniency_gap_json` is optional but recommended — pass Phase 1's own output (same `--ckpt`) to also get `leniency_vs_flip_rate.png`. Writes `flip_rate_by_temperature.png`, `flip_rate.json`, and (if `--leniency_gap_json` given) `leniency_vs_flip_rate.png` + `leniency_vs_flip_rate_stats.json`.

## Phase 3 — template sensitivity

```bash
uv run python main.py judge-template-sensitivity \
    --ckpt outputs_final/vision_baseline/google_gemma-4-E4B-it/ \
    --output_dir ablation/judge_template_sensitivity/vision_baseline_e4b/ \
    --label "Vision Baseline — Gemma-4-E4B-it" \
    --limit 500
```

Re-runs Phase 1 + 2 in full for each of 3 templates (current / specificity-strict / +full-class-list) — by far the most expensive step, so start with `--limit` on a small subsample before a full run. Writes each variant's complete Phase 1+2 output under `<output_dir>/<variant>/`, plus a top-level `template_sensitivity.png`, `length_bias.png`, and `template_sensitivity_table.tex` (LaTeX summary table across variants x temperatures, ready to paste into the paper).

## Notes

- Phases 1-3 each need a judge model loaded (`--judge_model`, default `google/gemma-4-26B-A4B-it`; `--judge_quantize` for `4bit`/`8bit`/`awq`/`gptq`) — GPU required, not a local-CPU step.
- Every rate reported (leniency_gap, flip_rate, judged_correct_rate) carries a bootstrap 95% CI, shown as error bars on every figure — computed by resampling rows already on disk, no extra judge calls.
- `--temperatures` (default `0.1 0.5 1.0`) and `--n_repeats` (Phase 2/3, default `5`) control cost directly — fewer/lower temperatures and fewer repeats run faster.
