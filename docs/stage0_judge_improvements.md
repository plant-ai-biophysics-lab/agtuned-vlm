# Paper 2, Stage 0 — Judge

[← Back to README](../README.md)

Grade an existing baseline checkpoint's predictions with the judge —
`hybrid` is the pipeline-wide default template
(`DEFAULT_JUDGE_TEMPLATE_KEY`, `src/common/metrics.py`;
`judge_classification_hybrid_template` in `prompts.yaml`) — every judge
call in the repo uses it unless a different template is explicitly
requested. Run against the checkpoint [Stage 1](stage1_baseline_classification.md)
generates — run that first if it doesn't exist yet.

## Judge a checkpoint

```bash
uv run python main.py judge-checkpoint \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/judge_variants/hybrid_v1/
```

One judging pass, same mechanism `vision-baseline`'s own inline judge
phase uses (`run_judge_phase`, reused directly) — no
leniency/consistency/template-sensitivity comparison. `--ckpt` is never
modified in place: it's copied into `--output_dir` first, then judged
there.

**Trying different hybrid-template wording**: hand-edit
`prompts.yaml`'s `judge_classification_hybrid_template`, then re-run
pointed at a **fresh** `--output_dir` with `--rejudge` (forces every row
to be re-graded under the new wording, not just ones still missing a
verdict):

```bash
uv run python main.py judge-checkpoint \
    --ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/judge_variants/hybrid_v2/ \
    --rejudge
```

A different `--output_dir` per variant means each one's judged checkpoint
+ `judge_log.jsonl` land in their own folder — nothing overwrites a
previous variant's results, so you can compare `judge_accuracy` across
wording changes directly.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--ckpt` | *(required)* | Checkpoint CSV to judge — never modified in place |
| `--output_dir` | *(required)* | Where the copied checkpoint + `judge_log.jsonl` are written |
| `--rejudge` | off | Re-grade every row (not just ones missing a verdict) — use after changing the template wording |
| `--judge_model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` | Judge model |
| `--judge_enable_thinking` | off | Turn on judge thinking (needs `--judge_chat_template_path` for Gemma-4) |
| `--judge_max_tokens` | `1024` | Judge generation budget (raise with thinking on) |
| `--judge_chat_template_path` | none | Custom chat template (required for Gemma-4 + thinking) |
