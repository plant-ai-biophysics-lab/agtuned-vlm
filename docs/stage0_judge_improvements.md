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

## Judge QA review export

After trying a different hybrid-template wording (the "Trying different
hybrid-template wording" command above), see whether it actually changed
anything a human would care about: `judge-qa-export` samples every row
where the two runs' `judged_correct` verdict **disagrees**, stratified by
`task_type` (`--n_per_task`, default 50 per task_type). No GPU, no model
calls — pure post-hoc comparison of two already-judged checkpoint CSVs.

```bash
uv run python main.py judge-qa-export \
    --baseline_ckpt outputs_final/vision_baseline_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --variant_ckpt outputs_final/judge_variants/hybrid_v2/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_csv dataset/judge_qa_review/judge_qa_review_v2.csv
```

`--baseline_ckpt` and `--variant_ckpt` must be the same underlying rows
judged twice under different template wordings — both are
`VISION_BASELINE_CKPT_FIELDS`-shaped (either a `vision-baseline`
checkpoint itself, or a `judge-checkpoint` output copied from one), so
`judged_correct`/`judge_reasoning`/`prediction`/`raw_response`/etc. all
line up by column name with no separate join needed.

Name `--output_csv` after the variant being reviewed
(`judge_qa_review_v2.csv` for a `hybrid_v2` comparison, `_v3` for the
next wording change, etc.) so re-running against a later variant never
overwrites an earlier review. Output lands in `dataset/` (git-tracked),
not `outputs_final/` (gitignored).

Each output row carries both verdicts side by side
(`baseline_judged_correct`/`baseline_judge_reasoning`,
`variant_judged_correct`/`variant_judge_reasoning`, each with its own
extracted `*_judge_confidence`) plus the shared
`prediction`/`raw_response`/`ground_truth`/`extracted_answer`, and blank
reviewer columns to fill in by hand: `reviewer_verdict`,
`visual_detail_score` (1-5, how much concrete visual evidence the
model's own reasoning cites), `categorical_score` (0/1/2, how far the
prediction is from ground truth — specificity gap vs. causal-category
mismatch, mirroring Paper 1 Appendix C.1), `failure_category`,
`which_judge_wrong` (`baseline`/`variant`/`both`/`neither`), `notes`.
