# Paper 2, Stage 2 — Baseline with Rubric

[← Back to README](../README.md)

An additional baseline, not a replacement for Stage 1's plain
thinking-enabled checkpoint — same single-shot pipeline (`vision-baseline`),
no tournament/verifier machinery, just a different prompt. `--rubric` wraps
the classification question through `src/config/verifier_prompts.yaml`'s
`structured_templates` (the same model-facing, fill-in-the-blank criteria
chain `verifier-cot`'s own k-candidate generation already uses — task-type
keyed, asking for e.g. affected species / symptom color-shape-texture /
damage pattern before the final diagnosis). This is the same prompt shape
as Paper 1's Appendix A "Manual CoT" condition, which found only a modest,
inconsistent gain on its own (overall Δ=0.008 vs. the lower bound) — this
reruns that idea on the current thinking-enabled model to see if it still
holds. (Paper 2's earlier Stage 2 slot, a PPT-verifier score diagnosis, is
paused — see `docs/legacy/stage2_verifier_corrections_paused.md`.)

`--rubric` combines freely with `--enable_thinking` — they're independent
(rubric changes what's asked, thinking changes whether the model can reason
before answering).

## Step 1 — generate (data-parallel across GPUs, judging inline)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --rubric \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_rubric_thinking/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID \
    --judge_model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --judge_enable_thinking --judge_max_tokens 2048 \
    --judge_chat_template_path src/config/tool_chat_template_gemma4.jinja
```

No `--skip_judge` here, deliberately — each of the `--num_shards` array
tasks generates its own shard, then runs its own judge phase over only its
own shard's rows, on its own GPU (`run_judge_phase` always runs after
generation unless `--skip_judge` is set, and each shard has its own
checkpoint file — see `vision_baseline.py`'s module docstring). This gives
genuine multi-GPU **judging** parallelism for free, using the same
sharding mechanism as generation — no separate single-GPU judging pass
needed afterward. `run_judge_phase`'s own log file
(`judge_log.jsonl`) is shard-aware too (`judge_log_shard<i>of<n>.jsonl`)
so concurrent shard tasks never write-collide on it.

As a SLURM job array (4 tasks, one GPU each):

```bash
#!/bin/bash
#SBATCH --job-name=agtuned-vlm-rubric-thinking
#SBATCH --output=farm_outputs/%A_%a.out
#SBATCH --error=farm_outputs/%A_%a.err
#SBATCH --account=jmearlesgrp
#SBATCH --partition=gpu-6000_ada-h
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00
#SBATCH --array=0-3

uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --rubric \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_rubric_thinking/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID \
    --judge_model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --judge_enable_thinking --judge_max_tokens 2048 \
    --judge_chat_template_path src/config/tool_chat_template_gemma4.jinja
```

## Step 2 — merge shards (single job, after ALL array tasks finish)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_rubric_thinking/ \
    --merge_shards
```

Each shard already carries `judged_correct`/`judge_reasoning` (from Step
1's inline judging), so the merged canonical checkpoint is fully judged
immediately — no separate judging-only pass needed. If submitting Steps
1–2 as separate `sbatch` scripts rather than one script that runs both
sequentially, use a real dependency so the merge doesn't start before every
array task lands:

```bash
JOB1=$(sbatch --parsable step1_generate_array.sh)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} step2_merge.sh)
```

`afterok:${JOB1}` on an array job waits for **every** array task to
succeed, not just the first to finish.

## Notes

- Raise `--gen_max_tokens` regardless of `--enable_thinking` — the rubric
  prompt alone asks for several fields before the final answer, on top of
  any thinking trace. A printed `[rubric]` note fires if `--gen_max_tokens`
  is left at the flat `256` default.
- `--num_shards`/`--shard_index` and judging are otherwise independent of
  `--rubric` — nothing here is rubric-specific except the prompt itself;
  the multi-GPU/judge-log-collision fixes apply to any sharded
  `vision-baseline` run, rubric or not.
- Once judged, this checkpoint slots into the same downstream tooling as
  the plain thinking baseline — `vision-baseline --backfill_judge_results`
  (to trust a different template's verdict), `--compare_templates`, and
  `judge-qa-export` (point `--current_csv`/`--hybrid_csv` at a
  `judge-template-sensitivity` run over this checkpoint, same as Stage 1)
  all work unchanged.
