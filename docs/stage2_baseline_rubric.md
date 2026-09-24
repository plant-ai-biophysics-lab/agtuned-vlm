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
    --num_shards 4 --shard_index $SHARD_INDEX \
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

Run the 4 shard tasks via your own cluster's job scheduler — one GPU per
task, `$SHARD_INDEX` set per task and passed as `--shard_index` — omitted
here, scheduler config is environment-specific.

## Step 2 — merge shards (single job, after ALL array tasks finish)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_rubric_thinking/ \
    --merge_shards
```

Each shard already carries `judged_correct`/`judge_reasoning` (from Step
1's inline judging), so the merged canonical checkpoint is fully judged
immediately — no separate judging-only pass needed. If Steps 1–2 are
submitted as separate jobs, make the merge job depend on ALL Step 1 array
tasks succeeding first (not just the first one to finish) — the exact
mechanism depends on your scheduler.

## Step 3 — verifier-cot over the same images, with the verifier fixes applied

A brief, informal revisit of the paused verifier diagnosis
(`docs/legacy/stage2_verifier_corrections_paused.md`) — not a full resumption
of that investigation, just applying the two live fixes it identified to a
real run over the *same image set* as the rubric baseline above, so the
results are directly comparable:

- **Sum-not-max letter aggregation** (`src/common/ppt_scoring.py`'s
  `extract_tag_score`) — live now, not diagnostic-only. Unconfirmed
  whether this actually helps (see its own docstring) — this run is part
  of how you'd find out.
- **Rewritten pairwise prompt** (`src/config/verifier_prompts.yaml`'s
  `pairwise_prompt`) — criteria-satisfaction-band anchors (A-C/D-H/I-M/N-T)
  + a forced-separability rule between Trajectory A/B scores.
- **More candidates/pivots** — now that more GPUs are available, raised
  from the earlier diagnostic run's `--k_candidates 3 --k_pivots 2` (closer
  to the paper's own `K ∈ {4, 8}`) — directly targets the pool-diversity
  and winner's-curse findings from that diagnosis (41% of pools were fully
  degenerate at K=3; more independent draws per image makes that less
  likely, and more comparisons per candidate reduces small-sample argmax
  noise). `--temperature` is left at its default (`1.0`, unchanged) so this
  run isolates the effect of the scoring/prompt/K changes rather than
  conflating them with a temperature change too.
- **`--num_shards`/`--shard_index` now exist on `verifier-cot`** (added
  alongside this run — previously only `vision-baseline` had them), same
  array-job convention. `--chat_template_path` was also added to
  `verifier-cot`'s own k-candidates generation step, since without it
  `--enable_thinking` was a silent no-op for Gemma-4 here too (same root
  cause `vision_baseline.py`'s fix addressed) — now fixed. `--judge_enable_thinking`/
  `--judge_max_tokens`/`--judge_chat_template_path` were also missing from
  `verifier-cot`'s own CLI (the underlying `run_judge_phase` call already
  supported them via `getattr`, the flags just weren't registered) — fixed.
- **`src/config/verifier_config.yaml`'s defaults now match
  `vision_baseline_config.yaml`'s** — `manifest: dataset/selected_images.json`
  (the full manifest, not the smaller `selected_images_10.json` this file
  used to default to) and `n_samples: null` (uncapped), so a bare
  `--match_images_from` invocation is automatically full apples-to-apples
  with any vision-baseline run, no `--manifest`/`--n_samples` flags needed.
  This matters specifically because `load_hf_samples` scopes strictly to
  what the manifest lists — pointing `--match_images_from` at a 50/class
  reference checkpoint while the manifest itself only has 10/class silently
  caps every class at 10 regardless of `--n_samples`, since there's nothing
  beyond that in the manifest to draw from. `dataset/selected_images_10.json`
  still exists for a deliberately smaller/cheaper run — pass `--manifest`
  explicitly to opt into it.

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --match_images_from outputs_final/vision_baseline_rubric_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --k_candidates 5 --k_pivots 3 --temperature 1.0 \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_rubric_verifier_thinking/ \
    --num_shards 4 --shard_index $SHARD_INDEX \
    --judge_model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --judge_enable_thinking --judge_max_tokens 2048 \
    --judge_chat_template_path src/config/tool_chat_template_gemma4.jinja
```

`--manifest`/`--n_samples` are deliberately omitted — they now come from
`verifier_config.yaml`'s own defaults (full manifest, uncapped), which is
what makes this a true apples-to-apples match against the rubric baseline
without needing to remember either flag.

`--match_images_from` guarantees this run scores the exact same images the
rubric baseline used (not an independent sample), so any comparison between
them is apples-to-apples. Run the array tasks the same way as Step 1 (your
own scheduler, one GPU per task, `--shard_index` set per task).

Then merge (single job, after all 4 tasks finish):

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_rubric_verifier_thinking/ \
    --merge_shards
```

Then diagnose whether the fixes actually moved anything:

```bash
uv run python main.py verifier-score-analysis \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_rubric_verifier_thinking/
```

Compare `score_analysis_summary.json`'s `image_level.point_biserial_r` and
`pair_level.granularity.max_vs_sum_mass_gap` against the paused doc's
post-mortem numbers (`r=-0.097` image-level, `r=+0.076` pair-level, at
K=3) to see whether K=5/pivots=3 plus the two live fixes actually closed
the gap — this is genuinely unknown until this run exists.

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
