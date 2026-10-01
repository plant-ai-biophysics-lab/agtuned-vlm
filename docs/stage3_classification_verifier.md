# Paper 2, Stage 3 — Classification Verifier (Rubric Grading)

[← Back to README](../README.md)

Grade model trajectories against each rubric criterion independently on a
1-3 scale, sum and normalize to a `verifier_score` in `[0, 1]`. No
pairwise duel, no tournament. Two modes, run in order:

- **Step 1 — `--verifier_type simple`** (the default): grade **one**
  trajectory per image. Does this coarse, cheap score correlate with
  correctness at all?
- **Step 2 — `--verifier_type top3`**: grade **three** independently
  generated trajectories per image, report the highest-scoring one. Is
  three trajectories better than one, and when the right answer is
  generated, does the verifier's own grading actually find it?

## Step 1 — one trajectory (`--verifier_type simple`)

```bash
uv run python main.py verifier-cot --model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

That's it — no other flags needed. Defaults do everything:
- `--verifier_type simple` (the default) — one trajectory, per-criterion
  grading, no tournament.
- Thinking is **on by default** — same standard workflow as [Stage 1](stage1_baseline_classification.md)'s
  classification baseline. `--chat_template_path` defaults to the vendored
  Gemma-4 template alongside it; `--gen_max_tokens` defaults to `4096`, enough
  headroom that a real thinking trace won't truncate before reaching
  `<answer>`. Pass `--no-enable_thinking` to opt out.
- Images auto-match an existing `vision-baseline` run for the same
  `--model`, if one exists, so results are directly comparable.
- Judging runs automatically unless you pass `--skip_judge`.

Swap in whatever model you're testing for `--model`. Everything else
(`--output_dir`, `--subsample`, `--num_shards`, etc.) is there if you need
it later — see `--help` — but you don't need to touch it to get a result.

**Faster — grade an existing baseline run instead of regenerating:** if
you already have a `vision-baseline --rubric` checkpoint for this model,
skip generation entirely and just grade what it already produced:

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --from_ckpt outputs_final/vision_baseline_rubric_thinking/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

One flag. It grades that checkpoint's own trajectories directly (one
grading call per image instead of a generation call + a grading call),
and automatically samples the same images that checkpoint used — nothing
else to set. `--from_ckpt` only applies to `--verifier_type simple` (there's
no single existing trajectory to replay for `top3`'s 3-candidate pool).

**Read the result**: look at the end of the run's own printed output, or
open `outputs_final/verifier_cot/<safe_model_name>/metrics.json` and check
`verifier_score_correlation`:

```json
"verifier_score_correlation": {
  "n": 480,
  "against": "judged_correct",
  "point_biserial_r": 0.31,
  "mean_verifier_score_correct": 0.79,
  "mean_verifier_score_incorrect": 0.61
}
```

A meaningfully positive `point_biserial_r` (and a visible gap between the
two means) means the signal is real; near zero (or reversed) means it
isn't — a legitimate negative finding, same as
`docs/legacy/stage2_verifier_corrections_paused.md`'s earlier verifier
diagnostic.

## Step 2 — three trajectories (`--verifier_type top3`)

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3
```

Same `--output_dir` as Step 1 is fine — `top3` writes its own
`ckpt_<model>_top3.csv` (Step 1's own `ckpt_<model>.csv` stays untouched),
so both modes' checkpoints coexist in one directory without colliding.
Thinking/`--gen_max_tokens`/`--chat_template_path` default the same way
Step 1's do (on, `4096`, vendored Gemma-4 template) — raising
`--gen_max_tokens` matters even more here, since a truncated thinking
trace now costs you 3 generations instead of 1.

Mechanism: one generation call draws `--top3_n_candidates` (default 3)
independent trajectories per image (needs `--temperature` > 0, the
default, or all 3 come back identical). Each candidate is graded
independently against the same rubric Step 1 uses — three separate
`verifier_score`s, not a tournament or pairwise comparison. The candidate
with the highest `verifier_score` (argmax) is reported as the row's final
answer; the other two candidates' extracted answers, rank, and score are
appended to the response as plain text for inspection, **never inside a
second `<answer>` tag** — the judge and `resolved_class` only ever see the
winner.

**Read the result**: same `verifier_score_correlation` as Step 1 (now
measuring the winning candidate's score), plus a new block that answers
the two questions a single trajectory structurally can't:

```json
"top3_pool_analysis": {
  "n": 480,
  "n_answer_in_pool": 312,
  "pool_coverage": 0.65,
  "verifier_pick_accuracy_given_in_pool": 0.74
}
```

- `pool_coverage` — out of all 3 candidates, how often does the ground
  truth show up in *at least one* of them (regardless of which one wins)?
  This is the ceiling: if the model never generates the right answer in
  any of the 3 tries, no amount of grading can save it.
- `verifier_pick_accuracy_given_in_pool` — when the right answer *is*
  somewhere in the pool, how often does the verifier's own per-candidate
  grading actually pick it as the winner? The gap between this and 1.0 is
  the verifier's own selection error, separate from the model's
  generation error `pool_coverage` measures.

Compare Step 1's `correct`/`judged_correct` rate against Step 2's to see
whether 3 trajectories + grading beats 1 trajectory + grading outright;
compare `pool_coverage` against that same rate to see how much of any gap
is "the model didn't even try the right answer" vs. "the verifier picked
the wrong one out of a pool that had it."

## Rubric criteria (what's actually being graded)

`src/config/verifier_prompts.yaml`'s `rubrics` — task-type-keyed, shown to
the verifier only. `disease_pest`'s criterion 7 targets a symptom-vs-cause
mix-up observed in real output (e.g. predicting "rice stem borer" — a
specific causal pest — when the ground truth was "dead heart," the
symptom it produces). Since ground-truth labels aren't consistent about
which convention they use (sometimes the symptom, sometimes the specific
disease/pest), the fix isn't forcing one or the other: the matching
`structured_templates.disease_pest` generation prompt now lets the model
state a symptom AND its probable cause together (e.g. "Dead heart, likely
caused by stem borer") instead of only ever printing a bare causal guess,
and criterion 7 checks for exactly that grounding — a trajectory that
states only the symptom with no causal guess at all still fully satisfies
it; only an ungrounded causal guess with no symptom mentioned doesn't.
Both `simple` and `top3` read this same rubric text, so this applies to
both automatically. Edit the YAML directly to add/change criteria for
other task types — no code change needed.

`prompts.yaml`'s shared `assessment_system` (every classification/detection
stage, not just this one) reinforces this at the system-prompt level: every
model is told to ground its conclusion in specific visual evidence, similar
to how the ReAct baseline's `finish()` requires a `justification` argument.
Gives the judge (`judge_classification_hybrid_template`, the pipeline-wide
default) more to work with — it already refuses to credit a plausible
causal guess against a symptom-named reference class, but can only apply
that logic to whatever detail is actually in the response.

## Splitting the work across multiple GPUs

`--num_shards N --shard_index i` partitions the (dataset, class) groups into N independent single-GPU processes, each generating **and** judging its own shard. Run one process per GPU with its own `--shard_index` — pass the SAME `--verifier_type` to every shard and to the merge call below, or the merge will look for the wrong filenames and find nothing:

```bash
# 4 GPUs, one process each ($SHARD_INDEX = 0, 1, 2, 3):
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --num_shards 4 --shard_index $SHARD_INDEX
```

Each shard writes its own `ckpt_<model>_top3_shard<i>of4.csv`, `verifier_log_top3_shard<i>of4.jsonl`, and `judge_log_top3_shard<i>of4.jsonl` (`simple` drops the `_top3` part — `ckpt_<model>_shard<i>of4.csv` — everything else is identical) — never the canonical filenames, so all 4 can run concurrently with zero write contention, and `simple`/`top3`/`ppt` can all share one `--output_dir` without colliding with each other either. Once every shard finishes, merge them into the canonical checkpoint:

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --output_dir outputs_final/verifier_cot/ \
    --merge_shards
```

## Legacy: `--verifier_type ppt`

Paper 1 Stage 4's original K-candidate + PPT-tournament verifier
(`docs/stage4_verifier.md`) still exists and still runs (`--k_candidates`/
`--k_pivots`), but is no longer actively tested or extended for Paper 2 as
of 2026-10-01 — `simple`/`top3` above are the current line of work. Kept
only for Paper 1 reproducibility.

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--model` | *(required)* | Model to run |
| `--verifier_type` | `simple` | `simple` (Step 1), `top3` (Step 2), or `ppt` (legacy, Paper 1 Stage 4) |
| `--top3_n_candidates` | `3` | `--verifier_type top3` only — independent trajectories per image |
| `--enable_thinking` / `--no-enable_thinking` | **on** | Extended thinking for the generation step (not the grading/pairwise calls) |
| `--chat_template_path` | vendored Gemma-4 template | Custom chat template — pass `""` to unset for a non-Gemma model |
| `--gen_max_tokens` | `4096` | Generation budget for the k_candidates/top3 step |
| `--from_ckpt` | none | `--verifier_type simple` only — grade an existing checkpoint's trajectories instead of regenerating |
| `--output_dir` | `outputs_final/verifier_cot/` | Checkpoint output dir — shared safely across all 3 `--verifier_type`s (distinct filenames) |
| `--manifest` | `dataset/selected_images_10.json` | Which manifest to sample from |
| `--match_images_from` | *(auto-detected)* | Reference checkpoint for guaranteed image overlap |
| `--n_samples` | `10` | Images per class |
| `--skip_judge` | off | Skip the judge phase |
| `--num_shards` / `--shard_index` | `1` / `0` | Data-parallel sharding across GPUs |
