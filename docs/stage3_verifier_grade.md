# Paper 2, Stage 3 — Simple Verifier (Rubric Grading)

[← Back to README](../README.md)

A cheaper alternative to Paper 1 Stage 4's K-candidate + PPT-tournament
verifier (`docs/stage4_verifier.md`): grade **one** trajectory per image
against each rubric criterion independently on a 1-3 scale, sum and
normalize to a single `verifier_score` in `[0, 1]`. No K-candidate pool,
no pairwise duel, no tournament — this is `verifier-cot`'s **default**
mode.

The question this stage answers: does this coarser, cheaper score still
correlate with correctness? `verifier-cot` prints that correlation
(point-biserial `r`) at the end of every run and writes it into
`metrics.json`. If it's meaningfully positive, `verifier_score` is a
candidate signal for gating [Stage 4](stage4_react_baseline.md)'s ReAct
agent later; if not, that's a real finding too.

## Just run this

```bash
uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it
```

That's it — no other flags needed. Defaults do everything:
- `--verifier_type simple` (the default) — one trajectory, per-criterion
  grading, no tournament.
- Images auto-match an existing `vision-baseline` run for the same
  `--model`, if one exists, so results are directly comparable.
- Judging runs automatically unless you pass `--skip_judge`.

Swap in whatever model you're testing for `--model`. Everything else
(`--output_dir`, `--subsample`, `--num_shards`, etc.) is there if you need
it later — see `--help` — but you don't need to touch it to get a result.

## Faster: grade an existing baseline run instead of regenerating

If you already have a `vision-baseline --rubric` checkpoint for this model,
skip generation entirely and just grade what it already produced:

```bash
uv run python main.py verifier-cot \
    --model google/gemma-4-26B-A4B-it \
    --from_ckpt outputs_final/vision_baseline_rubric_thinking/google_gemma-4-26B-A4B-it/ckpt_google_gemma-4-26B-A4B-it.csv
```

One flag. It grades that checkpoint's own trajectories directly (one
grading call per image instead of a generation call + a grading call),
and automatically samples the same images that checkpoint used — nothing
else to set.

## Read the result

Look at the end of the run's own printed output, or open
`outputs_final/verifier_cot/<safe_model_name>/metrics.json` and check
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
two means) means the signal is real and worth trying as a gate; near zero
(or reversed) means it isn't — a legitimate negative finding, same as
`docs/legacy/stage2_verifier_corrections_paused.md`'s earlier verifier
diagnostic.

## If it correlates: what's next

This stage only checks the correlation — it doesn't wire `verifier_score`
into anything yet. If the result looks useful, the next step (a separate
piece of work, not part of this stage) is threading `verifier_score` into
`docs/stage4_react_baseline.md`'s ReAct loop as a confidence signal the
agent can check before committing to `finish()`.

## For reference: everything above is `--verifier_type simple`

Pass `--verifier_type ppt` instead to run Paper 1 Stage 4's original
K-candidate + PPT-tournament verifier (`docs/stage4_verifier.md`) — a
different, heavier mechanism, not part of this stage.

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
Since both `--verifier_type simple` and `--verifier_type ppt` read this
same rubric text, this applies to both automatically. Edit the YAML
directly to add/change criteria for other task types — no code change
needed.

`prompts.yaml`'s shared `assessment_system` (every classification/detection
stage, not just this one) reinforces this at the system-prompt level: every
model is told to ground its conclusion in specific visual evidence, similar
to how the ReAct baseline's `finish()` requires a `justification` argument.
Gives the judge (`judge_classification_hybrid_template`, the pipeline-wide
default) more to work with — it already refuses to credit a plausible
causal guess against a symptom-named reference class, but can only apply
that logic to whatever detail is actually in the response.

## Splitting the work across multiple GPUs

`--num_shards N --shard_index i` partitions the (dataset, class) groups into N independent single-GPU processes, each generating **and** judging its own shard. Run one process per GPU with its own `--shard_index`:

```bash
# 4 GPUs, one process each ($SHARD_INDEX = 0, 1, 2, 3):
uv run python main.py verifier-cot \
    --model google/gemma-4-26B-A4B-it \
    --num_shards 4 --shard_index $SHARD_INDEX
```

Each shard writes its own `ckpt_<model>_shard<i>of4.csv`, `verifier_log_shard<i>of4.jsonl`, and `judge_log_shard<i>of4.jsonl` (never the canonical filenames), so all 4 can run concurrently with zero write contention. Once every shard finishes, merge them into the canonical checkpoint:

```bash
uv run python main.py verifier-cot \
    --model google/gemma-4-26B-A4B-it \
    --output_dir outputs_final/verifier_cot/ \
    --merge_shards
```

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--model` | *(required)* | Model to run |
| `--verifier_type` | `simple` | `simple` (this stage) or `ppt` (Paper 1 Stage 4) |
| `--from_ckpt` | none | Grade an existing checkpoint's trajectories instead of regenerating |
| `--output_dir` | `outputs_final/verifier_cot/` | Checkpoint output dir |
| `--manifest` | `dataset/selected_images.json` | Which manifest to sample from |
| `--match_images_from` | *(auto-detected)* | Reference checkpoint for guaranteed image overlap |
| `--n_samples` | `10` | Images per class |
| `--skip_judge` | off | Skip the judge phase |
| `--num_shards` / `--shard_index` | `1` / `0` | Data-parallel sharding across GPUs |
