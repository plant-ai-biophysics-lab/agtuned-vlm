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
open `outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/metrics.json` and check
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

**As a chart** instead of reading the raw numbers:

```bash
uv run python main.py verifier-score-correlation-plot \
    --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

No GPU needed — reads the checkpoint's own `verifier_score`/
`judged_correct` columns directly, recomputes point-biserial r itself
(not trusted from a possibly-stale `metrics.json`), and writes ONE PNG
(`<ckpt's own dir>/correlation/verifier_score_correlation.png`): a
box+jittered-strip plot and an overlaid histogram of `verifier_score`,
split by correct/incorrect, annotated with r and both group means. Works
against both `--verifier_type simple` (`ckpt_<model>.csv`) and `top3`
(`ckpt_<model>_top3.csv`) checkpoints.

**QC the rubric grading itself** — a random image-level gallery plus a
per-criterion score distribution, no GPU, no network (images resolved by
globbing whatever's already cached under `--images_root`, never
re-downloaded):

```bash
uv run python main.py verifier-cot-qc-plot \
    --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Writes to `<ckpt's own dir>/qc_plots/`:
- `rubric_gallery.png` (paginated via `--max_rows_per_figure`,
  `_2`/`_3`/... if needed) — `--n_datasets` (default 10) random datasets
  x `--n_per_dataset` (default 3) random images each, the actual image
  next to its `ground_truth`/`verifier_score`/`criterion_scores` (raw
  JSON plus each criterion's own rubric question text, so you don't have
  to cross-reference `verifier_prompts.yaml` by hand).
- `criterion_score_distribution.png`/`.json` — a stacked bar (counts of
  score 1/2/3) per criterion, faceted by `task_type` — different
  task_types use different rubrics (`verifier_prompts.yaml`'s
  `task_type_to_rubric`), so "criterion 1" means a different question
  depending on task_type and pooling across them would be misleading.
  If every bar is nearly all score=3, that's the same ceiling effect a
  near-zero `point_biserial_r` from the correlation plot above would
  predict — little variance left for the score to discriminate on.

## Step 2 — three trajectories (`--verifier_type top3`)

**Building forward from `verifier_rubric` only** (as of 2026-10-03) —
`explicit`/`resample` share Step 1 (`simple`)'s own grading mechanism: a
trajectory graded against a rubric using that SAME trajectory's own
self-stated observations, which is the mechanism already shown (Step 1's
`verifier-cot-qc-plot`) to produce near-ceiling score=3 results. Running
either to full scale would just apply that same problem 3x per image
instead of once, so **neither gets a full run** here. The smoke test
below is still worth running, though — it answers a different, narrower
question (does the model actually produce 3 genuinely different
candidates at all) that stays relevant regardless of which mode grades
them.

Three ways to generate/grade the 3-candidate pool (`--top3_mode`):

- **`explicit`** (the default): **ONE** generation call per image. The
  model states its observations once, then is explicitly asked to give 3
  genuinely different ranked possibilities itself
  (`<answer_1>`/`<answer_2>`/`<answer_3>` tags) — not three separate
  calls. Each candidate is then graded against the rubric using its OWN
  stated observations (3 independent grading calls, same rubric Step 1
  uses — no tournament, no pairwise comparison).
- **`resample`**: draws 3 independent trajectories via temperature
  sampling (`SamplingParams(n=3)`) instead — no explicit-diversity
  instruction, just independent re-draws of the same single-answer
  prompt. Graded the same way as `explicit`.
- **`verifier_rubric`** (2026-10-03): a different division of labor,
  built to test a specific hypothesis for why most criterion scores sit
  at a near-ceiling 3 (see `verifier-cot-qc-plot`'s
  `criterion_score_distribution.png` above) — `explicit`/`resample`'s
  generation step restates exactly the fields the rubric then checks for
  (e.g. "Crop or plant species affected:" pre-fills what criterion 1
  asks about), so a trajectory can trivially "cite" something for every
  criterion whether or not that citation is actually correct. Here:
  1. ONE generation call — candidates are just 3 bare answers, **no**
     observational preamble at all.
  2. ONE verifier evidence-gathering call — the verifier looks at the
     image independently (no candidate text shown) and fills in the
     rubric itself, criterion by criterion. This is the one shared
     evidence base every candidate is scored against.
  3. 3 separate scoring calls — each candidate's bare answer is graded
     against **that evidence** (not its own text, which doesn't exist
     here), on the same 1-3 per-criterion scale.

  `--enable_thinking` applies to **all three** calls in this mode
  (generation, evidence-fill, scoring) — unlike `explicit`/`resample`,
  whose grading call is always thinking-off. Raise `--grading_max_tokens`
  well above its 300 default when combining this with `--enable_thinking`
  (a thinking trace routinely truncates before the scoring tags at 300 —
  try 2048+).

**Smoke test first** — confirm the model actually produces 3 distinct
possibilities (not the same one restated 3 times). Run with the default
`explicit` mode (cheapest, no `verifier_rubric` flags needed) purely to
check candidate diversity — this smoke test is a generation-side check,
independent of which `--top3_mode` ends up grading the pool, so there's
no need to re-run it under `verifier_rubric` separately:

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --subsample arabica_coffee_leaf_disease_classification \
    --n_samples 2 \
    --output_dir outputs_final/verifier_cot_top3_smoketest/
```

Check the checkpoint's `candidate_predictions` column — it should show 3
genuinely different strings per row, not the same label 3 times. If the
model is just restating the same guess, `--top3_mode explicit`'s own
diversity instruction isn't working on this model, and `--top3_mode
resample` (or a smaller/larger `--temperature`) is worth trying instead.

**Testing `verifier_rubric` mode** — 3 images each from 3 random
datasets (`--random_datasets`, deterministic via `--seed`), thinking on
for all three calls, `--grading_max_tokens` raised:

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --top3_mode verifier_rubric \
    --grading_max_tokens 2048 \
    --random_datasets 3 \
    --n_samples 3 \
    --output_dir outputs_final/verifier_cot_top3_verifier_rubric_smoketest/
```

Thinking/`--chat_template_path`/`--gen_max_tokens` default on/vendored
Gemma-4 template/`4096` the same way the rest of this stage does — no
extra flags needed for those. `--n_samples` is images PER CLASS, so a
dataset with multiple classes will contribute more than 3 images total —
fine for a smoketest, just not an exact "9 images" guarantee.

Then QC the result:

```bash
uv run python main.py verifier-cot-qc-plot-top3 \
    --ckpt outputs_final/verifier_cot_top3_verifier_rubric_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv
```

— the candidate gallery shows each image's 3 bare candidates plus the
verifier's own evidence-derived criterion scores, and
`per_candidate_score_distribution.png`/`winner_score_correlation.png`
tell you whether `point_biserial_r` and the score spread actually look
different from `explicit`/`resample`.

**Full run — `verifier_rubric` mode** (the only mode that gets a full
run here — `--top3_mode` must be passed explicitly, every time; there is
no default that produces this mode, and bare `--verifier_type top3`
alone silently falls back to `explicit`):

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --top3_mode verifier_rubric \
    --grading_max_tokens 2048
```

Same `--output_dir` as Step 1 is fine — `top3` writes its own
`ckpt_<model>_top3.csv` (Step 1's own `ckpt_<model>.csv` stays
untouched). `--grading_max_tokens 2048` covers BOTH of this mode's own
verifier calls (the evidence-fill call and the per-candidate scoring
call) — both have thinking on by default in this mode, so the bare 300
default will truncate them. Thinking/`--gen_max_tokens`/
`--chat_template_path` for the candidate-generation call default the
same way Step 1's do (on, `4096`, vendored Gemma-4 template).

**Known risk, for context** (why `explicit`/`resample` were set aside
rather than ever getting their own full run): a near-identical approach
to `explicit` ("ask for several different hypotheses in one response")
was tried and reverted for the legacy `ppt` mode's K-candidate design —
forcing deliberate diversity competed against the model's own
best-calibrated guess rather than sharpening it (only 15.6% of wrong
predictions had the correct answer among 6 forced-diverse hypotheses
there). Combined with the same self-citation ceiling effect `simple`
already showed, there wasn't a strong enough case to validate `explicit`
at scale before moving straight to `verifier_rubric`.

The candidate with the highest `verifier_score` (argmax) is reported as
the row's final answer; the other two candidates' extracted answers,
rank, and score are appended to the response as plain text for
inspection, **never inside a second `<answer>` tag** — the judge and
`resolved_class` only ever see the winner.

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

**As charts** — QC the `verifier_rubric` full run above (same command
works against the `explicit`/`resample` smoke test too, if you ever want
to check those — just point `--ckpt` at whichever checkpoint). This is
the full run's own default `--output_dir` (`outputs_final/verifier_cot/`,
not overridden above), so no path changes needed if you ran the full-run
command exactly as shown:

```bash
uv run python main.py verifier-cot-qc-plot-top3 \
    --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv
```

Writes to `<ckpt's own dir>/qc_plots_top3/`:
- `candidate_gallery.png` (paginated) — each sampled image next to ALL 3
  candidate predictions (verifier_score each, the winner marked
  `[CHOSEN]`), the winner's own `criterion_scores` (bare numbers, in
  order), and the judge's verdict.
- `winner_score_correlation.png`/`.json` — the same chart
  `verifier-score-correlation-plot` produces, against this checkpoint's
  winning-candidate score vs. `judged_correct`.
- `per_candidate_score_distribution.png`/`.json` — **all 3 candidates
  per row**, not just winners, split by a cheap ground-truth text-match
  proxy (clearly labeled as NOT the real judge — only the winner is ever
  actually judged). A broader signal check: does `verifier_score`
  discriminate a right answer from a wrong one across the whole pool, or
  only among winners?
- `pool_summary.png`/`.json` — `pool_coverage` /
  `verifier_pick_accuracy_given_in_pool` as a bar chart, same numbers as
  `top3_pool_analysis` above.

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

`--num_shards N --shard_index i` partitions the (dataset, class) groups into N independent single-GPU processes, each generating **and** judging its own shard. Run one process per GPU with its own `--shard_index` — pass the SAME `--verifier_type`/`--top3_mode` to every shard and to the merge call below, or the merge will look for the wrong filenames and find nothing (checkpoint filenames depend on `--verifier_type` only, not `--top3_mode` — see the `verifier_rubric` full-run note above — so this doesn't change what to pass here, just know `--top3_mode` itself isn't part of the filename):

```bash
# 4 GPUs, one process each ($SHARD_INDEX = 0, 1, 2, 3):
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --verifier_type top3 \
    --top3_mode verifier_rubric \
    --grading_max_tokens 2048 \
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

`--merge_shards` only needs `--verifier_type` (it just looks for
`ckpt_<model>_top3_shard*of*.csv` files, regardless of which
`--top3_mode` produced them) — `--top3_mode` itself is not required on
this call.

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
| `--top3_mode` | `explicit` | `--verifier_type top3` only, must be passed explicitly — `explicit` (default, one call, model states N possibilities itself, set aside for full runs — see Step 2 above), `resample` (N independent temperature-sampled draws, also set aside), or `verifier_rubric` (**the mode this stage builds from**: N bare candidates, verifier independently fills the rubric once, scores each candidate against that shared evidence — raise `--grading_max_tokens` to ~2048+ with this mode) |
| `--top3_n_candidates` | `3` | `--verifier_type top3` only — pool size (candidates read back, or drawn, depending on `--top3_mode`) |
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
