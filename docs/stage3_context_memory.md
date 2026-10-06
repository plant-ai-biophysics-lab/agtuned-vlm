# Paper 2, Stage 3 — Context/Skill Memory Construction

[← Back to README](../README.md)

Status: **Steps 0-4 all built AND run for real** (ablation found a real
+0.0603 judge_accuracy gain, n=199, after 4 iterations — see Step 4's
own real-result table below). **Step 5 built** (`vision-baseline`
injection, gated to `task_type in {disease, pest_damage}`) but not yet
run for real. `verifier-cot` injection and Step 6 (repeat for detection)
still TODO. See
[Stage 4](stage4_classification_verifier.md) for the verifier work this
builds on (the leniency/evidence-contamination findings there are what
motivated this stage).

## Why this stage exists

Stage 4's `verifier_rubric` refinement work surfaced a specific, concrete
failure mode in the classification model's own domain knowledge — not a
vague "the model doesn't know enough," but reproducible confusions found
directly in transcripts (e.g. Lentil Rust mistaken for phosphorus
deficiency; regional/vernacular medicinal-plant names misidentified as
more common species). The idea: harvest these confusions empirically from
real model traces, distill them into a single reference `.md` file
("what worked", "what didn't work", common diseases and what
distinguishes them), and inject that file as extra context into zero-shot
classification — a static, non-agentic analog of context-engineering
ideas from the agentic-harness literature, but kept entirely within a
single model's own context window, no harness.

## Dataset: held out, never used for eval

[Digital Green's `Crop_Disease_Images`](https://huggingface.co/datasets/DigiGreen/Crop_Disease_Images)
dataset — 989 real images (expert agronomist labels, a `details` column
with written symptom observations, CC-BY-4.0). Chosen over
`DigiGreen/Crop-Disease-Image-Eval-Synthetic` after checking both:
the "Eval-Synthetic" dataset ships **no images at all** (label/ID
metadata only, photos withheld pending a privacy review) and its own
card explicitly says its labels are **model-consensus, not ground
truth** ("Treat them as a benchmark... not as clinical ground truth") —
neither property works for this stage, which needs real images and
trustworthy labels to reveal as ground truth. `Crop_Disease_Images` has
real pixels and real expert labels.

This dataset is **not** under the `Project-AgML` HuggingFace org, so it
can't go through `catalog-datasets`/`download-datasets` as-is (those
hardcode `repo_id = f"Project-AgML/{agml_name}"`). Its `diagnosis` column
can also hold MULTIPLE semicolon-separated labels in a single row (e.g.
"Pod Borer; Boron Deficiency", confirmed via the HF datasets-server —
rare, ~1/20 rows in a spot check, but real), which doesn't fit this
repo's one-label-per-image `DatasetConfig`/`load_hf_samples` convention
either. So this is a deliberately standalone script and manifest format,
not a `datasets.yaml` entry — kept in its own files, for safety, so it
can never be sampled into a normal eval run:
- `scripts/download_heldout_digitalgreen.py` (the downloader)
- `dataset/heldout_selected_images.json` (not `dataset/selected_images.json`)
- `dataset/images_heldout/` (not `dataset/images/`, gitignored same as it)

## Step 0 — download the held-out dataset

```bash
uv run python main.py download-heldout-digitalgreen
```

Caches all 989 images to `dataset/images_heldout/`, writes
`dataset/heldout_selected_images.json` — one record per row (`image_path`,
`crop`, `diagnosis` as a list, `details`, `country`, `state`, `split`).
Resumable (an already-cached image/row is skipped on re-run).

**The lesson-building vs. eval split is already decided and built**: a
**lesson-building slice** (`"split": "lessons"`, images the skill file is
written from) and a **held-out eval slice** (`"split": "eval"`, images
never shown during lesson-writing, used only to measure whether the
skill file actually generalizes) — `--eval_frac` (default `0.2`),
stratified by `crop` so both splits cover the same crops rather than an
eval slice that happens to miss whole crops entirely, seeded by `--seed`
(default `42`) for reproducibility. The eval slice must never be used to
write the skill-memory `.md` file — only to measure whether it
generalizes, once built.

## Step 1 — zero-shot prediction (built)

```bash
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

Plain single-shot classification on the lesson-building slice only
(`--split lessons`, the default — `--split eval` exists for Step 4 later,
never use it before then). No rubric, no verifier — same shape as
`vision-baseline`'s own lower bound, reusing its exact
`assessment_classification_tagged`/`assessment_system` prompts.

Judged against the real expert `diagnosis` label via the same
pipeline-wide hybrid judge template every other stage uses. That
template has its own `{class_list}` placeholder (it checks whether a
prediction is SPECIFIC enough to distinguish the reference diagnosis
from every other one, not just broadly consistent with it) — this
dataset isn't an AgML dataset (no `datasets.yaml`, no per-row
`agml_dataset` to key a class list off), so `{class_list}` is filled
from this dataset's OWN unique `diagnosis` values instead, via a new
`extra_keys` override added to `run_judge_phase` (`src/common/vision_eval.py`)
specifically so a future non-AgML caller doesn't need its own
reimplementation of the judge phase.

Writes `outputs_final/heldout_zeroshot/<safe_model_name>/ckpt_<safe_model_name>.csv`
— `class_label` is the real `diagnosis` (joined), `image_name` is the
actual cached filename, `judged_correct`/`judge_reasoning` as usual.
`prediction` is the FULL cleaned response (narrative + answer line +
`<answer>` tag, same convention `vision-baseline`'s own `prediction`
column uses) — fixed from an earlier bug where this stored just the bare
extracted answer, silently dropping the model's own stated reasoning
(and therefore also judging the bare answer, not the full
reasoning-bearing text every other judge call in this repo sees).
`extracted_answer` is the bare tag content alone. (`agml_dataset`/
`class_label`/`image_name` column names are reused deliberately, not a
stylistic choice — `load_ckpt`'s resume logic hardcodes exactly those 3
names to build its seen-keys set.)

**Real result** (n=827, `RedHatAI/gemma-4-31B-it-FP8-dynamic`, **from a
run predating the `prediction`-field fix above — re-run with `--rerun`
before trusting these numbers or building Step 2 traces from them**):
**`judge_accuracy = 0.3881`**. General trends found in this run (used to
shape Step 2's own system prompt below, not hard-coded into it
per-example — expected to still roughly hold after the re-run, since the
fix changes what the judge SEES, not the underlying diagnoses, but worth
re-checking):
- **149/827 rows were a false "Healthy"** — the single largest error
  bucket, bigger than any specific category confusion. The model
  over-predicts "Healthy" as a safe default when actually unsure.
- **Pest/insect damage is the hardest category to recognize at all**
  (32.7% bucket accuracy vs. 60.4% disease, 71.0% healthy, heuristic
  keyword bucketing) — the model defaults to a disease explanation
  rather than seriously considering insect damage as a hypothesis.
- **Nutrient deficiency is also weak** (19.4% bucket accuracy, n=36 —
  suggestive, not solid) — frequently confused with disease or missed
  entirely, consistent with the Lentil-Rust/phosphorus-deficiency
  confusion already found in Stage 4's own transcripts.
- **Strong crop-dependent variance**, not a uniform knowledge gap: Wheat
  (0.81) and Mustard (0.78) are strong; Brinjal (0.13), Banana (0.24),
  Mango (0.25), Groundnut (0.27), Tobacco (0.28) are weak.

## Step 2 — reveal ground truth on EVERY row, miss or not (built)

```bash
uv run python main.py heldout-explain-traces \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_zeroshot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Reads Step 1's own checkpoint and processes **every judged row, not just
misses** — the skill file needs both "what worked" and "what didn't
work" sides, so a correct prediction gets a trace too, not just a
sampled subset of them. For each row, shows the model its OWN full
prediction (the fixed, narrative-bearing `prediction` column — the
model's own stated reasoning, not just its bare final answer), the REAL
ground truth label, AND the dataset's expert-written `details`
observations — then asks for a short (2-4 sentence), visually-grounded
explanation, via one of two templates picked by `judged_correct`:

- **`heldout_explain_miss_prompt`** ("what didn't work") — explain what
  evidence should have led to the correct answer, pointing out
  specifically where the model's own stated reasoning went wrong.
- **`heldout_explain_correct_prompt`** ("what worked") — confirm and
  articulate the evidence that correctly supports the answer.
  Explicitly instructed to call out a right-answer-for-the-wrong-reasons
  case (a lucky guess) rather than crediting unsound reasoning just
  because the final label happened to match.

Both grounded in the real expert `details` text rather than the model
inventing plausible-sounding reasoning from nothing. Shared system
prompt (`src/config/heldout_prompts.yaml`'s `heldout_lesson_system`)
bakes in the three general trends found in Step 1's real results above
as reminders for what's worth emphasizing — never restated as facts
about any specific image, and never hard-coded per-crop/per-diagnosis.

Writes `outputs_final/heldout_explain_traces/<safe_model_name>/ckpt_<safe_model_name>.csv`
— every Step 1 column passed through unchanged, plus `trace_type`
(`"worked"`/`"didnt_work"`) and `explanation`.

**Real result** (n=827): **323 `"worked"` / 504 `"didnt_work"`**, zero
blank explanations, ~101K words total across all explanations (confirms
Step 3 genuinely needs map-reduce — nowhere close to fitting in one
context window). **Found and fixed a real bug on the real run**: ~5%
of rows (40/827) had their thinking trace truncate before the closing
`<channel|>` marker at the original `--gen_max_tokens 2048`, so
`extract_channel_thought`'s regex couldn't match and fell back to
returning the RAW unstripped `<|channel>thought...` text straight into
`explanation` (confirmed: those rows' word count was 15-20x the median).
Fixed by bumping the default to `4096`; the 40 affected rows were
removed from the local checkpoint so a plain resumed re-run (no
`--rerun` needed) regenerates only those, not all 827.

## Step 3 — summarize into the skill `.md` (built)

```bash
uv run python main.py heldout-summarize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Text-only map-reduce (no images — these are already-written explanations
about images, not the images themselves):

- **Map**: group all explanations by `(bucket_category(), trace_type)` —
  disease/pest/deficiency/healthy x worked/didnt_work, 8 buckets. A
  bucket whose raw explanation text would exceed `--max_words_per_chunk`
  (default `8000`) is split further into word-bounded chunks (not
  fixed-row-count — explanation length varies a lot). ONE
  `heldout_map_summarize_prompt` call per chunk, producing 4-8 bullet
  points of RECURRING, GENERALIZABLE patterns — never restating
  individual explanations, never per-image facts.
- **Reduce**: ONE final `heldout_reduce_summarize_prompt` call merges
  every chunk-level bullet summary into a single Markdown document
  (`# Crop Diagnosis — Lessons Learned`, with `## What Worked` /
  `## What Didn't Work` sections — the prompt used to also ask for a
  `## Common Diseases, Pests, and Deficiencies to Know` reference
  section; see the real-result note below for why that's gone).

**Prompt fix (2026-10-05), applies to every future run**: both
`heldout_map_summarize_prompt` and `heldout_reduce_summarize_prompt`
now explicitly forbid bare-category-word bullets (e.g. a bullet whose
entire content is just "Chewing Pests:") and the reduce prompt no
longer asks for the "Common Diseases..." section at all. v4's real
result (below) was produced by manually deleting that section from one
generated file (`skill_memory_no_common_section.md`) — the GENERATOR
itself was never fixed until now, so a fresh `heldout-summarize` run
before this date would have reintroduced the problem. Going forward,
`skill_memory.md` should be correct by construction, without a manual
post-edit step.

Chunk-building grouped by heuristic category (per your own choice over
crop-grouping or fixed-size batches) — keeps related explanations
together so each chunk's summary is naturally about one kind of problem,
and maps directly onto the final `.md`'s own section structure.

Map-step summaries are cached to
`outputs_final/heldout_summarize/<safe_model_name>/chunk_summaries.json`
after the map phase — the (cheap, 1-call) reduce phase can be re-run
independently without re-paying for the (more expensive, ~11-call) map
phase; `--rerun` discards the cache and redoes everything.

**Real chunk sizes** (n=787, post-leak-fix-removal — will shift slightly
once the 40 regenerated rows are back in): `pest/didnt_work` (214 rows)
and `disease/didnt_work` (174 rows) each needed 2 chunks; the other 6
buckets fit in 1 chunk each — **11 map calls + 1 reduce call = 12 total
LLM calls**, not 827.

Writes the final file to `--output_md` (default
`outputs_final/heldout_summarize/skill_memory.md`) — this is the
artifact Step 4 ablates and, if it helps, gets injected into
`vision-baseline`/`verifier-cot`.

**Real result** (pre-prompt-fix run): the real `skill_memory.md` had
four sections — Disease, Pest, Nutrient Deficiency, Healthy — each
under both "What Worked" and "What Didn't Work", plus a "Common
Diseases, Pests, and Deficiencies to Know" reference list that was
later manually deleted (see the prompt-fix note above — a fresh run now
should never produce that section in the first place). **Zero content
for quality
grading or species/weed identification** — DigiGreen's dataset doesn't
cover those domains at all, so there's nothing in the traces for the
model to summarize there. This directly scopes Step 5 below: injection
must be gated to `task_type in {disease, pest_damage}` — the only two
`task_type_to_rubric` buckets this skill file actually has real content
for — never applied blindly to `quality_identification`/
`species_identification`/`weed_identification` runs, where it would be
irrelevant context at best.

## Step 3b — self-optimization loop (built)

`heldout-skill-optimize` (`src/heldout_skill_optimize.py`) — EDITS an
existing skill-memory.md directly against this round's own eval-split
per-task scores, instead of re-summarizing from Step 2's traces. The
v4 skill file (the one that earned +0.0603, hand-patched to remove the
"Common Diseases..." section) is checked into the repo at
`src/config/skill_memory_disease-pest.md` — the fixed starting point ("round 0")
for this loop, so it survives independent of whatever's sitting in
`outputs_final/` on any given machine.

One round:

```bash
# 1. Score the CURRENT skill file on eval:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/heldout_zeroshot_eval_optimize_round0/

# 2. Edit the skill file against those scores + a sample of its own misses:
uv run python main.py heldout-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --eval_ckpt outputs_final/heldout_zeroshot_eval_optimize_round0/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_md outputs_final/heldout_skill_optimize/skill_memory_round1.md \
    --round_log outputs_final/heldout_skill_optimize/round_log.json

# 3. Score round 1's output, feed it into round 2's edit call, repeat —
#    --skill_memory_path/--eval_ckpt point at round N's own files each time.
```

Each round's `--output_md` is a NEW path (`skill_memory_round<N>.md`) —
never overwrites the input, so every round's version stays on disk.
Each round **replaces** the document rather than merging onto it (see
the earlier discussion on why: the edit call always has the full
current document in front of it, so nothing is lost by writing a fresh
revision — the alternative, accreting edits onto edits, risks unbounded
growth and a telephone-game drift in wording across rounds).

**Explicit, stated caveat**: every round's edit is steered DIRECTLY by
this fixed eval split's (n=199) own scores — this is intentionally
optimizing against the same split repeatedly, a real overfitting risk,
chosen on purpose despite it (see the loop-design discussion above —
the clean version of this would use a 3-way lessons/dev/final-eval
split; this version does not, by explicit choice, so treat any single
round's number as provisional). `--round_log` tracks every round's
per-task scores in one JSON file specifically so a noisy, non-monotonic
trajectory is visible rather than only ever looking at the latest
round's number in isolation.

## Step 4 — ablation (built)

Two `heldout-zeroshot` runs against `--split eval` (the slice no other
step has ever touched), then one comparison:

```bash
# Without the skill file — the real baseline:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --output_dir outputs_final/heldout_zeroshot_eval_baseline/

# With the skill file injected into the system prompt:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/heldout_zeroshot_eval_not_optimized_skill/

# Compare:
uv run python main.py heldout-ablation-summary \
    --baseline_ckpt outputs_final/heldout_zeroshot_eval_baseline/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --skill_ckpt outputs_final/heldout_zeroshot_eval_not_optimized_skill/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

`--skill_memory_path` (new flag on `heldout-zeroshot`) injects the file
into the SYSTEM prompt only — nothing else about the generation call
changes between the two runs, so any accuracy delta is attributable to
the skill file itself. `--output_dir` must differ between the two runs
(same `--model` otherwise collides on the same checkpoint path).

`heldout-ablation-summary` is pure post-hoc comparison (no GPU) —
`judge_accuracy` bar chart + JSON, with an explicit printed warning if
the two checkpoints don't cover the identical image set (they should,
since both come from the same seeded `--split eval`), and a blunt
`[finding]` line if the delta is `<= 0`: **do not inject into
vision-baseline/verifier-cot based on a non-improving or negative
result** — the whole point of this step is to not trust the skill file
just because it looks reasonable on paper.

**Real result** (n=199 held-out eval split, `judge_accuracy` split by
`bucket_category(class_label)` — disease n=69, pest n=62, deficiency
n=10, healthy n=58; full version-by-version iteration history that
arrived at this skill file — v1 through v4 — now lives in
`src/common/heldout_skill.py`'s own module docstring, the single source
of truth so Step 5's real injection can never drift from what was
actually validated):

| Version | overall | disease | pest | deficiency | healthy |
|---|---|---|---|---|---|
| baseline (no skill) | 0.3618 | 0.2319 | 0.1452 | 0.1000 | 0.7931 |
| not optimized skill (`skill_memory_disease-pest.md`) | **0.4221** (+0.0603) | 0.2754 (+0.0435) | 0.2419 (+0.0967) | 0.3000 (+0.2000) | 0.8103 (+0.0172) |

Pest carries the single biggest per-task gain (+0.0967); deficiency's
swing (+0.2000) is the least trustworthy given n=10 (each flipped row
moves that bucket's accuracy by 10 points). This matches the later
finding that the full-scale AgML run's gain concentrated in
`pest_damage` and was flat on `disease` — already visible here, just
not called out at the time. Step 3b's self-optimization loop (above) is
the ongoing attempt to improve on this row specifically.

v1's near-zero delta was a real, mechanistic regression, not just noise:
the model borrowed the skill file's own CATEGORY-level phrasing
("Chewing Insects:", "Sucking Pests:") instead of the specific diagnosis
name (e.g. "Fall armyworm" → "Chewing pests"), which the hybrid judge
template explicitly penalizes as not matching a specific reference
class — canceling out genuine fixes elsewhere (e.g. a false "Healthy"
correctly caught as "Chilli Leaf Curl Virus", exactly the problem this
stage started from). v2 fixed most of it; v3 drove bare-category answers
to zero and eliminated a second regression v2 introduced (over-diagnosing
genuinely healthy plants); v4's remaining 2 regressions are
specific-vs-specific confusions between genuinely similar conditions
(e.g. "Tobacco Mosaic Virus" → "Tobacco Leaf Curl Virus") — a sign of
real diagnostic difficulty, not a fixable instruction-following failure.

**Methodological caveat, stated plainly**: all 4 versions were checked
against the SAME eval split, which means this history amounts to
informally validating against held-out data — the reported delta is
likely somewhat optimistic relative to a true single-shot held-out test.
**v4 is the locked-in final config** (no further iteration against this
eval split) — a genuinely clean confirmatory number would need a fresh
eval sample (different `--seed` on `download-heldout-digitalgreen`),
not something this stage blocks on before Step 5.

## Step 5 — inject into vision-baseline (built)

`vision-baseline` only, for now (`verifier-cot`'s generation step and/or
`verifier_rubric_fill_prompt` still TODO — isolating `vision-baseline`'s
own effect cleanly first, same reasoning as the original plan):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path outputs_final/heldout_summarize/skill_memory_no_common_section.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/
```

(`--output_dir` deliberately points somewhere NEW, not the existing
`outputs_final/vision_baseline_thinking/` — that's the real baseline
checkpoint every other comparison in this repo references; overwriting
it would destroy the thing this is supposed to be compared against.
Everything else defaults the same way a normal `vision-baseline` run
does — see [Stage 1](stage1_baseline_classification.md) — thinking on,
`gen_max_tokens 4096`, vendored Gemma-4 chat template.)

`--skill_memory_path` injects the file into the system prompt, built via
the exact same `build_skill_system_prompt` wrapper the ablation
validated (now factored into `src/common/heldout_skill.py`, imported by
both `heldout_zeroshot.py` and `vision_baseline.py` — never re-derived
independently, so production can't silently drift from what was actually
tested). Gated **per-row** to `task_type in DISEASE_TASK_TYPES`
(`{"disease", "pest_damage"}`, `src/common/dataset_config.py`) — every
other `task_type` (quality/species/weed identification) uses the plain
system prompt unchanged, since the skill file has zero real content for
those domains. Verified the gating logic directly: `disease`/
`pest_damage` rows get the skill-augmented prompt, every other task_type
and the no-flag case both correctly fall through to plain.

Not yet run for real (no GPU locally) — the real next step is running
this against a full AgML disease/pest-damage sweep and checking whether
the held-out-eval gain actually transfers to real datasets.

**Splitting the work across multiple GPUs** — same `--num_shards N
--shard_index i` mechanism as every other `vision-baseline` run (see
[Stage 1](stage1_baseline_classification.md)'s own sharding section) —
`--skill_memory_path` is just another flag, independent of sharding, so
nothing extra to configure:

```bash
# 4 GPUs, one process each ($SLURM_ARRAY_TASK_ID = 0, 1, 2, 3):
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path outputs_final/heldout_summarize/skill_memory_no_common_section.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID
```

Each shard writes its own `ckpt_<model>_shard<i>of4.csv` (never the
canonical filename) plus its own `judge_log_shard<i>of4.jsonl`, so all 4
can run concurrently with zero write contention. Once every shard
finishes, merge them into the canonical checkpoint (single job, after
all array tasks complete):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --merge_shards
```

`--merge_shards` doesn't need `--skill_memory_path` — it only merges
whatever shard files are already on disk, it doesn't regenerate
anything. Same caveat as Stage 1: don't leave shard files sitting
alongside the merged canonical file when pointing other tooling at this
directory afterward.

## Planned pipeline, Step 6 (not built yet)

6. **If the classification gains hold up, repeat for object detection**
   ([Stage 5](stage5_detection_verifier.md)) — same mechanism (zero-shot
   predict → reveal ground truth on misses → explain → summarize into a
   skill file → ablate on a held-out detection slice before trusting it),
   applied to `detection-multiclass`/`detection-box-verifier` instead of
   `vision-baseline`/`verifier-cot`. Deliberately sequenced AFTER
   classification proves out, not in parallel — no reason to pay for a
   second, detection-specific held-out dataset/pipeline build if the core
   idea doesn't even work for the simpler classification case first. **Run
   this whole stage (through Step 5's injection, if it helps) to
   completion before starting real Stage 5 detection work** — not in
   parallel with it.

## Open questions / not yet decided

- Final `.md` file's real length/quality once a real run completes — if
  it's still too large for a comfortable context injection even after
  map-reduce, may need a second summarization pass over the file itself.
- Script/command names for Step 4 — not yet built.
