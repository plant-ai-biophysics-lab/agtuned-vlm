# Paper 2, Stage 3 — Context/Skill Memory Construction

[← Back to README](../README.md)

Status: **Steps 0-2 (download, zero-shot prediction, reveal-and-explain
on misses) built; Steps 3-4 (summarize, ablate) not started yet.** See
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

## Planned pipeline, Steps 3-4 (not built yet)

3. **Summarize traces into one `.md` file** — aggregate ALL of Step 2's
   `"worked"`/`"didnt_work"` explanations (likely too much raw text for
   one summarization call — ~800 explanations — so probably a map-reduce:
   summarize in chunks, then merge chunk-summaries into the final file)
   into a single skill-memory file: common diseases, what to look for,
   confusions to watch for.
4. **Ablate before trusting it** — zero-shot `judge_accuracy` on the
   **held-out eval slice**, with vs. without the skill file injected as
   extra context. Only a result that holds on held-out images counts as
   a real finding, not memorization of the lesson-building slice.
5. **If it helps**: inject into `vision-baseline` first (isolates the
   skill file's own effect cleanly), then `verifier-cot`'s generation
   step and/or `verifier_rubric_fill_prompt` if it still helps there.
6. **If the classification gains hold up, repeat for object detection**
   ([Stage 5](stage5_detection_verifier.md)) — same mechanism (zero-shot
   predict → reveal ground truth on misses → explain → summarize into a
   skill file → ablate on a held-out detection slice before trusting it),
   applied to `detection-multiclass`/`detection-box-verifier` instead of
   `vision-baseline`/`verifier-cot`. Deliberately sequenced AFTER
   classification proves out, not in parallel — no reason to pay for a
   second, detection-specific held-out dataset/pipeline build if the core
   idea doesn't even work for the simpler classification case first.

## Open questions / not yet decided

- Final `.md` file format/length and how it gets truncated/summarized if
  it grows too large for a single context injection.
- Map-reduce chunking strategy for Step 3 (fixed-size batches vs. grouped
  by heuristic category/crop) — not yet decided.
- Script/command names for Steps 3-4 — not yet built.
