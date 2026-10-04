# Paper 2, Stage 3 — Context/Skill Memory Construction

[← Back to README](../README.md)

Status: **Step 0 (download) and Step 1 (zero-shot prediction) built; Steps
2-4 (reveal-and-explain, summarize, ablate) not started yet.** See
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
actual cached filename, `prediction`/`judged_correct`/`judge_reasoning`
as usual. (`agml_dataset`/`class_label`/`image_name` column names are
reused deliberately, not a stylistic choice — `load_ckpt`'s resume logic
hardcodes exactly those 3 names to build its seen-keys set.)

## Planned pipeline, Steps 2-4 (not built yet)

2. **Reveal ground truth on misses** — for each incorrect prediction,
   show the model its own wrong answer AND the real ground truth label
   (`diagnosis`/`details` columns), ask it to explain concretely what it
   should have looked for and why it got it wrong. (Deliberately not
   reusing `verifier_rubric`'s own evidence-gathering call for this —
   Stage 4 found that mechanism can pre-commit to a wrong diagnosis
   during "neutral" observation; this step needs the real label revealed
   to produce a trustworthy explanation, not another guess.)
3. **Summarize traces into one `.md` file** — aggregate the gathered
   explanations (plus a smaller sample of *correct* predictions, for
   "what worked") into a single skill-memory file: common diseases,
   what to look for, confusions to watch for.
4. **Ablate before trusting it** — zero-shot `judge_accuracy` on the
   **held-out eval slice**, with vs. without the skill file injected as
   extra context. Only a result that holds on held-out images counts as
   a real finding, not memorization of the lesson-building slice.
5. **If it helps**: inject into `vision-baseline` first (isolates the
   skill file's own effect cleanly), then `verifier-cot`'s generation
   step and/or `verifier_rubric_fill_prompt` if it still helps there.

## Open questions / not yet decided

- Whether "what worked" traces need the same reveal-and-explain
  treatment as "what didn't work," or whether the model's own
  (already-correct) reasoning can be trusted as-is.
- Final `.md` file format/length and how it gets truncated/summarized if
  it grows too large for a single context injection.
- Script/command names for Steps 1-4 — not yet built.
