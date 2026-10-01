# Paper 2, Stage 4 — Detection Verifier

[← Back to README](../README.md)

Status: **generation step (Step 0) confirmed working on real hardware.
Verifier/filtering step (Step 1) built, not yet run on real hardware.**
This stage is the spatial-task analog of
[Stage 3](stage3_classification_verifier.md)'s classification verifier,
applied to object detection instead.

## Why this stage exists

QC on Stage 2's thinking-enabled smoke test
(`GEMINI_cowpea_flower_detection`) surfaced a real finding: the model
correctly detects flowers at every growth stage — buds, open blooms, dead/
senesced flowers — but ground truth only labels open/senesced blooms as
`"flower"`. Under `detection-baseline`'s single-class scoring, every
correctly-localized bud or dead flower counts as a false positive, even
though the model saw it right.

Two ways to fix this were considered:

- **Per-dataset prompt curation** (tell the model each dataset's own
  labeling convention, e.g. "only count open blooms") — rejected: doesn't
  generalize to zero-shot use (a user uploading their own photo with no
  known dataset convention to curate against).
- **Multi-class generation + a separate scoring/filtering step** (this
  stage) — the model reports every instance at every stage it can see,
  labeled with its own specific stage; a later step reconciles which
  stage(s) actually match a given dataset's ground-truth convention,
  instead of guessing that convention up front.

## Step 0 — smoke test: does multi-class generation work at all?

Same defaults as [Stage 2](stage2_object_detection.md)'s baseline
(thinking enabled, vendored Gemma-4 template, `--gen_max_tokens 4096`) —
only the prompt differs (`detection_multiclass` /
`detection_multiclass_json` in `src/config/prompts_detection.yaml`, not
`detection_baseline`'s own prompt, which stays untouched). Scoring is
still class-agnostic IoU matching (`bbox_metrics.match_boxes` never reads
a box's category), so precision/recall/F1 work exactly like Stage 2's —
only each box's stage/condition label is new.

```bash
uv run python main.py detection-multiclass \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/detection_multiclass_smoketest/ \
    --smoketest
```

`--smoketest` restricts the run to two GEMINI cowpea datasets —
`GEMINI_cowpea_flower_detection` (990 images, where the original finding
came from) and `GEMINI_cowpea_pod_detection` (young/mature/dried pods,
the same growth-stage-variation shape) — both need to be present in
`--config`/`--manifest`; see [Stage 2](stage2_object_detection.md)'s smoke
test section for the `download-detection-datasets --only` pattern if
either isn't downloaded yet.

What to check in the checkpoint: `pred_boxes_json`'s `category_name` field
per box should now contain `"category: sub_type_or_condition"` (e.g.
`"flower: open flower"`, `"flower: bud"`) instead of just repeating the
dataset's one class name — if every box still comes back labeled
identically to the dataset's class name, the model isn't actually
differentiating sub-types and the prompt needs another pass.

`detection-multiclass` only runs generation + scoring — it does not plot
anything itself. `detection_multiclass.py` writes the same
`DET_CKPT_FIELDS`-shaped checkpoint `detection_baseline.py` does, so
[Stage 2](stage2_object_detection.md)'s QC plotter works on it unchanged,
just pointed at the new checkpoint:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --color_by_subclass
```

`--color_by_subclass` colors each predicted box by its own `category_name`
value (e.g. `"flower: bud"` vs. `"flower: open flower"`) instead of one
flat lime — the whole point of this stage's QC is seeing whether the model
actually differentiates sub-types, which a single color would hide. Ground
truth stays dashed red regardless. Off by default on
`detection-qc-plot` — a plain `detection-baseline` checkpoint's
`category_name` is just the dataset's one class repeated on every box, so
this flag would add a legend with no real information there.

This is the visual confirmation that the model is actually drawing boxes
around different growth stages/conditions (not just ground truth's one
labeled stage) — the same QC step that originally surfaced this stage's
motivating finding.

## Step 1 — per-box verifier: crop, grade, filter

Once multi-class generation is confirmed working (Step 0 above), this
step scores *confidence* per predicted box rather than trusting every box
equally:

1. **Crop-and-regrade.** For each predicted box, crop it out of the
   source image (padded 15% of the box's own width/height per side by
   default — `--crop_pad_frac`) and show *just that crop* to the verifier
   model, independently of the other boxes and the full-image context —
   a genuine second opinion, not the same generation call grading its own
   output.
2. **Rubric, not a single trajectory score.** Grade each crop against
   `src/config/verifier_prompts_detection.yaml`'s `box_rubric` (4
   criteria: is this a real instance of the stated category, is the box
   reasonably tight, is it exactly one instance not several merged, is
   the stated sub-type/condition plausible) independently on a 1-3 scale,
   summed to a confidence in `[0, 1]` — same summed-rubric mechanic Stage
   3 uses (`src/verifier_cot.py`'s own `_parse_criterion_scores`/
   `_normalize_criterion_scores`, reused directly), just applied per-box
   instead of per-trajectory. No ground truth is shown to the grader —
   purely a visual plausibility check against the box's own stated label.
   Each box's individual `criterion_scores` and `verifier_feedback` (not
   just the summed confidence) are stored in the checkpoint, same
   convention as Stage 3's own per-criterion storage.
3. **Filter before reporting.** Boxes with confidence below
   `--confidence_threshold` (default `0.5`, no principled value yet — see
   below) are dropped into `pred_boxes_filtered_json`, a *separate* field
   from the original (unfiltered) `pred_boxes_json` — the checkpoint keeps
   both, so you can directly compare before/after instead of only seeing
   the filtered number.
4. **Scoring stays class-agnostic.** IoU-match the filtered box set
   against ground truth exactly as Step 0 already does
   (`bbox_metrics.match_boxes`) — no new scoring machinery, re-scored at
   `--iou_threshold` (default `0.5`, independent of the input
   checkpoint's own threshold).

```bash
uv run python main.py detection-box-verifier \
    --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_verifier_smoketest/ \
    --smoketest
```

`--smoketest` here restricts `--ckpt`'s own rows to the same two GEMINI
cowpea datasets Step 0's `--smoketest` uses — same explicit-intent
convention as `detection-multiclass`, not just "whichever checkpoint path
you happened to point `--ckpt` at."

**QC**: same plotter as Step 0, pointed at the box-verifier's own checkpoint:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_verifier_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --color_by_subclass
```

Note: this reads `pred_boxes_json` (the full graded pool, every box's
`confidence`/`criterion_scores` now attached but not yet distinguished by
color) — it does NOT currently plot `pred_boxes_filtered_json` separately,
so a filtered-out low-confidence box still renders the same as a kept one
here. Useful for confirming the grading step ran and attached real scores
(inspect a box's JSON directly for its `confidence`), not yet a visual
kept-vs-dropped comparison.

**Read the result**: `metrics.json` reports mean precision/recall/F1 for
both `unfiltered` (identical to the input checkpoint's own numbers) and
`filtered` (after dropping low-confidence boxes) side by side — this is
the direct before/after comparison that answers whether filtering
actually helps, same "compare relatively, don't trust one absolute
number" approach already used for classification's judge/verifier
results in this project (ground-truth conventions are fuzzy here too, not
just for disease naming).

`--confidence_threshold`'s default (`0.5`) is a starting guess, not a
tuned value. To actually find the threshold that maximizes performance,
sweep it — no GPU/re-grading needed, since confidence was already graded
once and is stored on every box:

```bash
uv run python main.py detection-box-verifier-summary \
    --ckpt outputs_final/detection_box_verifier_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Re-derives precision/recall/F1 at every `confidence_threshold` (`0.0`-`0.9`
in steps of `0.1`) × `iou_threshold` (`0.3`/`0.5`/`0.75`) combination
directly from the checkpoint's own stored per-box `confidence` — pure CPU
re-filtering, not new grading calls. Writes `f1_vs_confidence_threshold.png`
(one F1 line per IoU threshold, peak marked) and `threshold_sweep.csv`
(the full grid), and prints the single best combination to stdout.

Open question not yet resolved: how many rubric criteria are worth the
extra GPU call per box before diminishing returns — 4 was a starting
point, not validated against alternatives.

## Full pipeline (after the smoke tests pass)

Same three scripts, no `--smoketest`, run in order over the whole active
detection manifest:

```bash
# 1. Multi-class generation — every active detection dataset.
uv run python main.py detection-multiclass \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/detection_multiclass/

# 2. Per-box verifier — crop, grade, filter (default --confidence_threshold 0.5).
uv run python main.py detection-box-verifier \
    --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_verifier/

# 3. Sweep to find the threshold that actually maximizes F1 (no GPU needed).
uv run python main.py detection-box-verifier-summary \
    --ckpt outputs_final/detection_box_verifier/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

If step 3's best `confidence_threshold` differs meaningfully from the
default (`0.5`) used in step 2, re-run step 2 with that value to
regenerate the final `pred_boxes_filtered_json`/`metrics.json` at the
tuned threshold:

```bash
uv run python main.py detection-box-verifier \
    --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_verifier/ \
    --confidence_threshold <tuned value> \
    --rerun
```

`--rerun` is required here, not just a different threshold flag — every
box's `confidence` is graded once and the result is keyed by
`(dataset, class, image_name)` in the verifier's own checkpoint, so a
plain resume would see those rows as already done and skip them instead
of re-filtering at the new threshold. This re-grades every box (the same
GPU cost as step 2 again) — there is no cheaper "just re-filter, don't
re-grade" path built yet, since grading and filtering happen in the same
pass.
