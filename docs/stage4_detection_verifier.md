# Paper 2, Stage 4 — Detection Verifier

[← Back to README](../README.md)

Status: **generation step (Step 0) confirmed working on real hardware.
Verifier/filtering step (Step 1) run on real hardware, results so far
show filtering isn't helping (see its own QC section). Correct-and-
reverify step (Step 2) built, not yet run on real hardware.** This stage
is the spatial-task analog of
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

`box_rubric` is 3 criteria as of 2026-10-02 (criterion 4, "is the
sub-type/condition plausible," was dropped — see Step 2 below, which
reuses the remaining 3 as its own correct/reverify trigger and doesn't
need a 4th).

## Step 2 — box refine: correct-and-reverify, not just filter

Step 1 only grades — a box that fails a criterion gets a low confidence
and, at best, filtered out. This step takes actual corrective action
instead, one real generation call per failing criterion, then re-grades:

| Criterion fails | Correction | Mechanism |
|---|---|---|
| 3 (multiple instances merged) | **Split** | Shown the same padded crop, asked to localize each separate instance — may return 1 (disagrees), 2+ (real split), or 0 (drop) |
| 2 (box not tight) | **Tighten** | Same crop, asked for a single tighter box for the same instance |
| 1 (not a real instance) | **Reclassify or drop** | Same crop, shown this image's OTHER already-generated sub-class labels (detection-multiclass's own free-text vocabulary for this image, not a fixed taxonomy), asked which one (if any) this crop actually matches — no match drops the box |

Applied in **split → tighten → reclassify** order, not the rubric's own
1/2/3 numbering — splitting changes how many boxes exist, so tightening
or reclassifying a box that's about to be replaced by a split would waste
a correction. A box gets **at most one** correction per round (whichever
its current grade triggers, in that priority order), then is re-graded
fresh next round — a box's scores from a prior round are never trusted to
still apply after any correction.

```bash
uv run python main.py detection-box-refine \
    --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_refine_smoketest/ \
    --smoketest
```

Takes a `detection-multiclass` checkpoint directly (same input Step 1
takes) — any pre-existing `confidence`/`criterion_scores` on its boxes
are ignored, since every box is re-graded fresh in round 1 against the
current 3-criterion rubric regardless of what graded it before.
`--max_rounds` (default `2`) caps how many grade→correct→reverify rounds
a box gets before it's kept as-is, flagged `"exhausted"` in
`refinement_stats`, rather than looped forever.

**Read the result**: each row's `refinement_stats` (JSON) reports how
many split/tighten/reclassify calls and drops happened for that row —
e.g. `{"rounds_run": 2, "split_calls": 1, "tighten_calls": 0,
"reclassify_calls": 2, "dropped": 0}`. `pred_boxes_json` holds the FINAL
refined box set only (no separate unfiltered/filtered pair like Step 1 —
there's one pool here, the corrected one), each box carrying its own
final `refinement_status` (`"passed"` or `"exhausted"`) and
`refinement_rounds`. `metrics.json` reports mean precision/recall/F1 for
this final refined set, directly comparable to Step 1's own `filtered`
numbers (same `--iou_threshold`, same class-agnostic matching) to see
whether active correction beats pure filtering.

**QC**: same plotter, pointed at this checkpoint — `category_name` now
reflects any reclassification, and boxes are whatever survived
split/tighten/drop, so this is the most direct visual check of what the
pipeline actually settled on:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --color_by_subclass
```

(No `confidence` is shown in these labels — Step 2's final checkpoint
doesn't carry per-box confidence the way Step 1's does, since a box here
either passed outright or was corrected until it did/ran out of rounds,
not filtered by a threshold.)

Not yet validated: whether correct-and-reverify actually beats Step 1's
simpler grade-and-filter on real data — that's exactly what comparing
this step's `metrics.json` against Step 1's `filtered` numbers (same
`--ckpt` input, same `--iou_threshold`) is for.

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

**Step 2 (box refine) is an alternative to steps 2-4 above, not chained
after them** — it takes the SAME input `detection-box-verifier` does
(step 1's own `detection-multiclass` output), not `detection-box-verifier`'s
checkpoint:

```bash
uv run python main.py detection-box-refine \
    --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_refine/
```

Compare this run's `metrics.json` against `detection-box-verifier`'s own
`filtered` numbers (same `--iou_threshold`) to see whether active
correction is worth its extra GPU cost over simple grade-and-filter.
