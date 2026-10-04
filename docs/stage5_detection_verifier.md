# Paper 2, Stage 5 — Detection Verifier

[← Back to README](../README.md)

Status: **generation (Step 0) and correct-and-reverify (Step 1) both
built and syntax/logic-verified (synthetic dry runs + a real local
smoketest for Step 0); neither has had a full real-hardware run yet with
the current (2026-10-02) 3-criterion rubric.** This stage is the
spatial-task analog of [Stage 4](stage4_classification_verifier.md)'s
classification verifier, applied to object detection instead.

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
- **Multi-class generation + a separate scoring/correction step** (this
  stage) — the model reports every instance at every stage it can see,
  labeled with its own specific stage; a later step reconciles which
  stage(s) actually match a given dataset's ground-truth convention,
  instead of guessing that convention up front.

## The full flow, step by step

```
Step 0  detection-multiclass        generate every stage/condition's boxes, over-generating by design
   ↓
Step 1  detection-box-refine        crop each box, grade it, correct-and-reverify failing criteria
   ↓                                (no filtering — every surviving box kept, "passed" or "exhausted")
Step 2  detection-box-verifier-summary   sweep confidence_threshold × iou_threshold on Step 1's own
   ↓                                checkpoint — THIS is the only place filtering/thresholding happens
  (pick the threshold)              prints the best combo; doesn't write a filtered checkpoint itself
```

1. **detection-multiclass** (Step 0) generates boxes, deliberately
   over-generating — every growth stage/condition the model can see, not
   just whatever one stage ground truth happens to label.
2. **detection-box-refine** (Step 1) crops each box, grades it against a
   3-criterion rubric, and — unlike a plain grade-and-filter pass —
   actively corrects a failing box (split it, tighten it, or reclassify
   it) and re-grades, up to `--max_rounds` times. It does **not** filter
   anything out based on confidence; every box that survives correction
   (wasn't dropped by a correction finding nothing real) is kept in the
   output, whether it ultimately `"passed"` or ran out of rounds
   (`"exhausted"`) — every one of them carries an accurate, up-to-date
   `confidence`.
3. **detection-box-verifier-summary** (Step 2) is the *only* place
   filtering/thresholding happens in this flow. Run against Step 1's own
   checkpoint, it sweeps `confidence_threshold` × `iou_threshold`,
   reporting precision/recall/F1 at each combination and printing the
   single best one — but it's read-only analysis: it does **not** write
   a new filtered checkpoint. The threshold it reports is the number you
   now know to filter by (e.g. for whatever downstream use consumes
   `pred_boxes_json` next); there isn't yet a script that takes that
   threshold and materializes a final filtered checkpoint automatically.

A simpler, cheaper alternative to Step 1 also exists —
**detection-box-verifier** — which grades but never corrects anything,
just filters directly at a threshold you pick up front. It's documented
below Step 2 since Step 1 (refine) is the primary recommended path now;
verifier-summary works identically on either one's checkpoint, so you can
compare them head to head.

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
differentiating sub-types and the prompt needs another pass. **Confirmed
working on real data** (2026-10-01 local smoketest run): real labels like
`"flower: open yellow"`, `"flower: small white"`, `"flower: budding
yellow"` came back, not just `"flower"` repeated.

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

## Step 1 — box refine: correct-and-reverify

For each predicted box, crop it out of the source image (padded 15% of
the box's own width/height per side by default — `--crop_pad_frac`) and
show *just that crop* to the verifier model, independently of the other
boxes and the full-image context — a genuine second opinion, not the
same generation call grading its own output. Grade the crop against
`src/config/verifier_prompts_detection.yaml`'s `box_rubric` (**3**
criteria as of 2026-10-02 — a 4th, "is the sub-type/condition plausible,"
was dropped) independently on a 1-3 scale, summed to a confidence in
`[0, 1]` — same summed-rubric mechanic Stage 4 uses
(`src/verifier_cot.py`'s own `_parse_criterion_scores`/
`_normalize_criterion_scores`, reused directly), just applied per-box
instead of per-trajectory.

Unlike a plain grade-and-filter pass, a box that fails a criterion gets
**actual corrective action**, not just a low score:

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
still apply after any correction. A box still active after `--max_rounds`
(default `2`) gets one final grading pass (so its stored `confidence`
reflects its actual last state, not a stale pre-correction grade) and is
kept as-is, flagged `"exhausted"`.

**This step applies no filtering of its own** — every box that survives
correction (wasn't dropped) ends up in the output `pred_boxes_json`,
`"passed"` or `"exhausted"`, each with an accurate `confidence`.
Filtering happens entirely in Step 2, below.

**Thinking is on by default here too** (as of 2026-10-02) — grading and
all three correction calls (split/tighten/reclassify) use the same
`--enable_thinking`/`--chat_template_path` mechanism as
`detection-multiclass`'s own box generation, defaulting to *on*.
Originally these calls had no reasoning step at all, which made the
verifier a weaker, not-apples-to-apples second opinion against boxes
that were themselves generated *with* thinking — a likely real
contributor to the smoketest's baseline-beats-verifier finding noted in
Step 4 below. `--grading_max_tokens`/`--gen_max_tokens` were bumped from
300 to 1024 to leave headroom for a thinking trace before the short
criterion/box output. Same applies to `detection-box-verifier`'s own
grading call. Pass `--no-enable_thinking` on either script to opt out.

```bash
uv run python main.py detection-box-refine \
    --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_refine_smoketest/ \
    --smoketest
```

Takes a `detection-multiclass` checkpoint directly. Any pre-existing
`confidence`/`criterion_scores` on its boxes are ignored — every box is
re-graded fresh in round 1 against the current 3-criterion rubric
regardless of what graded it before.

**Read the result**: each row's `refinement_stats` (JSON) reports how
many split/tighten/reclassify calls and drops happened for that row —
e.g. `{"rounds_run": 2, "split_calls": 1, "tighten_calls": 0,
"reclassify_calls": 2, "dropped": 0}`. Each box in `pred_boxes_json`
carries its own `refinement_status` (`"passed"` or `"exhausted"`),
`refinement_rounds`, `confidence`, and `criterion_scores`. `metrics.json`
reports mean precision/recall/F1 for this *unfiltered* final set (every
surviving box counted, matching Step 2's own `confidence_threshold=0.0`
row in its sweep).

**QC #2 — after refinement, pre-filter**: same plotter, pointed at this
checkpoint — `category_name` now reflects any reclassification, and boxes
are whatever survived split/tighten/drop. Pass `--out_dir` explicitly
(same collision reason as the filtered QC further down) so this doesn't
land in the same default `qc_plots/` folder a later filtered run would
also use:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --color_by_subclass \
    --out_dir outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/qc_plots_unfiltered
```

(Confidence numbers show up in each box's label automatically — same
`bbox_viz.py` behavior as Step 2's own checkpoints, see below.)

## Step 2 — choose a confidence threshold

Run against **Step 1's own checkpoint** — this is the only place
filtering/thresholding happens in this flow:

```bash
uv run python main.py detection-box-verifier-summary \
    --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

No GPU needed — every box's `confidence` was already graded once by Step
1 and is stored on it; this is pure CPU re-filtering + re-matching.
Writes to two separate folders:

**`--out_dir` (default `<ckpt's own dir>/threshold_sweep/`)** — aggregate
statistics:

- `f1_vs_confidence_threshold.png` — F1 at every `confidence_threshold`
  (`0.0`-`0.9` in steps of `0.1`) × `iou_threshold` (`0.3`/`0.5`/`0.75`)
  combination, one line per IoU threshold, peak marked with a star.
- `confidence_distribution.png` — a pooled histogram of every box's own
  confidence score, across the whole checkpoint.
- `criterion_score_distribution.png` — a grouped bar chart, one color per
  rubric criterion, score (1/2/3) on the x-axis — shows WHICH criterion is
  actually doing the discriminating work vs. which ones are nearly always
  satisfied (and therefore contributing little to the final confidence
  spread).
- `threshold_sweep.csv` — the full grid.

**`--samples_dir` (default `<ckpt's own dir>/refine_samples/`)** — per-box
visual inspection:

- `crop_gallery.png` — actual box crops (re-cropped the same way the
  verifier itself was shown them) paired with their confidence +
  per-criterion scores, **stratified by confidence rank** (not random) so
  the gallery always spans low-to-high even when the underlying
  distribution skews heavily toward one end.
- `<dataset>/split_timeline.png`, `<dataset>/tighten_timeline.png`,
  `<dataset>/reclassify_timeline.png` — **Step 1 checkpoints only** (Step
  1's own `detection-box-refine` records a `history` list per box — one
  snapshot per round, with that round's bbox/label/scores/confidence and
  which correction was decided; Step 1's alternative,
  `detection-box-verifier`, never corrects anything so it has no history
  to show). One subfolder per dataset, one figure per correction TYPE —
  grouped by each box's FIRST correction, not mixed together — one row
  per sampled box, one column per round, showing the crop actually
  changing (tightened, relabeled, split) step by step, not just the final
  state. A split box's children share the same early columns (the
  pre-split lineage) and diverge from the split round onward. Sampled
  longest-history-first per figure, skipped entirely (with a warning, not
  an error) if that dataset has no boxes of that correction type.

Prints the single best `(confidence_threshold, iou_threshold)` combination
to stdout.

**This part is read-only analysis — it does not write a filtered
checkpoint.** For the actual deliverable metrics at your chosen threshold,
see the next step.

## Step 3 — final metrics at your chosen threshold

```bash
uv run python main.py detection-final-metrics \
    --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --confidence_threshold 0.6 --iou_threshold 0.3
```

Two variants, both computed at the exact `(confidence_threshold,
iou_threshold)` you pass:

- **Variant A — class-agnostic**: every box that passes the confidence
  threshold is pooled together and matched against ground truth, no
  sub-class distinction at all — exactly the matching this whole pipeline
  has used throughout. This is the real, deployable number.
- **Variant B — per-image oracle sub-class selection**: for EACH image
  independently, every distinct sub-class label present in its own
  filtered predictions is tried ALONE against that image's own ground
  truth, and whichever gives the highest F1 is kept (the rest of that
  image's boxes are dropped). **This is an oracle/upper-bound metric, not
  deployable** — the selection itself looks at ground truth to decide
  which boxes to keep, which nothing real could do. It answers "what's
  the best possible result if sub-class selection were perfect," useful
  for seeing the ceiling, not as a number to report as achieved
  performance. Deliberately **per-image**, not a per-dataset argmax over
  exact sub-class strings — an earlier version tried grouping by exact
  text across a whole dataset and it badly underperformed Variant A,
  because free-text labels vary in wording across images for the same
  real stage (e.g. `"flower: open"` vs `"flower: open yellow"`), which
  fragmented one real signal into many small, individually weak buckets.
  Per-image selection needs no cross-image text matching at all, since
  each image only ever compares among its own locally-present labels.

Writes `final_metrics_conf<X>_iou<Y>.json` (to `--out_dir`, default
`<ckpt's own dir>/final_metrics/`) with the full per-dataset breakdown for
both variants, plus (Variant B) which sub-class won each image, pooled
into a per-dataset `chosen_subclass_counts`.

## QC #3 — after filtering

`detection-qc-plot` now takes `--confidence_threshold` — drops boxes below
it before plotting (boxes with no `confidence` at all, e.g. a plain
`detection-baseline`/`detection-multiclass` checkpoint, are always kept,
never dropped). When set, the title's P/R/matched numbers are recomputed
on the filtered set at `--iou_threshold`, not the checkpoint's own stored
(unfiltered) columns, so what's printed always matches what's drawn.

**IMPORTANT — always pass `--out_dir` explicitly here.** Without it,
`detection-qc-plot` defaults to `<ckpt's own dir>/qc_plots/` **every
time**, regardless of `--confidence_threshold` — QC #2 (above) and this
filtered run against the same checkpoint will silently write into the
exact same folder and overwrite each other's images, with no warning.
Give the filtered run its own name:

```bash
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --confidence_threshold 0.6 --iou_threshold 0.3 \
    --color_by_subclass \
    --out_dir outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/qc_plots_filtered
```

**The three QC checkpoints in this stage, all using the same
`detection-qc-plot --color_by_subclass` command, just pointed at a
different checkpoint/threshold each time:**

| # | After | Checkpoint | `--confidence_threshold` | Folder |
|---|---|---|---|---|
| 1 | multi-class generation | `detection_multiclass`'s own ckpt | *(none)* | `qc_plots/` (under the multiclass output dir) |
| 2 | refinement, pre-filter | `detection_box_refine`'s own ckpt | *(none)* | `qc_plots_unfiltered/` (under the refine output dir) |
| 3 | filtering | `detection_box_refine`'s own ckpt | your chosen value | `qc_plots_filtered/` (under the refine output dir) |

## Step 4 — compare against baseline

```bash
uv run python main.py detection-comparison-summary \
    --baseline_ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --verifier_ckpt outputs_final/detection_box_verifier_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --refine_ckpt outputs_final/detection_box_refine_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --confidence_threshold 0.6 --iou_threshold 0.3
```

No GPU needed. One grouped bar chart, Precision/Recall/F1 on the x-axis,
four conditions compared at the exact same `--iou_threshold` (so the
comparison is fair regardless of what IoU threshold each checkpoint's own
original run happened to use):

- **Baseline** (`--baseline_ckpt`, plain `detection-multiclass`/
  `detection-baseline`, no confidence field) — drawn as a **dashed red
  reference line**, not a bar, one short segment per metric at that
  metric's own baseline value (not a single flat line — baseline's
  precision/recall/F1 are 3 different numbers).
- **Verifier (no refine)** (`--verifier_ckpt`, `detection-box-verifier`) —
  grades but never corrects, filtered at `--confidence_threshold`.
- **Verifier + refine** (`--refine_ckpt`, unfiltered — every surviving
  box kept regardless of confidence).
- **Verifier + refine + filter** (`--refine_ckpt` again, this time
  filtered at `--confidence_threshold`) — the actual end-to-end pipeline
  result.

Writes `comparison_conf<X>_iou<Y>.png` + the same numbers as
`comparison_conf<X>_iou<Y>.json` (to `--out_dir`, default
`<refine_ckpt's own dir>/comparison/`), and prints all four conditions'
P/R/F1 to stdout.

## Optional simpler alternative: detection-box-verifier (grade + filter, no correction)

A cheaper, non-corrective alternative to Step 1: crops and grades each
box the exact same way, but only ever filters — never splits, tightens,
or reclassifies anything. Useful as a baseline to compare Step 1's extra
correction cost against.

```bash
uv run python main.py detection-box-verifier \
    --ckpt outputs_final/detection_multiclass_smoketest/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/detection_box_verifier_smoketest/ \
    --smoketest
```

Unlike Step 1, this DOES filter on its own at run time
(`--confidence_threshold`, default `0.5`) — `metrics.json` reports
`unfiltered` vs. `filtered` precision/recall/F1 side by side. Re-run
`detection-box-verifier-summary` (Step 2) against *this* checkpoint
instead to sweep thresholds for it the same way.

## Full pipeline (after the smoke tests pass)

Recommended flow, no `--smoketest`, over the whole active detection
manifest — everything from the smoke-test walkthrough above, in order,
with every output folder named so nothing overwrites anything else:

```bash
MODEL=RedHatAI/gemma-4-31B-it-FP8-dynamic
SAFE=RedHatAI_gemma-4-31B-it-FP8-dynamic   # safe_model_name(MODEL) — matches every ckpt filename below

# ── Step 0 — multi-class generation, every active detection dataset. ──────────
uv run python main.py detection-multiclass \
    --model $MODEL \
    --output_dir outputs_final/detection_multiclass/

# QC #1 — after multi-class generation.
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_multiclass/$SAFE/ckpt_$SAFE.csv \
    --color_by_subclass

# ── Step 1 — crop, grade, correct-and-reverify (no filtering). ────────────────
uv run python main.py detection-box-refine \
    --ckpt outputs_final/detection_multiclass/$SAFE/ckpt_$SAFE.csv \
    --output_dir outputs_final/detection_box_refine/

# QC #2 — after refinement, pre-filter.
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_refine/$SAFE/ckpt_$SAFE.csv \
    --color_by_subclass \
    --out_dir outputs_final/detection_box_refine/$SAFE/qc_plots_unfiltered

# ── Step 2 — sweep to find the threshold that actually maximizes F1 (no GPU needed). ──
uv run python main.py detection-box-verifier-summary \
    --ckpt outputs_final/detection_box_refine/$SAFE/ckpt_$SAFE.csv
# Note the best (confidence_threshold, iou_threshold) it prints — used as
# CONF/IOU below.
CONF=0.6   # <- replace with the sweep's own best confidence_threshold
IOU=0.3    # <- replace with the sweep's own best iou_threshold

# ── Step 3 — final metrics at that threshold. ──────────────────────────────────
uv run python main.py detection-final-metrics \
    --ckpt outputs_final/detection_box_refine/$SAFE/ckpt_$SAFE.csv \
    --confidence_threshold $CONF --iou_threshold $IOU

# QC #3 — after filtering.
uv run python main.py detection-qc-plot \
    --ckpt outputs_final/detection_box_refine/$SAFE/ckpt_$SAFE.csv \
    --confidence_threshold $CONF --iou_threshold $IOU \
    --color_by_subclass \
    --out_dir outputs_final/detection_box_refine/$SAFE/qc_plots_filtered

# ── Optional — the simpler, non-corrective alternative, for comparison. ───────
uv run python main.py detection-box-verifier \
    --ckpt outputs_final/detection_multiclass/$SAFE/ckpt_$SAFE.csv \
    --output_dir outputs_final/detection_box_verifier/

# ── Step 4 — compare baseline vs. verifier (no refine) vs. verifier+refine vs. verifier+refine+filter. ──
uv run python main.py detection-comparison-summary \
    --baseline_ckpt outputs_final/detection_multiclass/$SAFE/ckpt_$SAFE.csv \
    --verifier_ckpt outputs_final/detection_box_verifier/$SAFE/ckpt_$SAFE.csv \
    --refine_ckpt outputs_final/detection_box_refine/$SAFE/ckpt_$SAFE.csv \
    --confidence_threshold $CONF --iou_threshold $IOU
```

Not yet validated on the full dataset: whether Step 1's active correction
actually beats the simpler `detection-box-verifier` grade-and-filter
alternative, or either beats plain baseline detection — Step 4's bar
chart is exactly what answers that, once the full run (not just the
20-image smoketest) has real numbers to compare.
