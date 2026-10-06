# Paper 2, Stage 4 — Detection Context/Skill Memory Construction

[← Back to README](../README.md)

**Status: design sketch — Step −1 built (`detection-skill-representation-ablation`),
not yet run for real. Steps 0-5 still TODO.**

## Why this stage exists

Stage 3's classification skill-memory pipeline found a real (if
modest, n=199) signal: a reference document built from the model's own
past mistakes, injected into the system prompt, measurably changed
`judge_accuracy` on held-out images. The question this stage asks: does
the same idea transfer to object detection — can a model's own past
box-prediction mistakes be distilled into a reference document that
improves future box predictions?

The detection analog is NOT a straightforward copy, for one specific
reason: classification's "ground truth" is a short text label a model
can directly compare against its own text answer. Detection's "ground
truth" is a set of `[x1, y1, x2, y2]` coordinates — there was no
established reason to assume a model can usefully reason about two
sets of raw numbers the way it can reason about two words, so this got
tested empirically (Step −1) rather than assumed. **Resolved**: it
handles the raw coordinates fine — see Step −1's own result below.

## Step −1 (NEW) — box-representation ablation (raw-coords / overlay / cropped)

Before Step 2 (explain traces) can be built for real, decide how the
model is shown its own prediction vs. ground truth when asked to
explain a box mistake. THREE candidate conditions, two independent
questions bundled into one ablation since the infrastructure (sampled
discrepancies, per-sample generation) is shared:

- **(a) Raw coordinates as text, full image**: "Predicted box: `[120,
  45, 300, 210]`. Ground truth box: `[130, 50, 290, 200]`." — cheap, no
  extra rendering step, but asks the model to do spatial reasoning from
  numbers alone, which VLMs are not generally reliable at (no visual
  grounding for what those numbers actually look like relative to the
  image).
- **(b) Rendered overlay, full image**: both boxes drawn on the actual
  image (predicted solid lime, ground truth dashed red — the SAME
  convention `src/common/bbox_viz.py`'s `plot_predictions` already uses
  throughout Stage 6's own QC tooling, not a new drawing convention to
  invent). Turns the task into "describe what's visually wrong with
  this overlay" — a vision task, not a numeric-reasoning task.
- **(c) Rendered overlay, CROPPED**: same drawing as (b), but on a
  tight, padded crop around just the box(es) involved (padded UNION of
  both boxes when two are present, so a loose-match/worked sample's two
  non-overlapping boxes don't get one of them cut off — NOT
  `detection_box_verifier.py`'s own single-box `crop_box()`, which
  would risk exactly that). Might sharpen focus on a small/occluded
  object; might also lose surrounding context some error types need
  (e.g. judging whether two nearby instances should really be one
  merged box needs to see both of them).

**Built** (`src/detection_skill_representation_ablation.py`, CLI
`detection-skill-representation-ablation`). Deliberately QUALITATIVE,
not an automated metric — a human reads the output, nothing in this
step computes a score:

```bash
# Needs an existing detection-multiclass checkpoint first — reuse one
# already on disk from Stage 6's own work, or generate one fresh:
uv run python main.py detection-multiclass \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/detection_multiclass/

# Then the ablation itself:
uv run python main.py detection-skill-representation-ablation \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --n_samples 30
```

Writes 30 `comparison_sample<i>_<error_type>.png` files to
`outputs_final/detection_skill_representation_ablation/` (default
`--out_dir`) — read these by eye to decide which representation Step 2
builds on. `--crop_pad_frac` (default `0.25`) controls condition (c)'s
padding if the crop looks too tight or too loose on a first read.

- Source: a **`detection-multiclass`** checkpoint. Real local metrics
  decided this (`outputs_final/detection_multiclass/.../metrics.json`
  vs. `outputs_final/detection_baseline/.../metrics.json`,
  `RedHatAI/gemma-4-31B-it-FP8-dynamic`): multiclass wins on F1 (0.3773
  vs. 0.3531) and recall (0.4283 vs. 0.3501), trading away some
  precision (0.3997 vs. 0.4392) to get there. The real pipeline builds
  on multiclass (see Step 1 below), so this ablation does too.
- Samples up to 30 box-level DISCREPANCIES (not every box — only the
  interesting cases), round-robin across 3 error types: **false
  positive** (predicted box, no real instance), **missed** (real
  instance, no predicted box), **loose match** (a real IoU-matched
  pair, but under 0.7 IoU — too imprecise to be "correct" but not
  unmatched either).
- Each box-level discrepancy gets shown ONE AT A TIME — never the
  whole image's full box set at once (same per-box, not per-image,
  convention `detection_box_verifier.py` already established for
  grading; conflating several of an image's errors into one
  explanation call would make it ambiguous which sentence refers to
  which box). Which box(es) get shown is itself error-type-dependent —
  false_positive shows the predicted box only (no legitimate ground
  truth exists there), missed shows the ground-truth box only (no
  prediction exists), loose_match shows BOTH (the whole point is
  comparing an imprecise prediction against the real one).
- Three generation calls per sample (90 total for 30 samples, ONE
  shared vLLM load).
- Output: ONE comparison PNG per sample — question at top, the full and
  cropped annotated images side by side in the middle (both always
  rendered, for the human reader — not implying either model saw both),
  and all three conditions' raw text responses below, labeled (a)/(b)/(c).

**Decided, from reading the real 30-sample output**: (a) — raw
coordinates as text, plain original image, no overlay rendering at
all. The model handled the raw `[x1,y1,x2,y2]` coordinates well across
the sampled error types, which settles the "VLMs aren't reliable at
numeric spatial reasoning" concern this ablation existed to check —
at least for this model, on this task, it wasn't the problem the
design worried it might be. This is also the cheapest condition (no
`plot_predictions`/crop rendering step, one image per call instead of
choosing between two), so the decision is a clean win on both
correctness and cost, not a tradeoff. (b)/(c) stay built and available
in `detection_skill_representation_ablation.py` if a future category of
error turns out to need them, but Step 2 below builds on (a) alone.

## Step 0 — held-out detection dataset(s)

**Decided**: hold out ENTIRE datasets, not a stratified slice of
images within the existing pool. A handful of Project-AgML detection
datasets (TBD which, once Step 0 is actually built) are carved out
completely — never sampled into Stage 2/Stage 6's own eval runs at
all — mirroring DigiGreen's complete independence from the
classification eval pool. Cleaner separation than a stratified-image
split: no risk of a dataset's own images being similar enough to each
other that within-dataset lessons/eval leakage softens the held-out
claim. Tradeoff accepted: fewer, less diverse datasets in the lesson
pool, and a one-time choice of which datasets to remove from the main
eval set — worth it for the cleaner generalization claim.

## Step 1 — zero-shot detection prediction (lessons split)

Reuses **`detection-multiclass`**'s own generation mechanism directly
(not reimplemented) — consistent with Step −1's own choice. Real local
metrics decided this (see Step −1): multiclass beats plain
`detection-baseline` on F1 and recall, so the skill-memory pipeline
builds on the stronger generation mechanism rather than the simpler
one. (An earlier version of this doc reasoned the other way — start
from baseline so multiclass's value could emerge from the skill file's
own content as a finding, rather than being assumed — but that
reasoning doesn't hold once baseline is already the WEAKER condition by
a real, measured margin; there's no reason to deliberately start from
the worse generation mechanism once that's known.)

Scored class-agnostically via `bbox_metrics.match_boxes`/
`precision_recall_f1`, same IoU convention every other detection stage
in this repo uses. Output: a checkpoint with `pred_boxes_json`/
`gt_boxes_json` per image, same shape
`detection_box_verifier_summary.py`/`detection_final_metrics.py`
already know how to read.

## Step 2 — explain traces (lessons split, ALL rows not just misses)

Mirrors Stage 3's own design choice (explain both "what worked" and
"what didn't work", not just misses) and reuses the WINNING
representation from Step −1. Failure taxonomy explicitly mirrors
`detection_box_refine.py`'s own already-validated correction categories
— not invented fresh, since that pipeline already solved "what kinds of
box errors are there":

- **not a real instance** (false positive — box_rubric criterion 1)
- **box not tight** (too loose/too large/cut off — criterion 2)
- **multiple instances merged** (needs a split — criterion 3)
- **missed entirely** (a ground-truth box with no matching prediction
  at all — detection's analog of classification's "false Healthy": the
  model didn't even attempt the instance)
- **wrong class label on an otherwise-correct box** (real now that
  Step 1 runs `detection-multiclass` — its `category_name` is free-text
  sub-class text, not the dataset's one coarse class, so a box can
  genuinely be positioned right but labeled wrong)

**Per-box, not per-image, same as Step −1 already established**: each
explanation call shows ONE box-level discrepancy at a time (one
predicted box and/or one ground-truth box — never an image's full box
set at once). An image with several separate errors gets several
separate explanation calls, one per discrepancy, exactly like Step −1's
own sampling (`find_discrepancies`, reused directly — not
re-implemented) and like `detection_box_verifier.py`'s own per-box
crop/grade convention. Showing every box at once would make a single
free-text explanation ambiguous about which sentence refers to which
error — the same conflated-feedback problem a whole-image rubric call
would have.

Each explanation call shows: the plain original image + the raw
`[x1,y1,x2,y2]` coordinates as text (Step −1's settled representation,
condition (a) — no overlay rendering needed), and asks for 2-4
sentences of concrete, generalizable guidance — same "explain what
should have been seen instead" framing as `heldout_explain_miss_prompt`/
`heldout_explain_correct_prompt`, adapted to spatial language instead
of diagnostic language.

## Step 3 — summarize into skill_memory_detection.md (map-reduce)

Same map-reduce mechanic as `heldout_summarize.py`, chunked by the
Step 2 failure-taxonomy buckets instead of disease/pest/deficiency/
healthy. Same hard rules carried over (learned the expensive way in
Stage 3, not re-litigated here): no "common objects to know" bare-label
reference section, no bullet whose entire content is a bare category
word, write generalizable guidance not per-image facts. New rule
specific to detection: a bullet must describe a VISUAL/SPATIAL
pattern (shape, boundary, occlusion, clustering), not a coordinate
number — "boxes around clustered fruit tend to need splitting when a
stem or leaf visually separates two individuals" is useful; "move the
box 15px left" is not (even if that number happened to be true for one
example, it's exactly the per-image-memorization failure mode Stage 3's
own edit prompt now explicitly forbids).

## Step 3b — self-optimization loop (reuse, not rebuild)

`heldout_skill_optimize.py`'s machinery (automated round loop, round 0
= score as-is, edit against per-bucket scores + sampled misses, log
diff + changelog + rule-violations, select best round) is ALREADY
generic enough in its bucket/scoring abstractions to be reused for
detection with the bucket categories swapped (box-error taxonomy
instead of disease/pest/deficiency/healthy) and the scoring call
swapped (`detection-multiclass`'s own generation+matching instead of
`heldout-zeroshot`'s classification scoring). Worth checking after
Stage 3's own loop has run for real whether `--edit_enable_thinking`
staying off and the channel-leak/rule-violation QC checks are still
sufficient here, or whether detection's own failure modes need their
own additions to `check_rule_violations`.

## Step 4 — ablation

Same shape as Stage 3's Step 4: baseline (no skill) vs. not-optimized
skill vs. (once Step 3b has run) self-optimized skill, scored on the
held-out eval split, `precision`/`recall`/`F1` instead of
`judge_accuracy` (detection has no judge-based correctness metric —
IoU-matched P/R/F1 is the ground truth here, already deterministic, no
judge-leniency confound to worry about the way classification has).

## Step 5 — real injection

Gated the same way Stage 3's was (`DISEASE_TASK_TYPES`-style gate, but
here gated to whichever detection datasets the skill file actually has
real content for — likely NOT all 54 AgML detection datasets, since
Step 0's held-out pool is a subset). Injection point:
`detection-multiclass`'s system prompt — same generation mechanism
Steps 1/−1 were built on, consistent end to end. Whether plain
`detection-baseline` also benefits is a secondary question to test
afterward, not primary.

## Open questions, summarized

1. Step −1's qualitative result (raw-coords / overlay / cropped-overlay)
   — blocks Step 2 from being built for real. Script is built
   (`detection-skill-representation-ablation`); needs a real GPU run +
   a human reading the 30 output images.
2. Which specific AgML detection datasets to hold out entirely for
   Step 0 — not yet chosen.
3. Whether `heldout_skill_optimize.py`'s QC checks need detection-
   specific additions, decided after Stage 3's loop has run for real
   and its own QC checks have been stress-tested against real failures
   (same way `check_rule_violations`'s channel-leak check only exists
   because a real run caught a real bug).
