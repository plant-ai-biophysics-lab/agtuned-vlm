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

## Step 0 — held-out detection dataset(s) (built)

**Decided**: hold out ENTIRE datasets, not a stratified slice of
images within the existing pool — mirroring DigiGreen's complete
independence from the classification eval pool. Cleaner separation
than a stratified-image split: no risk of a dataset's own images being
similar enough to each other that within-dataset lessons/eval leakage
softens the held-out claim.

**Final design: a real 3-way split, by DATASET MEMBERSHIP, not a
within-dataset random split.** Of the ~54-dataset catalog:

- **TRAIN (10)** — generates the lessons/explanations the skill file is
  built from:

  | Dataset | Why |
  |---|---|
  | `synthetic_cowpea_flower_detection` | synthetic, cheap/scalable |
  | `synthetic_cowpea_pod_detection` | synthetic, cheap/scalable |
  | `grape_detection_syntheticday` | synthetic, cheap/scalable |
  | `mango_growth_detection` | growth-stage distinctions (4 classes) |
  | `MH_Weed16_weed_detection` | extreme multiclass (15 species) — stress-tests wrong-class-label |
  | `ghai_strawberry_fruit_detection` | organ-level distinctions (10 classes) |
  | `cashew_detection` | multi-scale (tree + nuts) — stress-tests merged-vs-missed |
  | `weed_crop_detection` | crop-vs-weed across many species (13 classes) |
  | `tomato_ripeness_detection` | ripeness-stage multiclass |
  | `onionfoliageset_detection` | crop/no-crop/foliage distinction |

- **VAL (5)** — scores every round of Step 3b's self-optimization loop,
  NEVER shown to the edit call, selects which round's skill file to
  keep only (same TRAIN-steers/VAL-selects separation Step 3b's own
  classification design uses):

  | Dataset | Why |
  |---|---|
  | `embrapa_wgisd_grape_detection` | real grape |
  | `grape_detection_californianight` | real grape, different lighting |
  | `wheat_head_counting` | grain/counting — very different structure from TRAIN's own picks |
  | `CocoaMFDB_detection` | different crop family, simple single-class |
  | `coffee_detection` | multi-scale tree+fruit, different crop than cashew |

  (An earlier pick had VAL strictly domain-paired with TRAIN's 3
  synthetic datasets — `GEMINI_cowpea_flower_detection`,
  `GEMINI_cowpea_pod_detection`, `grape_detection_californiaday` — for a
  sim-to-real transfer test. Moved to TEST instead: VAL's job here is a
  representative general validation signal across rounds, not a strict
  synthetic/real pairing.)

- **TEST (~39, the remainder)** — read for a FINAL confirmatory score
  only, after the self-optimization loop has already picked a winning
  round via VAL. Never used to generate lessons, never scored mid-loop
  — stays in Stage 2/6's own active pool, untouched, no marking needed
  (the only thing that would compromise it is using it to GENERATE
  content, which nothing in this stage does).

Marked `skip: true` in `src/config/detection_datasets.yaml` for all 15
TRAIN+VAL datasets (each with a `notes` marker explaining why) — this
is the REAL, functional exclusion mechanism, since
`download_and_sample_detection.py` (Stage 2/6's own sampler) only ever
selects from `active_datasets()`. Active pool: 54 → 39.

```bash
uv run python main.py download-heldout-detection
```

Reuses `download_and_sample_detection.py`'s own `process_dataset()`
(class-balanced selection) and `dataset_config.py`'s own
`load_hf_detection_samples()` (HF download + local image/annotation
caching) directly — not reimplemented. Writes
`dataset/images_detection_heldout/` (a separate cache dir from
`dataset/images_detection/`, so a held-out image can never collide
with Stage 2/6's own eval cache) and
`dataset/detection_heldout_selected_images.json` — one record per row
(`agml_dataset`, `image_path`, `boxes`, `split`), `"split"` DETERMINED
BY WHICH GROUP THE ROW'S DATASET IS IN (`"lessons"` for TRAIN,
`"eval"` for VAL), not a randomized within-dataset split. Resumable.

**Merge-safety**: `migrate-detection-datasets` regenerates
`detection_datasets.yaml` from scratch on every run but is now
merge-safe (fixed directly — see that script's own module docstring,
`scripts/agml_hf/migrate_detection_datasets_yaml.py`) — a deliberate
`skip: true` holdout (notes starting with `"HELD OUT for Paper 2 Stage
4"`) and any existing dataset's hand-filled `crop` field both survive a
re-run automatically. This was fixed for real, not just documented as
a caveat, the moment a real re-catalog run (to pick up 2 new synthetic
datasets added to the Project-AgML org) needed it.

**Real repo-side data bug found and worked around while building
this**: `synthetic_cowpea_flower_detection`/`synthetic_cowpea_pod_detection`
declare a config (`"synthetic"`) in their own README whose `data_files`
glob pattern matches zero real files — `get_dataset_config_names()`
raises `EmptyDatasetError` for ANY config name passed, not just the
broken one. Fixed generally (not hardcoded to these two repo names) in
both `scripts/agml_hf/catalog_datasets.py` and
`scripts/agml_hf/catalog_detection_datasets.py`: on that specific
error, bypass the repo's own module/config resolution entirely and
load via the generic `"parquet"` builder pointed at explicit `hf://`
URIs for the real files (found via `huggingface_hub.list_repo_files`,
confirmed real layout: `augmented/train-*.parquet` for both). Their
`class_names` still came back unresolved even after that fix (category
is a bare int `0` in the real schema, no `ClassLabel`/`.names`
anywhere) — filled in manually (`['flower']`/`['pods']`, matching the
sibling `GEMINI_cowpea_flower_detection`/`GEMINI_cowpea_pod_detection`
naming convention already in the catalog).

## Step 1 — zero-shot detection prediction (built)

**Built** (`src/heldout_detection_zeroshot.py`, CLI
`heldout-detection-zeroshot`) — detection analog of
`heldout_zeroshot.py` (Stage 3's own Step 1). Reuses
**`detection-multiclass`**'s own generation+scoring mechanism directly
(`detection_baseline.py`'s `run_batch`/`score_row`/`DET_CKPT_FIELDS` +
`detection_multiclass.py`'s own prompt key, imported not
reimplemented) — consistent with Step −1's own choice. Real local
metrics decided this (see Step −1): multiclass beats plain
`detection-baseline` on F1 and recall, so the skill-memory pipeline
builds on the stronger generation mechanism rather than the simpler
one.

Reads `dataset/detection_heldout_selected_images.json` directly (Step
0's own output) — NOT `active_datasets()`/`DatasetConfig`, since these
15 TRAIN+VAL datasets are deliberately marked `skip: true` to stay out
of that normal pipeline entirely. Per-dataset class list/crop for the
prompt's `{class_names}`/`{crop}` slots come straight from
`detection_datasets.yaml`'s own `classes`/`crop` fields — no
`DatasetConfig` object needed (no configured aliases for these
datasets).

```bash
uv run python main.py heldout-detection-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split lessons \
    --output_dir outputs_final/heldout_detection_zeroshot_lessons/
```

`--split lessons` (default) = TRAIN, the primary use here, feeds Step
2's explain-traces. `--split eval` = VAL, for Step 3b's
self-optimization loop and Step 4's ablation later — never use `eval`
to write the skill-memory file itself.

Scored class-agnostically via `bbox_metrics.match_boxes`/
`precision_recall_f1`, same IoU convention every other detection stage
in this repo uses. Output: a `DET_CKPT_FIELDS`-shaped checkpoint with
`pred_boxes_json`/`gt_boxes_json` per image, same shape
`detection_box_verifier_summary.py`/`detection_final_metrics.py`
already know how to read. `DET_CKPT_FIELDS` already carries
`agml_dataset`/`class_label`/`image_name` natively, so (unlike
classification's `HELDOUT_CKPT_FIELDS`) nothing needed repurposing to
satisfy `load_ckpt`'s hardcoded resume-key contract.

## Step 2 — explain traces (lessons split, ALL rows not just misses)

Built: `src/heldout_detection_explain_traces.py`.

Mirrors Stage 3's own design choice (explain both "what worked" and
"what didn't work", not just misses) and reuses the WINNING
representation from Step −1 (condition (a), raw coordinates + plain
image — no overlay rendering here at all). Failure/success taxonomy
(`error_type`, 4 values) names `detection_box_refine.py`'s own
already-validated correction categories for the first 3 — not invented
fresh, since that pipeline already solved "what kinds of box errors are
there" — plus one more (`worked`) for Stage 3's own "explain what
worked, not just misses" design choice, mirrored here:

- **false_positive** (box_rubric criterion 1, "not a real instance") —
  a predicted box with no matching ground-truth instance at all.
- **missed** — a ground-truth box with no matching prediction at all
  (detection's analog of classification's "false Healthy": the model
  didn't even attempt the instance).
- **loose_match** (criterion 2/3, "not tight") — a REAL match (passes
  `--iou_threshold`) but below `--loose_iou_threshold`: too loose/large/
  cut off, OR covers more than one real instance and needs splitting.
  The specific "too loose" vs. "needs splitting" distinction is left to
  the model's own free-text explanation — Step 3's map-reduce summary
  is what's expected to surface "clustered instances need splitting" as
  a recurring pattern, not a separately-coded detector.
- **worked** — a tight match (>= `--loose_iou_threshold`).

**No separate "wrong class label" category**, despite predicted boxes
carrying their own `category_name` — deliberately, matching a
precedent already established everywhere else in this pipeline:
`detection_baseline.py`'s own `score_row` (`match_boxes` +
`precision_recall_f1`) NEVER reads `category_name`, because
`detection-multiclass` predicts an open-vocabulary free-text SUB-class
(e.g. "grape: ripe cluster") while ground truth only ever carries the
dataset's one coarse class label (e.g. "grape") — the two were never
meant to be string-compared. A first version of this step DID compare
them directly and produced a real false positive on the actual lessons
checkpoint: a correct "grape: ripe cluster" prediction (IoU=0.86) was
flagged `wrong_class_label` against ground truth whose own label was
literally `"1"` (`Project-AgML/grape_detection_syntheticday`'s own
upstream `ClassLabel(names=['grape', '1'])` — a genuinely uninformative
class name, not fixable from this side), and more generally would have
misfired on ANY correct prediction whose free-text sub-class wasn't a
byte-for-byte match to the GT's single coarse label. A loose
substring-match compromise was considered and rejected — it still
can't resolve a genuinely uninformative GT label like `"1"` — so
category agreement is dropped entirely: tight IoU is "worked",
regardless of what either box's `category_name` says.

One pass of `bbox_metrics.match_boxes` per row yields all 4 categories
directly (`enumerate_items`) — not Step −1's own `find_discrepancies`
(covers false_positive/missed/loose_match only) plus a second pass for
`worked`, which would call `match_boxes` on the same row twice for no
reason.

**Per-box, not per-image, same as Step −1 already established**: each
explanation call shows ONE box-level item at a time (one predicted box
and/or one ground-truth box — never an image's full box set at once).
An image with several separate items gets several separate explanation
calls, exactly like Step −1's own sampling and like
`detection_box_verifier.py`'s own per-box crop/grade convention.
Showing every box at once would make a single free-text explanation
ambiguous about which sentence refers to which error — the same
conflated-feedback problem a whole-image rubric call would have.

Each explanation call shows: the plain original image + the raw
`[x1,y1,x2,y2]` coordinates as text, and asks for 2-4 sentences of
concrete, generalizable guidance — same "explain what should have been
seen instead" framing as `heldout_explain_miss_prompt`/
`heldout_explain_correct_prompt`, adapted to spatial language instead
of diagnostic language (`src/config/detection_skill_prompts.yaml`'s
`detection_skill_question_*` templates).

`box_idx` (0, 1, 2, ... per image, in a fixed deterministic enumeration
order) plus `error_type` stand in for `class_label` in `load_ckpt`'s
hardcoded 3-field resume key (`agml_dataset`/`class_label`/
`image_name`) — same repurposing trick `heldout_zeroshot.py`'s own
`HELDOUT_CKPT_FIELDS` used for a different reason, needed here because
several box-level items share one `(agml_dataset, image_name)` pair,
which alone isn't a unique resume key.

```bash
uv run python main.py heldout-detection-explain-traces \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_detection_zeroshot_lessons/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/heldout_detection_explain_traces/
```

**QC**: `src/heldout_detection_explain_traces_qc.py` (`heldout-detection-explain-traces-qc`)
— no model calls, pure report generator over Step 2's own checkpoint,
same "read it yourself before trusting it" purpose Step −1's own
ablation and `detection_qc_plot.py` already serve for the stages
before this one. Renders one PNG per sampled item: the image with
predicted (solid lime) and ground-truth (dashed red) boxes overlaid
(`bbox_viz.plot_predictions`, reused directly), plus a text panel with
`error_type`/`category`/IoU, both boxes' raw coordinates, the question
asked, and the model's own written explanation. Round-robin sampled
across every `error_type` actually present in the checkpoint (not
hardcoded to a fixed list) so `false_positive` — the most numerous
category, from `detection-multiclass`'s own deliberate over-generation
— doesn't dominate `--n_samples` purely by base rate.

```bash
uv run python main.py heldout-detection-explain-traces-qc \
    --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --n_samples 20
```

## Step 3 — summarize into skill_memory_detection.md (map-reduce, built)

Built: `src/heldout_detection_summarize.py`.

Same map-reduce mechanic as `heldout_summarize.py`, chunked by Step 2's
own `error_type` (false_positive/missed/loose_match/worked — no
separate trace_type dimension to cross it with, since error_type
already encodes worked-vs-failure directly) instead of disease/pest/
deficiency/healthy x worked/didnt_work. Same hard rules carried over
(learned the expensive way in Stage 3, not re-litigated here): no
"common objects to know" bare-label reference section, no bullet whose
entire content is a bare category word, write generalizable guidance
not per-image facts. New rule specific to detection: a bullet must
describe a VISUAL/SPATIAL pattern (shape, boundary, occlusion,
clustering), not a coordinate number — "boxes around clustered fruit
tend to need splitting when a stem or leaf visually separates two
individuals" is useful; "move the box 15px left" is not (even if that
number happened to be true for one example, it's exactly the
per-image-memorization failure mode Stage 3's own edit prompt already
forbids).

```bash
uv run python main.py heldout-detection-summarize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_md outputs_final/heldout_detection_summarize/skill_memory_detection.md
```

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

`--feedback_prior_round` (Stage 3's own A/B flag — see
`docs/stage3_context_memory.md`'s Step 3b: feeds each edit call the
immediately-prior round's changelog + the TRAIN scores that edit
produced) carries over for free once the above is wired up, since it's
part of the same shared `heldout_skill_optimize.py` machinery, not a
classification-specific addition. Same command shape, pointed at a
detection skill file once one exists:

```bash
uv run python main.py heldout-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --init_skill_path <detection skill file, TBD> \
    --max_rounds 10 \
    --feedback_prior_round \
    --output_dir outputs_final/heldout_detection_skill_optimize_feedback/
```

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
