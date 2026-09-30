# Paper 2, Stage 4 — Detection Verifier

[← Back to README](../README.md)

Status: **generation step built, smoke test not yet run on real hardware.
Verifier (scoring/filtering) step designed but not built.** This stage is
the spatial-task analog of [Stage 3](stage3_classification_verifier.md)'s
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
per box should now contain a specific stage/condition (e.g. `"bud"`,
`"open flower"`, `"dead flower"`) instead of just repeating the dataset's
one class name — if every box still comes back labeled identically to the
dataset's class name, the model isn't actually differentiating stages and
the prompt needs another pass.

## Planned: per-box verifier (not built yet)

Once multi-class generation is confirmed working, the next piece is
scoring *confidence* per predicted box rather than trusting every box
equally:

1. **Crop-and-regrade.** For each predicted box, crop it out of the
   source image (with some padding) and show *just that crop* to the
   verifier model, independently of the other boxes and the full-image
   context — a genuine second opinion, not the same generation call
   grading its own output.
2. **Rubric, not a single trajectory score.** Grade each crop against a
   small rubric (e.g. "is this really an instance of the object," "is the
   box reasonably tight," "is the stage label plausible given what's
   visible") and sum to a confidence in `[0, 1]` — same summed-rubric
   mechanic Stage 3 uses, just applied per-box instead of per-trajectory,
   so there's no single `verifier_score` per image, only one confidence
   per predicted box.
3. **Filter before reporting.** Drop boxes below some confidence
   threshold from the final prediction set before computing precision/
   recall/F1 — the multi-class generation step over-generates by design
   (every stage, every instance), and this is what prunes it back down to
   a trustworthy final set.
4. **Scoring stays class-agnostic.** IoU-match the filtered box set
   against ground truth exactly as Step 0 already does
   (`bbox_metrics.match_boxes`) — no new scoring machinery needed there,
   the verifier only changes which boxes make it into that matching step.

Open questions for when this gets built: the crop-padding amount (too
tight loses context the model needs to judge plausibility; too loose
re-includes neighboring instances), and how many rubric criteria are
worth the extra GPU call per box before diminishing returns.
