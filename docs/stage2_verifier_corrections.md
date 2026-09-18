# Paper 2, Stage 2 — Verifier Corrections

[← Back to README](../README.md)

Paper 1's Appendix C (Figures 8–9) found the PPT verifier's own
fine-grained-reward letter score (`src/common/ppt_scoring.py`, A–T scale) is
both **under-varied** (92%/80% of scored letters land in A–C, regardless of
model) and **inverted** in the specific sense that matters for filtering:
retained-set accuracy *declines*, not rises, as the win-rate acceptance
threshold is raised. Stage 1 (`stage1_judge_improvements.md`) fixed judge
leniency/consistency on plain classification predictions; Stage 2 targets
this separate, upstream problem in the verifier (`verifier-cot`) itself —
diagnose before prescribing, same principle as Stage 1's Phase 1/2.

- **Step 1 — baseline verifier run**: generate the *current, unmodified*
  verifier setup's results, on the same images already used for the Stage 1
  vision-baseline checkpoint, so this is a direct before/after comparison
  point once a fix is applied later.
- **Step 2 — diagnose**: `verifier-score-analysis` re-derives Paper 1's
  Figure 8/9 checks (image-level `top1_win_rate` vs. correctness, pair-level
  raw score vs. local correctness, letter-scale histogram) directly from
  that run's own checkpoint + `verifier_log.jsonl` — no GPU, no new
  inference, pure post-hoc analysis.
- **Step 3 — granularity** (this stage's own addition to `verifier_cot.py`'s
  logging, see below): three concrete, testable hypotheses for *why* the
  score is under-varied/inverted, each backed by a new logged field instead
  of guessed at.
- **Step 4 — fix** (not yet started): once Step 3 points at a specific
  cause, apply the targeted change (rescoring-prompt rewrite, sum-not-max
  aggregation, a different scale, etc.) and re-run Steps 1–2 as the
  after-comparison.

## Step 1 — generate the baseline

Reuse the exact same images already sampled for the Stage 1 vision-baseline
checkpoint (`--match_images_from`, `verifier_cot.py`'s own auto-detection
already defaults to this when a matching `vision_baseline` checkpoint
exists for the model):

```bash
uv run python main.py verifier-cot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --k_candidates 3 --k_pivots 2 --temperature 1.0
```

Output: `outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/`
(`ckpt_<model>.csv` + `verifier_log.jsonl`, per `verifier_cot.py`'s module
docstring).

## Step 2 — diagnose

```bash
uv run python main.py verifier-score-analysis \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

Writes `image_level_threshold.png`, `pair_level_threshold.png`,
`letter_distribution.png`, and `score_analysis_summary.json` under
`<verifier_dir>/score_analysis/`. See `src/verifier_score_analysis.py`'s
module docstring for what each chart means and why there are two threshold
charts (image-level vs. pair-level — they can disagree, and disagreement is
itself informative about whether the problem is in the scoring call or the
tournament aggregation).

## Step 3 — granularity: three hypotheses, three logged signals

Added to `verifier_cot.py` / `src/common/ppt_scoring.py` /
`src/common/ppt_orchestration.py` this stage, all purely additive
(read-only diagnostic fields; nothing here changes what score a trajectory
actually receives):

1. **Max-not-sum on letter space-variants.**
   `llm_verifier.fine_grained_reward.extract_score` folds a letter's plain
   and leading-space spellings (`"C"` / `" C"`) together by taking the
   **max** of their two probabilities, not the sum (`ppt_scoring.py`'s own
   `extract_tag_score`, unchanged, still the sole authority on the actual
   score). `letter_probability_diagnostics()` (new) computes both
   aggregations side by side from the same top-20 logprobs and is logged as
   `letter_probs_{a,b}_{max,sum}` per `tournament_pair` entry.
   `verifier-score-analysis`'s `max_vs_sum_mass_gap` reports the mean
   probability mass discarded by the max convention — near 0 rules this
   hypothesis out; a consistently large gap makes it a real contributor to
   the Figure 9 collapse.
2. **Length bias.** Each scored trajectory's word count is logged
   (`trajectory_{a,b}_words`). `length_vs_score` reports the correlation
   between a trajectory's length and its own score — a positive correlation
   here means the verifier is (at least partly) rewarding verbosity, not
   correctness.
3. **Confidence-language carryover.** The prior CoT-verifier findings
   ([[project-cot-verifier-findings]] memory) found confidence language
   ("clearly," "characteristic of") *anti-correlated* with correctness in
   `self_cot`/`cot_yaml` runs. `_CONFIDENCE_MARKERS_RE` counts the same
   family of markers in the PPT verifier's own pairwise comparative-analysis
   text (Pass 1, before the forced-letter continuation) and logs
   `n_confidence_markers` per pair. `confidence_language_vs_winner_correct`
   correlates that count against whether the pair's higher-scored ("winner")
   candidate was actually locally correct — replicating the same check in
   this different pipeline instead of assuming it carries over.

All three show up under `score_analysis_summary.json`'s
`pair_level.granularity` key once a run with the updated logging exists;
older `verifier_log.jsonl` files (predating this stage) are read fine —
`verifier-score-analysis` just omits `granularity` for them.

## Step 4 — fix (not started)

Deferred until Step 3's results on a real run point at a specific cause.
