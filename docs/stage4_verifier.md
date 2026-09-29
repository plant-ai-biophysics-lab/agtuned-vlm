# Stage 4 — Verifier-CoT (single-round K-candidate + PPT tournament)

[← Back to README](../README.md)

Per image: the model answers a structured set of diagnostic criteria K
times (temperature-sampled), a verifier ranks that one pool of K candidates
via a full Probabilistic Pivot Tournament, and the reported prediction is
simply `argmax(win_rate)` over that single tournament pass. No incumbent
carried across rounds, no persistent per-answer score, no stopping rule —
there is only one round.

## Why single-round

This is a deliberate simplification (2026-08-23) from an earlier multi-round
design that carried a persistent per-answer net score (wins − losses) across
rounds, with a leader-margin gate and a consecutive-defends lock to decide
when to stop. The full derivation of that design (a Gaussian-belief update,
later simplified to a plain net score) is preserved for reference at
[`docs/legacy/verifier_score_belief_update.md`](legacy/verifier_score_belief_update.md)
— it no longer describes running code.

On a full 9,104-image run, the multi-round machinery did **not** reliably
improve on stopping after round 1 once judge-evaluation noise was controlled
for: round 1's own judge_f1_macro (0.3556) was statistically indistinguishable
from the noise-corrected final-round score (0.3547), and genuine
mechanism-caused answer swaps netted **-8 images out of 9,104** — a
legitimate negative finding, not a bug. Meanwhile round 1 alone (structured
candidate generation + PPT tournament selection) substantially beats the
naive single-shot Lower Bound baseline (~0.36 vs ~0.18-0.20, roughly 2x) —
that comparison is the real, well-evidenced contribution this pipeline now
exists to produce, and reflects what a workshop-scale paper can credibly
claim: the structured prompts + tournament selection are the contribution,
not open-ended multi-round self-improvement.

Given that, the repo was simplified to match: the round loop, persistent
clusters, leader-margin gate, stopping-rule, and same-answer identity judge
were all removed, leaving a clean single-round K-candidate + PPT-tournament
verifier — for both classification (`verifier_cot.py`) and object detection
(`detection_verifier.py`).

## Conditions compared

- **Verifier** (`src/verifier_cot.py`) — our contribution: *structured
  candidate generation + tournament selection*, not the ranking mechanism
  itself. K candidates are generated per image from a structured
  fill-in-the-blank template that forces the model to answer specific
  diagnostic criteria BEFORE stating its `<answer>`. That pool of K is
  ranked once by the FULL, unmodified Probabilistic Pivot Tournament from
  the PPT paper (arXiv:2607.05391 Section 3.2, via
  `llm_verifier.pivot_tournament`'s `ring_cycle`/`select_pivots`/
  `pivot_round_pairs` — same library `src/verifier_lmv.py` uses): a ring
  pass over the whole pool, then a pivot round on the top-k. Every
  comparison in both passes is scored via the paper's own dual-trajectory
  logprob mechanism (`src/common/ppt_scoring.py`, shared with
  `verifier_lmv.py` — same scoring, different aggregation). The reported
  prediction is the pool member with the highest aggregate win-rate.
- **PPT-Direct-N10** (`src/verifier_lmv.py`, pre-existing, run separately)
  — the full Probabilistic Pivot Tournament from *LLM-as-a-Verifier*
  (arXiv:2607.05391): a static 10-candidate pool ranked via the real
  `llm_verifier.pivot_tournament` library. Its scoring mechanism was
  corrected as part of this work — from a single free-generation call
  (empirically found to collapse onto the scale's first letter ~96% of the
  time) to the same two-pass teacher-forced/prefill extraction the verifier
  uses.

## Step 1 — run verifier-cot and verifier-lmv

`--verifier_type ppt` is required explicitly — `verifier-cot`'s default is
now `simple` (Paper 2 Stage 3, a different, cheaper single-trajectory
verifier, see [docs/stage3_verifier_grade.md](stage3_verifier_grade.md)),
not this stage's K-candidate + PPT-tournament mechanism.

```bash
uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it --verifier_type ppt
uv run python src/verifier_lmv.py --model google/gemma-4-26B-A4B-it \
    --output_dir outputs_final/verifier_lmv/
```

`verifier-lmv` isn't wired into `main.py` (pre-existing convention) — run it
directly. Its `--output_dir` is flat (one model's worth of output per run,
no `<model>/` subfolder), unlike everything else in this pipeline — use a
distinct `--output_dir` per model if comparing more than one.

Writes to `<output_dir>/<safe_model_name>/` (verifier-cot) or
`<output_dir>/` (verifier-lmv):
- `ckpt_<safe_model_name>.csv` (verifier-cot) — one row per image
  (CKPT_FIELDS-shaped, same as every other classification stage), plus a
  small set of verifier-specific extras: `pool_size`/`n_comparisons`/
  `top1_win_rate` describe the tournament; `verifier_feedback` is the
  pairwise analysis text that first mentioned the winning candidate;
  `prompt_tokens`/`completion_tokens`/`elapsed_s`/`avg_time_per_sample_s`
  are compute cost.
- `verifier_log.jsonl` (verifier-cot) — one record per individual verifier
  call: every candidate generated (`call_type="candidate"`, raw text +
  extracted `<answer>` + whether the tag was found) and every pairwise duel
  (`"tournament_pair"`) — full prompt + response + parsed result, same
  convention as `run_judge_phase`'s `judge_log.jsonl`.
- `compute_summary.json` (verifier-cot) — LLM calls/prompts made this run,
  plus cumulative token totals across the whole checkpoint.
- `ckpt_judged.csv` (verifier-lmv) — one row per (image, gen_mode, N-slice);
  `ppt_reward` is the tournament's win-rate-derived score.
- `metrics.json` (verifier-cot) — F1 / judge accuracy, same shape as every
  other stage.

Both are resumable per-model checkpoints (`--rerun` to force a clean
restart) with the same `--judge_model` / `--skip_judge` / `--rejudge` flags
as every other stage.

Key knobs (see `--help` for the full list; defaults below match
`src/config/verifier_config.yaml`, not the CLI's own bare defaults —
`--verifier_config` is loaded automatically and any flag passed on the
command line overrides it):
- `--k_candidates` (default 4) — candidates generated per image; the
  tournament pool size.
- `--k_pivots` (default 2) — pivot-set size (k) for the PPT tournament that
  ranks the pool, matches `llm_verifier.pivot_tournament.DEFAULT_PIVOTS`.
- `--temperature` (default 0.8) — sampling temperature for the K
  candidates; needs to be > 0 for `--k_candidates > 1` to actually differ.
- `--seed` (default 42) — seeds the k-candidate sampling and the pairwise
  duel's A/B-slot randomization/analysis sampling, so repeated runs over
  the same images/config are reproducible.
- `--match_images_from` — deterministic image selection guaranteed to
  overlap with an existing `vision_baseline` run (auto-detected per
  `--model` by default), instead of depending on independent sampling RNG.
- `--ppt_pivots` / `--n_evals` / `--n_criteria` (verifier-lmv) — pivot-set
  size (k), repeated-evaluation count (K), criteria decomposition (C).

## Step 2 — compare against the lower/upper bound and each other

Two ways to compare, depending on scope:

**Quick, apples-to-apples, no full-corpus dependency** —
`verifier-summary` (`src/verifier_summary.py`, no GPU/model needed) filters
Lower Bound / Self-CoT / Manual-CoT / Upper Bound down to the EXACT
`(agml_dataset, class_label, image_name)` triples the verifier run actually
scored (verifier-cot commonly runs at `--n_samples 10` while the bounds run
at 50/class — comparing at dataset level would silently mismatch sample
counts), and writes everything into the verifier run's own `--verifier_dir`
rather than touching `outputs_final/cot_compare/`:

```bash
uv run python main.py verifier-summary \
    --verifier_dir outputs_final/verifier_cot/ \
    --model google/gemma-4-26B-A4B-it
```

Produces `verifier_summary.png` / `verifier_summary_by_task.png` — bar chart
vs. bounds + CoT conditions, 95% CI, comparing the Verifier's single-round
result against Lower Bound / Self-CoT / Manual-CoT / Upper Bound.

**Full-corpus, paper-facing** — once `vision-baseline` has run for the
model at its full sample size, `cot-compare` (Stage 3's own comparison
script — see [stage3_cot.md](stage3_cot.md) Step 3) picks up verifier-cot
and verifier-lmv output automatically:

```bash
uv run python main.py cot-compare \
    --verifier_lmv_ckpt outputs_final/verifier_lmv/ckpt_judged.csv
```

Adds, into `<output_dir>/` (Gemma-4-26B-A4B-it only — the only model with
validated CoT/verifier data, same scoping as every other Stage 3/4 chart):
- `verifier_gemma_f1_by_task.png` — Verifier / PPT-Direct-N10 raw judged F1,
  per task + overall, lower/upper bound as dotted reference lines (same
  pairing as `cot_gemma_f1_by_task.png`).
- `verifier_all_conditions_by_task.png` — the delta view (vs. the lower
  bound) of the same conditions.

Both are no-ops (skipped, no file written) until at least one of
verifier-cot/verifier-lmv has actually produced output for that model.

## Object detection (Stage 5d)

`src/detection_verifier.py` is the same single-round design applied to
bounding-box localization instead of free-text classification — see
[stage5_object_detection.md](stage5_object_detection.md) for the detection
pipeline as a whole.
