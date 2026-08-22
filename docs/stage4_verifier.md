# Stage 4 — Verifier-CoT (round-based net-score tracking)

[← Back to README](../README.md)

Full derivation of the original Gaussian-belief design:
[`verifier_score_belief_update.md`](verifier_score_belief_update.md) — the
PPT tournament mechanics and stopping-rule motivation there still apply, but
the score representation itself changed (see below); read this page first
for what's actually running today.

One evolving thread per image: each round the model answers a structured set
of diagnostic criteria, a verifier ranks that round's candidates via a full
PPT tournament, and each distinct answer's persistent **net score**
(wins − losses) gets updated — instead of a fixed CoT step count
([stage3_cot.md](stage3_cot.md)) or a one-shot N-candidate pool, rounds
continue until a leader has genuinely stabilized (or the round cap is hit).

## What changed from the original design

The original design (linked above) tracked a Gaussian belief `(mu, sigma^2)`
per incumbent answer, updated via Bradley-Terry moment matching, with a
confidence-threshold early stop. Two rounds of simplification since then,
both driven by what real runs actually showed:

1. **Per-trajectory net score, not belief.** Every DISTINCT answer this
   image has ever proposed keeps its own persistent `wins`/`losses` counters
   (a "cluster" — `state[i]["clusters"]`) rather than one incumbent belief
   that resets on any answer change. The reported answer each round is
   whichever cluster has the highest net score (`wins - losses`) overall,
   gated by `leader_margin` (a challenger must beat the current leader by at
   least this many net-score points to actually take over — real duel
   margins run narrow, so without this an already-correct leader could get
   displaced by one noisy near-tie). This replaced the Gaussian
   moment-matching update: on real runs, the evidence-scaling machinery it
   depended on was pinned near its weakest-evidence setting almost every
   round, and only ~2 of 12 observed leader flips were ever caused by a
   genuine margin-beaten swap — the machinery wasn't earning its complexity.
2. **Stopping is now `lock_after_n_defends` or `max_round` only** — no
   confidence-threshold early stop. The old
   `Phi((mu-mu_1)/sqrt(...)) >= high_conf` trigger fired on evidence as
   thin as a single round's net-score edge, before a defend streak ever had
   a chance to form — exactly the noisy-snapshot failure mode the stopping
   design exists to prevent. `stop_confidence`/`confidence_bin` are still
   computed and logged every round (and still drive `--high_conf`/
   `--med_conf`) but are informational only now. If `max_round` is hit
   without ever locking, the reported answer is the cluster with the
   **longest defend streak seen so far**, not whatever's raw-leading that
   instant.

`mu`/`sigma2` are still written to every CSV row and still drive
`belief_by_round_belief.png` — kept as a synthetic
`Belief(mu=net_score, sigma2=1/(1+wins+losses))` purely so the existing
checkpoint schema and `verifier_summary.py`'s charts didn't need to change;
`mu` there means the leader's net score, not a Gaussian mean.

There is no more `--baseline argmax` ablation — **`--max_round 1` through
this same pipeline IS the argmax-equivalent baseline**: with no incumbent
and nothing to persist, the round's own tournament win-rate argmax is the
entire result.

## Conditions compared

- **Verifier-CoT** (`src/verifier_cot.py`) — our contribution: *when to
  stop and how to track competing answers across rounds*, not the ranking
  mechanism. Each round's pool is the previous round's incumbent (if any) +
  K new candidates, generated from a structured fill-in-the-blank template
  that forces the model to answer specific diagnostic criteria BEFORE
  stating its `<answer>` — the incumbent is a POOL MEMBER, not a fixed
  always-compared anchor. The pool is ranked each round by the FULL,
  unmodified Probabilistic Pivot Tournament from the PPT paper
  (arXiv:2607.05391 Section 3.2, via `llm_verifier.pivot_tournament`'s
  `ring_cycle`/`select_pivots`/`pivot_round_pairs` — same library
  `src/verifier_lmv.py` uses): a ring pass over the whole pool, then a
  pivot round on the top-k. Every comparison in both passes is scored via
  the paper's own dual-trajectory logprob mechanism
  (`src/common/ppt_scoring.py`, shared with `verifier_lmv.py` — same
  scoring, different aggregation). Winner/loser identity from that ranking
  resolves into a persistent-score cluster (see above) — identity is keyed
  off the extracted `<answer>` text, via a lightweight text-only judge
  (`same_answer_prompt`) that decides whether a new top candidate is a
  genuine answer change or a reworded restatement of an existing cluster.
- **PPT-Direct-N10** (`src/verifier_lmv.py`, pre-existing, run separately)
  — the full Probabilistic Pivot Tournament from *LLM-as-a-Verifier*
  (arXiv:2607.05391): a static 10-candidate pool ranked via the real
  `llm_verifier.pivot_tournament` library, no adaptive stopping, no
  round-over-round score persistence. Its scoring mechanism was corrected
  as part of this work — see the design doc's Section 15 note — from a
  single free-generation call (empirically found to collapse onto the
  scale's first letter ~96% of the time) to the same two-pass
  teacher-forced/prefill extraction Verifier-CoT uses.

## Step 1 — run verifier-cot and verifier-lmv

```bash
uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it
uv run python src/verifier_lmv.py --model google/gemma-4-26B-A4B-it \
    --output_dir outputs_final/verifier_lmv/
```

`verifier-lmv` isn't wired into `main.py` (pre-existing convention) — run it
directly. Its `--output_dir` is flat (one model's worth of output per run,
no `<model>/` subfolder), unlike everything else in this pipeline — use a
distinct `--output_dir` per model if comparing more than one.

Writes to `<output_dir>/<safe_model_name>/` (verifier-cot) or
`<output_dir>/` (verifier-lmv):
- `ckpt_belief_<safe_model_name>.csv` (verifier-cot) — one row per
  (image, round); `is_final_round=1` marks the row to score. `mu`/`sigma2`/
  `stop_confidence`/`confidence_bin` are the synthetic net-score view (see
  above); `pool_size`/`n_comparisons`/`top1_win_rate`/`incumbent_survived`/
  `pairwise_winner`/`n_clusters`/`verifier_answer` describe that round's
  tournament and cluster state (`pairwise_winner` = `seed`/`defended`/
  `replaced` — whether the LEADER changed this round, not whether the raw
  top-1 beat pool slot 0); `reasoning` holds the running transcript;
  `prompt_tokens`/`completion_tokens`/`elapsed_s` are that round's compute
  cost.
- `verifier_log_belief.jsonl` (verifier-cot) — one record per individual
  verifier call: every candidate generated each round (`call_type=
  "candidate"`, raw text + extracted `<answer>` + whether the tag was
  found), every pairwise duel (`"pairwise"`), every same-answer identity
  check (`"same_answer"`), and every round's full cluster snapshot
  (`"belief_clusters"` — every distinct answer this image has proposed so
  far, with its wins/losses/net and full membership) — full prompt +
  response + parsed result, same convention as `run_judge_phase`'s
  `judge_log.jsonl`.
- `compute_summary_belief.json` (verifier-cot) — LLM calls/prompts made
  this run, plus cumulative token totals across the whole checkpoint.
- `ckpt_judged.csv` (verifier-lmv) — one row per (image, gen_mode, N-slice);
  `ppt_reward` is the tournament's win-rate-derived score.
- `metrics_belief.json` (verifier-cot) — F1 / judge accuracy on
  `is_final_round=1` rows, same shape as every other stage.
- `belief_by_round_belief.png` (verifier-cot only) — mean net score (`mu`)
  per round with a 95% bootstrap CI, one line per task type + overall (the
  round-tracking analogue of `cot-yaml-step-report`'s
  `step_f1_overall.png`).

Both are resumable per-model checkpoints (`--rerun` to force a clean
restart) with the same `--judge_model` / `--skip_judge` / `--rejudge` flags
as every other stage. `--judge_scope first_and_final` restricts judging to
`round==1` and `is_final_round==1` rows only (~5x cheaper on long runs) —
everything else stays unjudged (blank `judged_correct`), which
`verifier_summary.py`'s charts already account for.

Key knobs (see `--help` for the full list; defaults below match
`src/config/verifier_config.yaml`, not the CLI's own bare defaults —
`--verifier_config` is loaded automatically and any flag passed on the
command line overrides it):
- `--k_candidates` (default 4) — candidates generated per round.
- `--k_pivots` (default 2) — pivot-set size (k) for the PPT tournament that
  ranks each round's pool, matches `llm_verifier.pivot_tournament.
  DEFAULT_PIVOTS`.
- `--min_round` / `--max_round` (default 3 / 6) — `max_round 1` is the
  argmax-equivalent baseline (see above); `min_round` gates how early
  `lock_after_n_defends` can fire.
- `--leader_margin` (default 2) — minimum net-score advantage a challenger
  cluster needs over the current leader to actually take over as leader.
- `--lock_after_n_defends` (default 3) — stop early once the leader has
  "defended" this many consecutive rounds; 0 disables (only `max_round`
  decides, and the best-streak timeout fallback always applies).
- `--beta` (default 0.5) — Bradley-Terry comparison noise, used by
  `stop_probability`'s (informational-only) confidence value.
- `--high_conf` / `--med_conf` (default 0.6 / 0.3) — confidence-bin
  thresholds, informational only (see "What changed" above).
- `--seed` (default 42) — seeds incumbent-slot randomization and every
  generation `SamplingParams`, so repeated runs over the same images/config
  are reproducible.
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

Produces `verifier_summary.png` / `verifier_summary_by_task.png` (bar chart
vs. bounds + CoT conditions, 95% CI) plus a set of round-level diagnostic
charts (`diag_*.png` — confidence calibration, round-vs-accuracy,
round1-vs-final, stop-round distribution, regression/recovery, answer
diversity, and more) — see the module docstring for the full list and what
question each one answers.

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
- `verifier_gemma_f1_by_task.png` — Verifier-CoT / PPT-Direct-N10 raw judged
  F1, per task + overall, lower/upper bound as dotted reference lines (same
  pairing as `cot_gemma_f1_by_task.png`).
- `verifier_all_conditions_by_task.png` — the delta view (vs. the lower
  bound) of the same conditions.

Both are no-ops (skipped, no file written) until at least one of
verifier-cot/verifier-lmv has actually produced output for that model.
