# Stage 4 — Verifier belief tracking

[← Back to README](../README.md)

Full derivation, formulas, and worked example:
[`verifier_score_belief_update.md`](verifier_score_belief_update.md).

One evolving thread per image: each round the model answers, a verifier
scores/compares candidates, and a belief `(mu, sigma^2)` about the current
best answer's quality gets updated — instead of a fixed CoT step count
([stage3_cot.md](stage3_cot.md)) or a one-shot N-candidate pool, rounds
continue only until the belief is confident enough to stop (3–10 rounds,
adaptive).

Three conditions are directly comparable at an equal round/candidate
budget — the first two share the same round/candidate-generation infra:

- **Verifier-Belief** (`src/verifier_cot.py --baseline belief`) — our
  contribution: *when to stop*, not the ranking mechanism. Each round's
  pool is the previous round's incumbent (if any) + K new candidates —
  the incumbent is a POOL MEMBER, not a fixed always-compared anchor. The
  pool is ranked each round by the FULL, unmodified Probabilistic Pivot
  Tournament from the PPT paper (arXiv:2607.05391 Section 3.2, via
  `llm_verifier.pivot_tournament`'s `ring_cycle`/`select_pivots`/
  `pivot_round_pairs` — same library `src/verifier_lmv.py` uses): a ring
  pass over the whole pool, then a pivot round on the top-k. Every
  comparison in both passes is scored via the paper's own dual-trajectory
  logprob mechanism (`src/common/ppt_scoring.py`, shared with the
  full-tournament baseline below — same scoring, different aggregation),
  slot randomized per pair on top of ring-pass's own bias cancellation,
  aggregated into Bradley-Terry win-mass to rank the pool. Exactly ONE
  persistent-belief update happens per round, using only the WINNER/LOSER
  IDENTITY from that ranking — the belief update needs an identity, not a
  raw score, so it doesn't require a specific pair to have been directly
  compared under PPT's O(nk) budget. That identity is keyed off the
  ANSWER, not the pool slot: each candidate's generation also carries an
  open-ended `<answer>...</answer>` tag (no closed-set vocabulary), and
  when a new candidate outranks the incumbent's own slot, a small
  text-only judge call (`same_answer_prompt`, just the two extracted
  `<answer>` strings) decides whether it's a genuine answer change (belief
  resets) or a reworded restatement of the same conclusion (belief carries
  forward onto the new text) — so a rephrase alone doesn't wipe out
  accumulated confidence. `incumbent_survived` tracks whether the ANSWER
  stays the same round over round.
- **Verifier-Argmax** (`src/verifier_cot.py --baseline argmax`) — ablation:
  same round budget, no pairwise/belief math, just keeps whichever
  candidate scored highest so far.
- **PPT-Direct-N10** (`src/verifier_lmv.py`, pre-existing, run separately)
  — the full Probabilistic Pivot Tournament from *LLM-as-a-Verifier*
  (arXiv:2607.05391): a static 10-candidate pool ranked via the real
  `llm_verifier.pivot_tournament` library (ring-pass + pivot scheduling),
  no adaptive stopping. Its scoring mechanism was corrected as part of this
  work — see the design doc's Section 15 note — from a single free-generation
  call (empirically found to collapse onto the scale's first letter ~96% of
  the time) to a two-pass teacher-forced/prefill extraction.

## Step 1 — run verifier-cot (both baselines) and verifier-lmv

```bash
uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it
uv run python main.py verifier-cot --model google/gemma-4-26B-A4B-it --baseline argmax
uv run python src/verifier_lmv.py --model google/gemma-4-26B-A4B-it \
    --output_dir outputs_final/verifier_lmv/
```

`verifier-lmv` isn't wired into `main.py` (pre-existing convention) — run it
directly. Its `--output_dir` is flat (one model's worth of output per run,
no `<model>/` subfolder), unlike everything else in this pipeline — use a
distinct `--output_dir` per model if comparing more than one.

Writes to `<output_dir>/<safe_model_name>/` (verifier-cot) or `<output_dir>/`
(verifier-lmv):
- `ckpt_<baseline>_<safe_model_name>.csv` (verifier-cot) — one row per
  (image, round); `is_final_round=1` marks the row to score. `mu`/`sigma2`/
  `stop_confidence`/`confidence_bin` track the belief at that round;
  `pool_size`/`n_comparisons`/`top1_win_rate`/`incumbent_survived`/
  `verifier_answer` describe that round's tournament (belief mode only —
  `verifier_answer` is the winning pool member's extracted `<answer>` tag);
  `reasoning` holds the running transcript; `prompt_tokens`/
  `completion_tokens`/`elapsed_s` are that round's compute cost.
- `verifier_log_<baseline>.jsonl` (verifier-cot) — one record per
  individual verifier call: every candidate generated each round
  (`call_type="candidate"`, raw text + extracted `<answer>` + whether the
  tag was found — for checking the tagging mechanism's own reliability),
  every ring-pass/pivot-round pair (`"tournament_pair"`), every
  class-identity check (`"same_answer"`, belief mode only), every
  independent candidate score (`"score"`, argmax mode) — full prompt +
  response + parsed result, same convention as `run_judge_phase`'s
  `judge_log.jsonl`.
- `compute_summary_<baseline>.json` (verifier-cot) — LLM calls/prompts made
  this run, plus cumulative token totals across the whole checkpoint.
- `ckpt_judged.csv` (verifier-lmv) — one row per (image, gen_mode, N-slice);
  `ppt_reward` is the tournament's win-rate-derived score.
- `metrics_<baseline>.json` (verifier-cot) — F1 / judge accuracy, same
  shape as every other stage.
- `belief_by_round_<baseline>.png` (verifier-cot only) — mean belief per
  round with a 95% bootstrap CI, one line per task type + overall (the
  belief-tracking analogue of `cot-yaml-step-report`'s `step_f1_overall.png`).

Both are resumable per-model checkpoints (`--rerun` to force a clean
restart) with the same `--judge_model` / `--skip_judge` / `--rejudge` flags
as every other stage.

Key knobs (see `--help` for the full list, defaults match
[the design doc](verifier_score_belief_update.md)'s Section 13):
- `--k_candidates` (verifier-cot) / `--n_max` (verifier-lmv) — candidates
  generated per round / pool size.
- `--min_round` / `--max_round` (verifier-cot) — stopping bounds, default 3–10.
- `--beta` (verifier-cot) — Bradley-Terry comparison noise.
- `--high_conf` / `--med_conf` (verifier-cot) — confidence-bin thresholds,
  default 0.85 / 0.6.
- `--k_pivots` (verifier-cot) — pivot-set size (k) for the full PPT
  tournament that ranks each round's pool, `--baseline belief` only,
  default matches `llm_verifier.pivot_tournament.DEFAULT_PIVOTS`.
- `--seed` (verifier-cot) — seeds the per-pair A/B slot randomization in
  `--baseline belief`'s pairwise scoring (on top of ring-pass's own bias
  cancellation), default 42.
- `--ppt_pivots` / `--n_evals` / `--n_criteria` (verifier-lmv) — pivot-set
  size (k), repeated-evaluation count (K), criteria decomposition (C).

## Step 2 — compare against the lower/upper bound and each other

Once `vision-baseline` has run for the model, `cot-compare` (Stage 3's own
comparison script — see [stage3_cot.md](stage3_cot.md) Step 3) picks these
up automatically:

```bash
uv run python main.py cot-compare \
    --verifier_lmv_ckpt outputs_final/verifier_lmv/ckpt_judged.csv
```

Adds, into `<output_dir>/` (Gemma-4-26B-A4B-it only — the only model with
validated CoT/verifier data, same scoping as every other Stage 3/4 chart):
- `verifier_gemma_f1_by_task.png` — Verifier-Belief / Verifier-Argmax /
  PPT-Direct-N10 raw judged F1, per task + overall, lower/upper bound as
  dotted reference lines (same pairing as `cot_gemma_f1_by_task.png`).
- `verifier_all_conditions_by_task.png` — the delta view (vs. the lower
  bound) of the same three conditions.

Both are no-ops (skipped, no file written) until at least one of
verifier-cot/verifier-lmv has actually produced output for that model.
