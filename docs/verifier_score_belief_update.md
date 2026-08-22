# Verifier Score: Belief Update via Bradley–Terry Comparisons

*Design reference for the Stage 4 verifier work. Source for `verifier_score_belief_update.md.pdf`.*

Scope note: this version deliberately leaves out the rephrasing step and the alignment-score
track (Track A). It covers only the verifier — an LLM judge that compares two candidates
and says which one is better — and how we turn that comparison into an updated belief
about each candidate's quality, plus how we decide when to stop comparing. Rephrasing and
alignment can be layered back in once this piece is solid.

## 1. The setup, in plain language

We have a pool of candidate answers. We want to know which one is best. We don't ask the
verifier "how good is this candidate, on a scale of 1–10?" — instead we ask it a pairwise
question: "between candidate i and candidate j, which is better?" This is usually an easier
and more reliable question for an LLM judge to answer than an absolute score.

The problem: a single yes/no comparison doesn't hand us a clean number for "how good is
candidate i." It only tells us candidate i beat candidate j this time. We need a principled way
to turn a sequence of these pairwise outcomes into a running estimate of each candidate's
quality, complete with a sense of how confident we are.

That's what this document walks through:

1. How we represent "belief about a candidate's quality" as a number plus an uncertainty.
2. The probability model the verifier's comparisons are assumed to follow (Bradley–Terry).
3. Why updating our belief from a comparison outcome is mathematically awkward, and the trick (moment matching) used to make it tractable.
4. The full update rule, term by term.
5. A worked numerical example.
6. How to decide "we're confident enough, stop comparing."
7. How we generate the candidates/pairs being compared.
8. How this relates to the Probabilistic Pivot Tournament (PPT) in *LLM-as-a-Verifier* (arXiv:2607.05391).
9. Our contribution: an adaptive stopping rule, and what it buys us under a noisy or non-monotonic verifier.

## 2. Representing a belief as a bell curve

For each candidate i, we track two numbers:

- **$\mu_i$** ("mu") — our current best guess at candidate i's quality.
- **$\sigma_i^2$** ("sigma squared") — how uncertain we are about that guess. Small $\sigma_i^2$ means we're fairly sure; large $\sigma_i^2$ means we're still unsure.

Together, $(\mu_i, \sigma_i^2)$ describe a bell curve (a Gaussian / normal distribution) centered at $\mu_i$, with
a width controlled by $\sigma_i^2$. Think of it as a smeared-out guess: "I think this candidate's quality
is around $\mu_i$, but it could plausibly be a bit higher or lower, and $\sigma_i^2$ tells you how much room
I'm leaving for that."

Every candidate starts with some initial belief (a "prior") — for instance, $\mu_i = 0$, $\sigma_i^2 = 1$ for
everyone, meaning "no information yet, wide open uncertainty." Each verifier comparison
then narrows and shifts this bell curve.

## 3. The verifier's comparison model (Bradley–Terry)

When the verifier compares candidates i and j, we assume its judgment follows the
Bradley–Terry model, a standard way to model "which of two things is better" outcomes:

$$p_{ij} = \sigma\big(R(x,\tau_i) - R(x,\tau_j)\big)$$

Reading this:

- $R(x, \tau_i)$ is the *true* (unknown) quality score of candidate i, and likewise for j.
- The difference $R(x, \tau_i) - R(x, \tau_j)$ is how much better i is than j, according to the verifier's internal sense of quality.
- $\sigma(\cdot)$ here is the logistic sigmoid function, $\sigma(t) = \dfrac{1}{1+e^{-t}}$ — it squashes any real number into a probability between 0 and 1.
- $p_{ij}$ is the probability the verifier says "i is better."

Intuition: if i is much better than j, the difference is a big positive number, the sigmoid
pushes it close to 1, and the verifier almost always picks i. If they're equally good, the
difference is 0, sigmoid gives 0.5, and it's a coin flip.

We don't know the true R values — that's exactly what $\mu_i$ is estimating. So each verifier
comparison is a noisy peek at the gap between two unknown numbers, and we want to use
that peek to sharpen our belief.

## 4. Why the update isn't a simple formula

If the verifier gave us a direct number (like "candidate i scores 7.2"), updating a Gaussian
belief would be easy — you'd just blend the old belief with the new number, weighted by
how confident each one is. That's a standard, exact, closed-form operation.

But the verifier doesn't give us a number — it gives us a win/loss, filtered through a sigmoid.
Mathematically, combining a Gaussian belief with a sigmoid-shaped observation does **not**
produce another Gaussian. The exact answer is some lumpy, asymmetric shape that's hard
to work with, store, or carry forward into the next round.

The fix used here is called **moment matching** (also known as *assumed density filtering*, and
it's the same core idea behind game-rating systems like Microsoft's TrueSkill). Instead of
tracking the true, awkward posterior shape exactly, we:

1. Compute the exact mean and variance the true update would produce.
2. Find the Gaussian bell curve that has that same mean and variance.
3. Use that Gaussian as our new belief, discarding the exact (non-Gaussian) shape.

This is an approximation, not an exact update — but it's a good one, and critically it keeps
our belief always representable as a simple (μ, σ²) pair, so the process can repeat indefinitely
without the representation getting more complicated round after round.

One more subtlety worth flagging: the win/loss probability above uses the logistic sigmoid
(Bradley–Terry), but the moment-matching correction below is derived assuming
Gaussian-shaped comparison noise (this is sometimes called the *probit* version of the
same idea). Mixing the two is a standard, well-tested approximation in this literature — it's
not internally inconsistent so much as "close enough to work well in practice," because
logistic and Gaussian curves have very similar shapes once you match their spread.

## 5. The full update rule

Say candidates i and j were just compared, and i won. Define:

$$c = \sqrt{\sigma_i^2 + \sigma_j^2 + 2\beta^2} \qquad\qquad z = \frac{\mu_i - \mu_j}{c}$$

- **c** is the *combined uncertainty* of the matchup: how much spread there is between the two candidates' beliefs, plus $\beta^2$, a fixed "how noisy is a single verifier judgment" parameter (a hyperparameter you choose — bigger $\beta$ means you trust any one comparison less).
- **z** is a *normalized gap*: how far apart the two candidates' current mean estimates are, measured in units of combined uncertainty. If z is large and positive, i was already believed to be clearly better than j, so i winning isn't very surprising. If z is near 0, the two were considered roughly equal, so the win is more informative.

Now define two correction terms, built from the standard normal density $N(z)$ (the
bell-curve height at z) and cumulative distribution $\Phi(z)$ (the probability mass to the left of z):

$$v(z) = \frac{N(z)}{\Phi(z)} \qquad\qquad w(z) = v(z)\big(v(z) + z\big)$$

These come from the mean and variance of a *truncated* normal distribution — i.e., "a normal
distribution, but we've thrown away the part that's inconsistent with the observed
outcome." v(z) tells us how much to shift the mean; w(z) tells us how much to shrink the
variance. Both are always positive, and both get **larger** (more effect) when z is small — i.e.,
when the outcome was more surprising given our current beliefs.

The update for the winner, i:

$$\mu_i^{\text{new}} = \mu_i + \frac{\sigma_i^2}{c}\,v(z) \qquad\qquad
(\sigma_i^2)^{\text{new}} = \sigma_i^2\left[1 - \frac{\sigma_i^2}{c^2}\,w(z)\right]$$

Reading this in words:

- The new mean is the old mean, **nudged upward** by an amount proportional to how uncertain we were ($\sigma_i^2$) and how surprising the win was ($v(z)$). A candidate we were already confident about barely moves; a candidate we were unsure about moves more.
- The new variance is the old variance, **shrunk** by a factor that depends on the same surprise term ($w(z)$). Winning a comparison always makes us more certain (variance never increases from this step), and it shrinks more when the win was surprising.

By the symmetric logic (and this is the "swap i<->j" note in the win/loss formula), the loser, j,
gets pushed the opposite direction:

$$\mu_j^{\text{new}} = \mu_j - \frac{\sigma_j^2}{c}\,v(z) \qquad\qquad
(\sigma_j^2)^{\text{new}} = \sigma_j^2\left[1 - \frac{\sigma_j^2}{c^2}\,w(z)\right]$$

Same shrinkage logic, but the mean moves down instead of up. Both candidates get more
certain after a comparison; the winner's mean rises and the loser's mean falls, by amounts
proportional to how uncertain each one individually was going in.

## 6. Worked numerical example

Suppose two brand-new candidates start with identical, wide-open beliefs:

$$\mu_1 = 0,\ \sigma_1^2 = 1 \qquad\qquad \mu_2 = 0,\ \sigma_2^2 = 1 \qquad\qquad \beta = 0.5$$

($\beta = 0.5$ is the chosen noise level for a single verifier judgment.)

**Step 1 — compute c and z.**

$$c = \sqrt{1 + 1 + 2(0.5)^2} = \sqrt{2.5} \approx 1.581 \qquad\qquad
z = \frac{\mu_1 - \mu_2}{c} = \frac{0-0}{1.581} = 0$$

Makes sense — the two candidates start out believed equal, so z = 0.

**Step 2 — compute the correction terms at z = 0.**

$$N(0) \approx 0.3989 \qquad\qquad \Phi(0) = 0.5$$

($N(0)$ is the peak height of the standard bell curve; $\Phi(0) = 0.5$ means half the mass is to
the left of the center.)

$$v(0) = \frac{0.3989}{0.5} \approx 0.7979 \qquad\qquad
w(0) = 0.7979(0.7979 + 0) \approx 0.6367$$

**Step 3 — suppose the verifier says candidate 1 wins. Update candidate 1:**

$$\mu_1^{\text{new}} = 0 + \frac{1}{1.581}(0.7979) \approx 0.505 \qquad\qquad
(\sigma_1^2)^{\text{new}} = 1\left[1 - \frac{1}{2.5}(0.6367)\right] = 1 - 0.2547 \approx 0.745$$

Candidate 1's estimated quality moved up from 0 to about 0.505, and its uncertainty shrank
from 1.0 to about 0.745 (equivalently, $\sigma_1$ went from 1.0 to about 0.863).

**Step 4 — update candidate 2 (the loser) symmetrically:**

$$\mu_2^{\text{new}} = 0 - \frac{1}{1.581}(0.7979) \approx -0.505 \qquad\qquad
(\sigma_2^2)^{\text{new}} = 1\left[1 - \frac{1}{2.5}(0.6367)\right] \approx 0.745$$

Candidate 2's estimate dropped to about −0.505, with the same shrunk uncertainty. After
one comparison, the two beliefs have separated symmetrically around 0, and both are more
confident than before.

If you ran a *second* comparison between the same two candidates, you'd recompute c and z
using the new μ's and σ²'s — the gap (μ₁ − μ₂) is now about 1.01 instead of 0, so z would no
longer be 0, and a repeat win for candidate 1 would move things by a smaller amount
(because it's now less surprising).

## 7. Deciding when to stop comparing

Once you have updated beliefs for all candidates, you'll want to know: "am I confident
enough in the current best one to just output it?" This reuses the exact same building block
as the update step, applied once at the top level instead of pairwise.

Let i* be the current best candidate (highest $\mu$) and "runner-up" the second-best. Because
the difference of two independent Gaussians is itself Gaussian, we can compute:

$$P(i^* \text{ truly better than runner-up}) = \Phi\!\left(\frac{\mu_{i^*} - \mu_{\text{runner-up}}}
{\sqrt{\sigma_{i^*}^2 + \sigma_{\text{runner-up}}^2 + 2\beta^2}}\right)$$

This is structurally identical to the z-score computation from Step 1 above, just fed into $\Phi$
directly rather than into v and w. If this probability clears a chosen threshold (say, 0.95),
you stop and output candidate i*. Otherwise, you run another verifier comparison and
repeat the update.

## 8. Where this sits, if you want to study the underlying theory

- The Bradley–Terry probability formula in Section 3 (`p_ij = σ(R_i − R_j)`) is the same sigmoid-of-a-utility-difference formula from classical preference-learning theory — the same structure DPO relies on for language model alignment.
- The moment-matching trick in Section 5 (using N and Φ to correct a Gaussian belief after a win/loss outcome) is the same core technique behind Bayesian skill-rating systems (TrueSkill, and its ancestor Glicko) — it's worth reading about "assumed density filtering" or "expectation propagation" if you want the general theory behind why matching moments gives a good Gaussian approximation.
- The stopping rule in Section 7 is structurally a best-arm identification / dueling bandit stopping criterion — you're deciding when you've gathered enough pairwise evidence to commit to a winner, which is a well-studied problem in the sequential decision-making literature.

When we bring back the alignment-score track and the rephrasing step, the alignment
score will plug in as a *direct* (non-pairwise) observation — which turns out to be the easy
case, since a direct Gaussian observation combines with a Gaussian belief exactly, no
moment-matching approximation required. That's a good next module to add once this one
feels solid.

## 9. Generating the candidates and pairs

Two separate questions hide inside "how do we generate pairs": (a) how do we get a *pool*
of diverse candidate statements/answers to compare in the first place, and (b) once we have
a pool, which two do we actually feed the verifier at each round?

**(a) Producing diverse candidates.** Sampling temperature (e.g. T=1.0, same prompt,
different draws) is the baseline way to get diversity — it's what `self_cot`'s free-form
description turn already does implicitly across images, just not yet across repeated draws
of the *same* image. Rephrasing the input question is a complementary, not competing,
source of diversity: our own [[project_cot_verifier_findings]] work already found that
presupposition-loaded phrasing anchors the model toward a specific (sometimes wrong)
diagnosis. Two different phrasings of the same question can pull out genuinely different
reasoning paths, not just different wording of the same one — sampling temperature alone
won't surface that. Recommendation: use both — temperature for cheap within-prompt
diversity, rephrasing specifically for the cases where we suspect a single phrasing is
anchoring every sample toward the same (possibly wrong) answer.

**(b) Scheduling which pairs to compare.** This is a separate, budget question, addressed
in Section 10 below.

## 10. Relationship to the Probabilistic Pivot Tournament (PPT)

*Kwok et al., "LLM-as-a-Verifier: A General-Purpose Verification Framework," arXiv:2607.05391.*

That paper's ranking algorithm (Section 3.2, Fig. 6) — the Probabilistic Pivot Tournament
(PPT) — answers question (b) above: given N candidates and a fixed comparison budget,
which pairs do you actually spend verifier calls on? It does **not** address when to stop; it
assumes a fixed budget chosen in advance (via k, the pivot-set size) and a single-shot
schedule, not an adaptive stopping decision. Their scoring is also different from ours: they
compute a *continuous* reward `R(x,τ)` per candidate by taking the expectation over
scoring-token logprobs (Eq. 3.1 — score granularity × repeated evaluation × criteria
decomposition), then feed that continuous value into the same Bradley–Terry sigmoid we
use (their Eq. 3.2 is literally our Section 3 formula). The two approaches share the
Bradley–Terry layer but diverge above it:

| | PPT (arXiv:2607.05391) | This doc |
|---|---|---|
| Per-candidate score | Continuous `R(x,τ)`, from logprob-weighted expectation over score tokens | Gaussian belief `(μ, σ²)`, updated round-by-round |
| Pair schedule | Fixed, budgeted upfront: ring pass (all N, cancels positional bias) -> pick top-k pivots -> every pair involving a pivot. Total comparisons = `N + k(N−k) + C(k,2)`, i.e. O(Nk) | Same schedule, reused as-is (decided below) — run fresh each round over that round's pool |
| Aggregation | Win mass `wᵢ` / comparison count `cᵢ` per candidate, static, computed once after all scheduled comparisons finish | Sequential moment-matching update after every single comparison |
| Stopping rule | **None** — budget is fixed by choice of k before comparisons start | Confidence threshold on `P(i* beats runner-up)` (Section 7) — adaptive, comparisons continue only until confident |

**Where they complement each other.** PPT's *ring-pass* step (a random Hamiltonian cycle
so every candidate appears once as "A" and once as "B") is a clean, cheap way to cancel the
verifier's positional bias, and its *pivot selection* (spend the remaining budget only on
pairs involving the current empirical leaders) is a sensible way to avoid wasting comparisons
on candidates that are already clearly weak. Both of those are schedule-level choices,
independent of how the score itself is aggregated. **Decision: reuse PPT's ring-pass +
pivot-selection schedule verbatim** (via the vendored `llm_verifier.pivot_tournament`
library — `ring_cycle`, `select_pivots`, `pivot_round_pairs`, `bradley_terry`, unmodified)
to pick *which* pairs to feed our sequential Bradley–Terry/Gaussian updater, run fresh each
round over that round's pool, rather than round-robin (O(N²), and more importantly not what
the paper does) or PPT's own static win-rate aggregation (which has no natural stopping
point) on its own. That combination — PPT's pairing discipline, our belief tracking and
stopping rule — is the actual implementation (Section 13); the only thing this doc's own
contribution changes about the paper's algorithm is layering the stopping rule on top of it,
not the tournament mechanism itself.

## 11. Our contribution: when do we stop?

PPT commits to a fixed number of comparisons decided in advance (by k) and never asks
"do we already know enough to stop early." Section 7 of this doc is the part PPT doesn't
have: instead of a fixed budget, we track a live confidence estimate — `P(i* beats
runner-up)` — after every comparison, and stop as soon as it clears a threshold. Two
practical benefits over a fixed budget:

- **Cheaper on easy cases.** If one candidate is clearly best, z separates fast (Section 6's worked example shows a single comparison already moves the gap from 0 to ~1) and the threshold clears in very few rounds — no need to spend PPT's full `O(Nk)` budget on a case where a handful of comparisons already settle it.
- **Honest on hard cases.** If candidates are genuinely close (or the verifier itself is inconsistent — see Section 12), the confidence estimate simply keeps failing to clear the threshold. A fixed-budget scheme like PPT still returns *some* winner (`argmax wᵢ/cᵢ`) once the budget runs out, even if that winner isn't meaningfully better than the runner-up. Our stopping rule can instead report "not confident after N rounds," which is itself useful signal, not a forced, possibly arbitrary commitment.

## 12. Does this stabilize scoring under a noisy verifier?

A natural worry: what if the verifier's judgments bounce around — round to round, it
sometimes prefers i, sometimes j, with no real signal underneath? Does the stopping rule
protect us from mistaking noise for a real winner? Yes, by construction, and it's worth being
precise about *why*.

Each round's correction terms `v(z)`/`w(z)` are largest when z is near 0 (Section 5) — i.e. when
the two candidates are still considered close. If the verifier's judgments are genuinely
inconsistent (close to a coin flip every round, no real underlying gap), z keeps getting pulled
back toward 0 after every update instead of steadily growing, because wins and losses
roughly cancel out over successive rounds. `P(i* beats runner-up)` then stays hovering near
0.5 and never reliably clears the stopping threshold — the process correctly refuses to
commit, rather than eventually converging on an arbitrary "winner" the way an average-then-
threshold-after-N-rounds scheme would. β (the per-comparison noise hyperparameter) is the
knob that controls how much any single flip-flop is allowed to move belief in the first place —
larger β makes the whole system more skeptical of any one verdict, which is the right lever
to raise if we find the verifier is noisier than we assumed.

The second half of the question — "should the score steadily increase after each reasoning
step?" — is worth resisting rather than assuming. Nothing in this framework requires (or
even encourages) monotonic improvement, and that's deliberate: we already have direct
empirical evidence from [[project_cot_verifier_findings]] and the Stage 3 CoT step-analysis
work (`cot_yaml_step_report.py`'s `swap_step_f1` / anchor-recovery charts) that per-step F1
on the `disease_pest` chain **declines** with chain depth, and that anchor rate rising with step
position is a proximity-to-endpoint artifact, not evidence of increasing reliability — recovery
rate stays flat and low throughout. If we baked in an assumption that later reasoning steps
should score higher, we'd be fighting our own data. Instead, treating each step's statement as
just another candidate in the tournament, and letting pairwise verifier comparisons decide
the ranking empirically, is the correct framing here: if belief keeps failing to separate
across steps (stopping condition never clears, even after many rounds), *that itself* is
diagnostic evidence that additional reasoning isn't reliably improving the answer for that
task type — which is exactly what the existing swap-order data already suggests for
`disease_pest`, and this framework would let us test it directly and quantitatively rather than
inferring it indirectly from F1 curves.

## 13. The concrete round-based algorithm

This section pins down the exact mechanics decided on after the Section 9–12 discussion —
this is what gets implemented, not a hypothetical. One evolving thread per image, not a
one-shot tournament over a static pool.

**Belief vs. verifier score — these are two different things, but the score still shapes the
update.** The verifier score (`R_a`, `R_b` from the logit extraction below) is per-comparison
and ephemeral — it exists to decide who wins *that one* pairwise comparison, and is discarded
once that's decided. The belief `(μ, σ²)` is persistent — it survives across rounds and drives
the stopping rule. A win/loss outcome doesn't average scores into it; it moves the belief by an
amount that depends on how *surprising* the outcome was given the belief's own existing
uncertainty (`z`/`v(z)`/`w(z)`, Section 5). This part is unchanged. What changed (see "Evidence-
scaled beta" below): early versions of this design fed a flat, fixed `β` into that formula
regardless of how decisive or how absolutely good the winning score actually was — so a
"best of two mediocre candidates" win and a genuinely decisive win produced *identical* belief
shifts, since `pairwise_update` never saw `R_a`/`R_b` at all, only categorical win/loss. That's
now fixed: `β` is computed per-comparison from the winner's and loser's raw scores before
`pairwise_update` runs, so the raw score value *does* now shape the belief arithmetic — just
indirectly, through how much the comparison is trusted, not by being averaged in directly.

**Why every comparison is pairwise, and why the incumbent enters the pool instead of being a
fixed anchor.** Absolute 1–20 scoring (still used by the `--baseline argmax` ablation) asks
the verifier to judge a candidate with nothing to calibrate against — LLM judges are known to
be poorly calibrated in that mode. Pairwise comparison is more reliable precisely because it
hands the verifier a concrete reference to reason against, which is the actual motivation for
pairwise LLM-judging generally and the PPT paper's own starting premise. Every comparison
that touches the tracked belief is scored via the exact same PPT-paper mechanism as the
full-tournament baseline (arXiv:2607.05391 Section 3.2 — dual-trajectory `<score_A>`/
`<score_B>` logprob extraction, `src/common/ppt_scoring.py`, shared with
`src/verifier_lmv.py`) — the two conditions differ in what happens *after* a comparison is
scored (a persistent belief + adaptive stopping here, vs. static win-mass aggregation over a
fixed schedule there — Section 10/11), not in how a single comparison is scored.

**Persistence is keyed off the answer, not the trajectory text or the pool slot.** The
tournament ranks *trajectories* — full reasoning + conclusion, since that's what the verifier
needs to judge quality. But what the belief is actually supposed to track is confidence in
*an answer*, and a trajectory's exact wording can change round to round (a rephrase, more or
less hedging, a different level of detail) without the underlying conclusion changing at all.
Treating "did the exact same pool slot win again" as the test for "did the answer survive"
conflates the two: a freshly generated candidate that reaches the *same* conclusion as the
incumbent, just phrased differently, would look identical to a genuine answer flip under a
purely positional test — and a positional test would needlessly reset the persistent belief
to a fresh prior every time that happens, discarding real accumulated confidence over
cosmetic rewording. The fix: every candidate's free generation also carries an open-ended
`<answer>...</answer>` tag (the model restates its conclusion inline, no closed-set
vocabulary — real-world answers aren't a fixed label set, unlike `resolve_class`'s
ground-truth-vocabulary string matching, which is deliberately *not* reused here for exactly
that reason). When this round's tournament top-1 is the incumbent's own pool slot, "same
answer" is trivially true, no extra work needed. When top-1 is a new candidate, a lightweight,
**text-only** judge call (just the two extracted `<answer>` strings — no image, no full
trajectory, since the tournament has already judged trajectory quality and this call only
asks "same conclusion or not") decides whether it's a genuine answer change or a reworded
restatement, via `same_answer_prompt` (`src/config/verifier_prompts.yaml`) — the only YAML
prompt `--baseline belief` actually uses (its trajectory-ranking duels still use the paper's
own fixed Python template, per the note above). This call only fires in that one branch — round
1 (no incumbent) and the incumbent's own slot winning again both skip it as trivial — so it
adds at most one small, cheap call per round, not one per pool member.

**Flow chart:**

```
ROUND k (k=1 has no incumbent; k>=2 does)
  generate K candidates (image + question, temperature sampling — see
  Section 9(a)); --rephrase (off by default) ALSO adds one candidate per
  model-generated alternate wording of the question (generated once per
  group, before round 1 — see "Rephrase step" below), so pool size is
  K + rephrase_count when enabled, not just K
        |
  pool = [incumbent from round k-1, if k>=2] + [K new candidates]
  (the incumbent is a POOL MEMBER, not a fixed always-compared anchor)
  pool size n = K, or K+1 from round 2 onward
        |
  FULL PPT tournament over the pool (arXiv:2607.05391 Section 3.2, via
  llm_verifier.pivot_tournament -- unmodified, same code src/verifier_lmv.py
  uses for its own static baseline):
    1. ring pass -- random Hamiltonian cycle, n adjacent pairs, every
       pool member appears once as "A" and once as "B" (cancels
       positional bias by construction)
    2. pivot selection -- top-k by ring-pass win rate (k = --k_pivots,
       default DEFAULT_PIVOTS=2)
    3. pivot round -- every non-pivot vs. each pivot, plus pivot vs.
       pivot; total comparisons this round = n + k(n-k) + C(k,2), i.e.
       O(nk), NOT n(n-1)/2
  every comparison in both passes is scored via the same PPT
  dual-trajectory mechanism (src/common/ppt_scoring.py) -- slot (A/B)
  additionally randomized per pair as a second safeguard on top of
  ring-pass's own cancellation; two-pass prefill -> (score_a, score_b)
  in [0,1] per pair -> Bradley-Terry win-mass w_i/c_i accumulated per
  pool member across BOTH passes
        |
  top1 = argmax(win_rate);  runner_up = 2nd by win_rate
        |
  k==1 (no incumbent)?
    winner=top1, loser=runner_up                            ["seed"]
        |
  k>=2 and top1 == incumbent's pool slot (trivially the same answer)?
    yes -> winner=incumbent, loser=runner_up                ["defended"]
        |
  k>=2 and top1 != incumbent's pool slot (a NEW candidate topped the
  ranking) -- is its <answer> tag the SAME underlying answer as the
  incumbent's? (same_answer_prompt, text-only, just the two extracted
  <answer> strings -- one small call, only in this branch)
    same     -> winner=top1, loser=runner_up                 ["defended"]
                (persistent belief carries forward onto the new,
                differently-worded winning text -- not a reset)
    not same -> winner=top1, loser=incumbent                 ["replaced"]
                (genuine answer change -- fresh prior for the winner,
                incumbent's persistent belief discarded)
        |
  Section 5 moment-matching update needs WHICH SIDE WON (pairwise_update
  (winner: Belief, loser: Belief, beta) still takes belief objects, not a
  score) plus the winner's/loser's OWN raw scores this round, averaged
  across whichever comparisons they were part of, to compute an
  evidence-scaled beta (see below) before calling pairwise_update -- no
  lookup of one SPECIFIC pair's score is needed (PPT's O(nk) schedule
  doesn't guarantee (incumbent, top1) or (incumbent, runner_up) was ever
  directly compared), just each side's own average; the tournament's own
  aggregate win-rate ranking decides who won, the same-answer check (or
  triviality) decides who's PERSISTENT
        |
  Section 5 update: winner's belief (persistent (mu, sigma^2) if
  "defended", else fresh (0,1)) vs loser's belief (persistent if
  "replaced"'s incumbent, else fresh) -> new (mu, sigma^2);
  C_best <- winner's text; answer <- winner's extracted <answer> tag
        |
  (sigma^2 always shrinks — a round always adds evidence)
        |
  k==1?  freeze the resulting (mu, sigma^2) as (mu_1, sigma_1^2),
         the stopping-rule anchor
        |
        v
  round >= 3?
    P_k = Phi( (mu - mu_1) / sqrt(sigma^2 + sigma_1^2 + 2*beta^2) )
        |
    P_k >= 0.85 ("high confidence")?  -- no, and round < 10 -> loop back
        |
       yes  --  or round == 10 regardless
        v
      STOP -- report C_best + confidence bin
        high  (P_k >= 0.85)
        medium (0.6 <= P_k < 0.85)
        low    (P_k < 0.6)
```

**State carried across rounds**, per image:

```
C_best        — text of the current incumbent (reasoning + predicted class)
answer        — the incumbent's extracted <answer> tag (open-ended, no closed-set vocabulary)
(μ, σ²)       — running belief in C_best's quality
(μ₁, σ₁²)     — frozen snapshot of the belief after round 1 (the stopping-rule anchor)
round         — current round index k
```

Top-1 only for now — `C_best`/`(μ,σ²)` track a single incumbent, not a pool of several
lineages. A top-k *pool of persistent lineages* (as opposed to this round's single-use
tournament pool, which already exists) — carrying several candidate lineages forward across
rounds instead of one, to reduce anchoring risk — is a deliberate future extension, not built
now.

**Every round, uniformly (no round-1 special case in the belief-update logic).** Build the
pool: the previous round's incumbent (if `round >= 2`) plus the K freshly generated
candidates, pool size `n`. Run the **full, unmodified PPT tournament** over that pool —
`llm_verifier.pivot_tournament.ring_cycle` / `select_pivots` / `pivot_round_pairs` /
`bradley_terry`, the exact same functions `src/verifier_lmv.py` already uses for its static
baseline, imported directly rather than reimplemented: (1) a ring pass over all `n` members
(cancels positional bias by construction — every member is "A" once and "B" once across the
cycle); (2) select the top-`k` members by ring-pass win rate as pivots (`k` = `--k_pivots`,
default `DEFAULT_PIVOTS=2`); (3) score every non-pivot-vs-pivot pair plus every pivot-vs-pivot
pair. Total comparisons per round = `n + k(n-k) + C(k,2)` — O(nk), not the quadratic
`n(n-1)/2` a round-robin would cost — this is the paper's own budget-efficient schedule,
reused as-is rather than a from-scratch pairing scheme (Section 10's decision). Every
comparison in both passes is scored via `src/common/ppt_scoring.py`'s two-pass prefill
mechanism (pass 1 free-generates a comparative analysis; pass 2 forces `<score_A>`/
`<score_B>`'s very next token under a grammar mask and reads the logprob-expectation score for
each, paper Eq. 3.1). Each pair's own slot (A/B) is additionally randomized — a coin flip —
on top of ring-pass's own cancellation, as a second, cheap safeguard. Aggregate into
Bradley-Terry win-mass per pool member (`P(i≻j) = σ(R_i − R_j)`, accumulated as `w_i += P`,
`c_i += 1` for every pair `i` appears in, across both the ring pass and the pivot round) and
take `top1 = argmax(win_i/count_i)`.

Exactly ONE persistent-belief update happens per round — not one per pair — using the
**identity** of a winner and a loser (`pairwise_update(winner: Belief, loser: Belief, beta)`,
Section 5, still takes belief objects, not scores) plus each side's own average raw score this
round, fed through evidence_scaled_beta (below) to compute `beta` before the call. Nothing
about the schedule's *coverage* matters here — PPT's O(nk) budget doesn't guarantee the
specific pair the update cares about was ever directly compared, and it doesn't need to be,
since each side's average is taken across whichever comparisons it happened to be part of. The tournament's own
aggregate win-rate ranking (`top1`, `runner_up` — both guaranteed defined for every pool
member, since the ring pass alone touches every index at least once) decides *which trajectory
wins*; whether that trajectory's belief is treated as PERSISTENT is a separate question,
decided by answer identity rather than pool position (see the class-identity note above):
- **`round == 1`** (no incumbent yet): winner = top-1, loser = runner-up — both beliefs start
  from the flat prior `(0, 1)`, status `"seed"`.
- **`round >= 2`, top-1 is the incumbent's own pool slot**: winner = incumbent, loser =
  runner-up, status `"defended"` — trivially the same answer, no judge call needed.
- **`round >= 2`, top-1 is a new candidate, same_answer_prompt says SAME**: winner = top-1,
  loser = runner-up, status `"defended"` — a genuine reworded restatement of the incumbent's
  own conclusion, so the WINNER (the new, better-worded text) inherits the incumbent's
  *persistent* `(μ, σ²)` rather than starting fresh; the incumbent's belief isn't discarded,
  it moves onto the new text.
- **`round >= 2`, top-1 is a new candidate, same_answer_prompt says DIFFERENT**: winner =
  top-1, loser = incumbent, status `"replaced"` — a genuine answer change: the new answer
  starts from a fresh `(0, 1)` prior, immediately updated by beating the incumbent's actual
  accumulated `(μ, σ²)` (harder to displace a well-established incumbent than a fresh one).

In every case, the belief-update's winner keeps a carried-over `(μ, σ²)` only when it is, by
answer identity, the incumbent's own lineage continuing (`"defended"`, whether via the literal
pool slot or a same-answer reworded win); otherwise it starts fresh at `(0, 1)`. Same rule for
the loser (whose updated belief is computed and then discarded — only the winner's belief
persists into the next round; a `"replaced"` loser's belief was the incumbent's real
accumulated one, not a fresh one, which is why displacing an established incumbent is
harder than displacing a brand-new one). `C_best` becomes the winner's text; `answer` becomes
the winner's extracted `<answer>` tag; `(μ, σ²)` becomes its updated belief.
`incumbent_survived` (stored per round) now records whether the ANSWER survived (status ==
`"defended"`), not whether the literal pool slot did — the direct, literal answer to "does the
conclusion stay across steps," immune to cosmetic rewording. The first analysis text the
tournament produced that mentions the winner's pool index (guaranteed to exist, since
ring-pass touches every index) doubles as the feedback fed back to the VLM (the PPT template
has no separate feedback field) for round k+1's re-ask.

**Stopping**, checked starting at round 3, hard cap at round 10:

$$P_k = \Phi\!\left(\frac{\mu - \mu_1}{\sqrt{\sigma^2 + \sigma_1^2 + 2\beta^2}}\right)$$

```
P_k ≥ 0.85            -> "high confidence"   -> stop
0.6 ≤ P_k < 0.85       -> "medium confidence"
P_k < 0.6              -> "low confidence"
```

If round 10 is reached without clearing 0.85, stop anyway and report whichever bin `P_k`
falls into — the bin is always reported, never withheld pending more rounds. `β` is the
comparison-noise hyperparameter; 0.85/0.6 are starting defaults, easy to move once real
score distributions are in hand.

### 13.1 Evidence-scaled beta — making the update sensitive to score quality

**The problem, found on a real run.** Bradley-Terry's win probability,
$P(a \succ b) = \sigma(R_a - R_b)$, is *translation-invariant* to the absolute level of
$R_a$/$R_b$ — a winner scoring 0.40 against a 0.05 loser (both weak) produces the same win
probability as a 0.95-vs-0.60 winner (both strong), if the gap is similar. Combined with
Section 5's `pairwise_update(winner, loser, beta)` never even receiving `R_a`/`R_b` (only the
categorical winner/loser identity, plus a *flat, fixed* `β`), this meant a "best of several
weak candidates" win produced an **identical** belief shift to a genuinely decisive win. On a
real 160-image run this showed up concretely: `stop_confidence` at the final round averaged
0.800 for images whose final answer was judged *correct* and 0.779 for images judged
*incorrect* — a gap of 0.02, close to noise. The belief signal wasn't tracking correctness
because it wasn't sensitive to how good the winning score actually was, only to *whether* it
won.

**Why the fix scales beta up, not down.** The natural first instinct — divide the flat `β`
down for a decisive win, up for a weak one — turns out to be numerically almost inert, because
`pairwise_update`'s moment-matching formula is close to flat/saturated for any `β` below
roughly 1.0: `c` (Section 5) grows with `β` in the denominator, but the resulting smaller `z`
pushes `v(z)`/`w(z)` up almost enough to cancel that growth, so the belief shift barely moves
across the whole `β ∈ [0.05, 0.5]` range. It only becomes meaningfully sensitive to `β` once
`β` climbs toward roughly 2–10. So instead of scaling `β` *down* from a 0.5 baseline (still
inside the flat zone), the fix blends `β` *up* toward a much larger `max_beta` as evidence
weakens — the only direction with real sensitivity.

**The formula.** Let $\bar R_w$, $\bar R_\ell$ be the winner's and loser's own raw scores this
round, each averaged across whichever comparisons that pool member happened to be part of
(the O(nk) tournament schedule doesn't guarantee the winner and loser were ever directly
compared to each other — see the flow chart above):

$$
\text{floor\_factor} = \min\!\left(1,\ \frac{\bar R_w}{\tau}\right)
\qquad
\text{margin\_factor} = \min\!\left(1,\ \frac{\max(0,\ \bar R_w - \bar R_\ell)}{m}\right)
\qquad
\text{confidence} = \text{floor\_factor} \times \text{margin\_factor}
$$

$$
\beta_{\text{eff}} = \beta + (\beta_{\max} - \beta)\,(1 - \text{confidence})
$$

- **floor_factor** — is the winner *itself* any good, independent of what it beat? $\tau$
  (`--floor_threshold`, default 0.5) is the "good enough on its own" bar. A winner scoring 0.4
  only clears 80% of that bar even if it demolished its opponent.
- **margin_factor** — how decisive was the win? $m$ (`--margin_ref`, default 0.3) is the gap
  that counts as "decisive enough" to saturate this factor at 1.
- **confidence = floor_factor × margin_factor** — both must be satisfied. A big margin between
  two *weak* candidates (e.g. 0.40 vs. 0.05 — margin 0.35, well past $m$) still gets capped by
  a low floor_factor (0.8), so "best of two bad options" can't fake full confidence just by
  having a wide gap.
- **$\beta_{\text{eff}} = \beta$** when confidence = 1 (today's unchanged default, near-maximal
  update strength — the flat zone means shrinking further wouldn't help). **$\beta_{\text{eff}}
  = \beta_{\max}$** (`--max_beta`, default 4.0) when confidence = 0 — verified to produce a
  visibly smaller shift, not a cosmetic one: on a representative belief state, a decisive
  good-vs-good win shifts $\mu$ by 0.123, while a same-margin bad-vs-bad win only shifts it by
  0.113; a narrow win (margin under $m$ regardless of absolute quality) drops to about 0.062 —
  roughly half the unscaled shift.

Implementation: `evidence_scaled_beta` (`src/common/belief_update.py`), called from
`_process_round_belief` (`src/verifier_cot.py`) right before `pairwise_update`, using
`score_sum[i][k]`/`score_n[i][k]` accumulated alongside the existing win-rate tracking during
the round's ring-pass/pivot comparisons. Config: `beta`, `floor_threshold`, `margin_ref`,
`max_beta` in `src/config/verifier_config.yaml`.

### 13.2 Rephrase step — question diversity, not answer diversity

**The problem.** Temperature sampling alone doesn't guarantee the K candidates a round
generates actually say different things. On a real run (K=8, temperature=1.0), **35.6%** of
image-rounds produced only 1–2 distinct `<answer>` tags among the 8 candidates — the model
often collapses onto (near-)the same conclusion regardless of resampling.

**Why rephrase the question, not the answer.** An earlier design considered asking the model
to explicitly "reconsider from a fresh angle" on the *same* question — i.e. rephrasing the
answer-generation instruction while leaving the question itself untouched. Rejected: a model
asked to introspect on its own just-stated conclusion, given the exact same prompt, tends to
produce shallow hedging ("it's most likely X, but could also be Y") rather than genuinely
different reasoning. Varying the **question's own wording** instead changes what the model
actually anchors on — a different literal prompt can surface different vocabulary and framing,
which is closer to how prompt-ensembling is normally used to get real diversity, not just
paraphrased confidence.

**Mechanism.** Once per (dataset, class) group, before round 1 — not per round, not per image,
since the classification question is identical across the whole group — one text-only call
(`vprompts["question_rephrase_prompt"]`, `src/config/verifier_prompts.yaml`) is sampled
`rephrase_count` times at `rephrase_temperature` to produce that many alternate wordings of the
question, meaning-preserving by instruction. Every round thereafter, each rephrased-question
variant additionally generates exactly one candidate (one separate `generate()` call per
variant, not `n` samples of one prompt — each candidate must come from a genuinely different
prompt), merged into the same tournament pool as the base K candidates. Pool size becomes
`K + rephrase_count` (plus the incumbent from round 2 on), not just `K` — `effective_k_
candidates` is threaded into `_process_round_belief` so the ring-pass/pivot tournament sees the
correct pool size.

Off by default (`--rephrase`, config `rephrase: false`); `--rephrase_count` (default 4) and
`--rephrase_temperature` (default 1.3, deliberately hotter than the base K candidates'
`--temperature` default of 1.0 — the whole point is more diversity, not the same prompt
resampled at the same temperature) are both in `src/config/verifier_config.yaml`.

## 14. What gets stored

One checkpoint file, one row per (image, round) — merged into a single file rather than the
two originally sketched, since a `reasoning` column carrying the round's transcript and the
numeric belief columns serve both purposes without duplication:

**`ckpt_belief_<model>.csv`** (or `ckpt_argmax_<model>.csv` for the ablation) —
`agml_dataset, image_name, task_type, round, mu, sigma2, stop_confidence, confidence_bin,
pool_size, n_comparisons, top1_win_rate, incumbent_survived, verifier_answer,
verifier_feedback, pairwise_winner, is_final_round, prediction, reasoning, ...` plus the
standard classification/judge columns. `is_final_round=1` marks the row to score for final
metrics; `pool_size`/`n_comparisons` (belief mode only) record this round's tournament size
(`n + k(n-k) + C(k,2)` PPT ring-pass + pivot-round comparisons, O(nk) — not a round-robin
`n(n-1)/2`); `top1_win_rate` is the tournament's own confidence in its
pick (distinct from the persistent belief `mu` — see the "belief vs. verifier score" note
above); `verifier_answer` (belief mode only) is the winning pool member's extracted
`<answer>` tag — the same signal `incumbent_survived` is now keyed off of, exposed directly
for spot-checking; `incumbent_survived` (1/0, blank at round 1) is now the direct, literal
"did the ANSWER stay the same round over round" signal (answer identity, not pool position —
see Section 13's class-identity note); `pairwise_winner` holds `"seed"` (round 1) /
`"defended"` (the answer survived, whether via the literal pool slot or a same-answer
reworded win) / `"replaced"` (a genuine answer change). This is the
direct source for the requested "average per conversation step of verifier score, with
CI/percentile bars" plot — group by `round`, aggregate `mu` across images with a bootstrap
CI, one line per task type + overall, structurally the belief-tracking analogue of
`step_f1_overall.png`. `reasoning` holds the running transcript (one line per round's
decisive comparison, including the winning `<answer>` tag), which is what lets
`cot_yaml_step_report.py`-style per-round F1 be computed the same
way it already is for `cot_yaml` — reuse `compute_step_metrics()` directly against this file
rather than writing a parallel implementation.

**Full verifier-call transparency and compute accounting.** The checkpoint above only ever
holds the *parsed* result of each round (final `mu`/`sigma2`, final feedback) — under the
hood, a belief-mode round runs several individual tournament-pair comparisons (`n + k(n-k) +
C(k,2)` of them, the PPT ring-pass + pivot-round schedule for pool size `n`), and none of
their raw prompts or responses were previously recoverable once a round finished. Two
additions close that gap, both following conventions already established elsewhere in this
codebase rather than inventing new ones:

- **`verifier_log_<baseline>.jsonl`** — one record per individual verifier call, same
  convention as `run_judge_phase`'s `judge_log.jsonl` (`src/common/vision_eval.py`), now with
  four `call_type` values instead of one: `"candidate"` (mode-agnostic — every K-candidate
  trajectory generated each round, whether or not it ends up scored or selected, with its
  raw text plus `extracted_answer`/`answer_tag_found` — this is the direct answer to "log the
  trajectories for parsing and quality checking later": `answer_tag_found`'s hit rate across
  a run is exactly how the `<answer>`-tag mechanism's own reliability gets checked, rather
  than trusted blindly), `"tournament_pair"` (every ring-pass/pivot-round pair, belief mode —
  `pool_pair` [a, b] pool indices, `pool_text_a`/`pool_text_b`, `analysis`, `score_a`/
  `score_b`), `"same_answer"` (the class-identity check, belief mode only, fired only when a
  new candidate outranks the incumbent's pool slot — `answer_a`/`answer_b`, `parsed_same`),
  `"score"` (every independent candidate score, argmax mode — `candidate_text`,
  `parsed_score`, `parsed_feedback`). Every record carries `agml_dataset, class_label,
  image_name, round, user_prompt` (the fully-formatted prompt text actually sent),
  `prompt_tokens`, `completion_tokens`. This is what makes every individual call within a
  round — generation, tournament, and identity-check alike — auditable after the fact, not
  just the round's final state.
- **`prompt_tokens` / `completion_tokens` / `elapsed_s`** — added to the per-round checkpoint
  row itself (Section 14 above), summing every LLM call made that round (generation +
  verifier calls). `elapsed_s` is wall-clock for the whole round's *batch* (every image
  active that round shares one set of calls), not exclusive per image — still informative
  for "does cost grow with round depth," just not additive across images without
  double-counting.
- **`compute_summary_<baseline>.json`**, written once per run — `this_run` (LLM calls and
  prompts actually made in *this* invocation, from a shared call-counter dict incremented at
  every `llm.generate()` call in this module — batch size never inflates the count, and a
  resumed run's skipped rows never appear here) and `cumulative_all_rows` (total
  prompt/completion tokens summed across every row in the checkpoint, old and new, for the
  full lifetime picture of a resumed run).

## 15. Two baselines

**Basic PPT-score baseline** (`Verifier-Argmax` condition, `--baseline argmax`) — the
ablation with no pairwise Bradley–Terry, no belief carry-over, no reference-anchored
judgments at all. Same round budget (min 3 / max 10) and the same K-candidates-per-round
generation, but every round just independently scores K fresh candidates 1–20
(`verifier_vision.py`'s existing mechanism, unchanged) and keeps the best-scoring candidate
seen so far across all rounds. "Confidence" is the raw `score_best / 19`, binned with the
same 0.85/0.6 boundaries for direct comparability. This isolates exactly what the
pairwise/belief-tracking machinery buys over naive repeated absolute-scoring at an equal
compute budget.

**Flow chart (argmax baseline)** — contrast with Section 13's belief-mode chart:

```
ROUND 1
  generate K candidates (image + question, temperature sampling)
        |
  verifier scores each candidate INDEPENDENTLY 1-20 + feedback
  (no reference, no other candidate shown -- this is exactly the
   calibration-free mode Section 13's "why every comparison is
   pairwise" note argues against)
        |
  champion = argmax(K scores)  -->  C_best = champion, best_score = its score
        |
        v
ROUND k >= 2  <-----------------------------------------------------+
  generate K candidates, conditioned on C_best + feedback from        |
  THIS ROUND's eventual champion -- not necessarily the same          |
  candidate C_best itself came from, see note below                  |
        |                                                            |
  verifier scores each candidate independently 1-20 + feedback        |
  (K independent calls -- no pairwise call exists in this mode)       |
        |                                                            |
  champion = argmax(K scores) this round                              |
        |                                                            |
  champion_score >= best_score (all-time)?                            |
    yes -> C_best <- champion, best_score <- champion_score           |
    no  -> C_best UNCHANGED (this round's champion is discarded)      |
    (either way: feedback <- this round's champion's feedback --      |
     see the mismatch note below)                                    |
        |                                                            |
  round >= 3?                                                         |
    confidence = (best_score - 1) / 19                                |
        |                                                            |
    confidence >= 0.85 ("high")?  -- no, and round < 10 --------------+
        |
       yes  --  or round == 10 regardless
        v
      STOP -- report C_best + confidence bin
        high  (>= 0.85)
        medium (0.6 <= . < 0.85)
        low    (< 0.6)
```

**A real quirk this comparison surfaces**: unlike belief mode (where the incumbent shown as
context and the feedback describing it are always the same candidate, by construction),
argmax mode's re-ask can show a *stale* `C_best` as "your previous answer" alongside
feedback that actually describes a *different*, discarded candidate from the round that just
ran — because `feedback` updates unconditionally every round but `C_best` only updates when
that round's champion beats the all-time best. Belief mode can't have this mismatch, since
`C_best` and `feedback` always update together, from the same duel. Not fixed as part of this
work — flagged here since building this comparison chart is what surfaced it.

**Full PPT tournament baseline** (`PPT-Direct-N10` condition) — **not a new script.**
`src/verifier_lmv.py` already implements arXiv:2607.05391 Section 3.2 faithfully, using the
real `llm_verifier` package's own `pivot_tournament` module (`ring_cycle`, `select_pivots`,
`pivot_round_pairs`, `bradley_terry`) rather than a hand-rolled reimplementation, plus
`fine_grained_reward.extract_score` for the continuous-reward extraction. It generates N=10
candidates per image (`direct` and `self_cot` strategies), runs the ring-pass -> pivot-selection
-> pivot-round schedule, and judges the argmax-by-`ppt_reward` winner at N in {1,5,10} — its
condition labels (`PPT-Direct-N{1,5,10}`, `PPT-SelfCoT-N{1,5,10}`) were already registered in
`src/common/method_comparison.py`'s `CONDITION_STYLES` before this belief-tracking work
started. We compare against its `PPT-Direct-N10` slice specifically (closest in candidate count
to a typical full `verifier_cot` run's total across 3–10 rounds × K candidates/round). No
adaptive stopping in this baseline (per Section 10/11 — that's the point of the comparison).

**Score-extraction correction (important for the paper's methods section).** The scoring
mechanism `verifier_lmv.py` originally used — one free-generation call asking the model to
write both `<score_A>` and `<score_B>` tags inline, then reading whatever top-20 logprobs
happened to land at wherever the tag ended up — is unreliable: a real feasibility test in this
project (documented in `src/legacy/verifier_ppt/run_verifier_ppt.py`'s module docstring —
archived prototype) found it collapses onto the first letter of the scale ("A") ~96% of the time, because the
model's unconstrained top-20 logprobs at a freely-reached tag position usually aren't
dominated by the 20 target scale letters at all. The fix, now applied in `verifier_lmv.py`'s
`_score_pairs_batch`, is a genuine **two-pass teacher-forced/prefill** extraction:

1. **Pass 1** — free-generate the comparative analysis only (temperature 1.0, no logprobs
   needed; any tag content it writes here is discarded, not trusted).
2. **Pass 2** — literally continue that exact prompt text with the analysis plus a prefilled
   tag (`"\n<score_A>"`, then separately `"\n<score_B>"`) appended, and force exactly one more
   token, constrained via a grammar mask (`SamplingParams(max_tokens=1, logprobs=20,
   structured_outputs=StructuredOutputsParams(choice=<the 20 scale letters, plus their
   leading-space tokenization variants>))`) to just the scale's own letters. The returned
   top-20 logprobs are then already a distribution over the scale itself, not competing
   against the model's full vocabulary — which is what `extract_score`'s expectation formula
   (Eq. 3.1) actually assumes.

This is the offline-vLLM equivalent of an OpenAI-compatible server's
`continue_final_message=True` (which is what the vendored `llm_verifier` package's own
`call_verifier`/`_score_tags_by_prefill` uses when talking to a server — not directly usable
here, since this whole repo generates via vLLM's offline batch `LLM.generate()` API, not a
served endpoint).

## 16. Wiring into `cot_compare`'s delta chart

`verifier_belief`/`verifier_argmax` (our contribution and its ablation) get added the same way
`self_cot`/`cot_yaml` already are: a `compute_condition_stats`/`compute_delta_stats` call and
a `CONDITION_STYLES`/`CONDITION_COLOR` entry in `src/common/method_comparison.py` (unused
hex, distinct from the existing entries) — that file already generalizes over whatever keys are
present, per its own docstring. The PPT baseline reuses `verifier_lmv.py`'s existing
`PPT-Direct-N10` condition/color entry rather than adding a new one. One wrinkle:
`verifier_lmv.py`'s `--output_dir` is flat (one model's worth of output per run, no `<model>/`
subfolder, unlike every other condition here) — its `ckpt_judged.csv` rows don't self-identify
a model, so `cot_compare.py` attaches them to whichever model `verifier_cot`'s own
belief/argmax data identifies, via an explicit `--verifier_lmv_ckpt` path rather than
directory auto-discovery. Per this project's established convention, scope to
`google_gemma-4-26B-A4B-it` only for the delta/gemma-only charts, same as every other CoT
condition.

## 17. Compute cost note

Re-including the image on every round's VLM call and every verifier call is not a new cost
regime — `cot_yaml.py`'s multi-step chain and `verifier_vision.py`'s per-candidate scoring
already do exactly this (image present in every turn's `msgs`, not cached across turns at the
Python/vLLM-call level), and both already run at the project's current scale on the farm GPU.
This design's cost scales with `K × rounds` generate calls plus, per round, either
`3 × (n + k(n-k) + C(k,2))` tournament-pair calls (belief mode — one analysis pass + two
forced-decode score passes per pair, `src/common/ppt_scoring.py`, over the full PPT ring-pass
+ pivot-round schedule for pool size `n` = K or K+1, `k` = `--k_pivots`) or `K` scoring calls
(argmax mode — a single JSON-parsed call per candidate, no pairwise structure) — the
belief-mode term is O(nk) rather than linear, the direct cost of running the paper's real
tournament schedule every round instead of a single fixed pairing, but still bounded by the
round cap (10), K (small, e.g. 3–5), and k (small, default 2), so worst case is a fixed, known
multiple of a single `cot_yaml` run's cost, not an open-ended one. Belief mode adds one more,
much smaller term: the `same_answer_prompt` class-identity check (Section 13) fires at most
once per round — only when a new candidate outranks the incumbent's own pool slot, never per
pair — and is text-only (no image, `max_tokens=40`), so its cost is negligible next to the
tournament term even though it runs on the same LLM instance. Belief mode's per-round
`prompt_tokens`/`completion_tokens`/`elapsed_s`
columns (Section 14) make this measured, not just bounded — see also `compute_summary_
<baseline>.json`'s `n_llm_calls`, which will show belief mode running noticeably more calls
per round than argmax at the same K, the direct cost of the more reliable scoring mechanism.

**`--rephrase`'s added cost** (Section 13.2, off by default): one extra text-only call *per
group* (not per round) — the question-rephrasing call, sampled `rephrase_count` times in one
`generate()` — plus, *every* round, `rephrase_count` additional single-candidate generate calls
(one per rephrased-question variant; deliberately not batched as `n` samples of one prompt,
since each candidate must come from a genuinely different question). Pool size grows from `K`
to `K + rephrase_count`, so the tournament's O(nk) term grows accordingly too (`n` in Section
13's cost formula becomes `K + rephrase_count`, or `+1` from round 2 on). With the current
defaults (`K=8`, `rephrase_count=4`), that's a 1.5× increase in per-round generation calls and
a `12/8 = 1.5×` (round 1) to `13/9 ≈ 1.44×` (round $\geq$ 2) increase in tournament pool size —
bounded and known ahead of time, same reasoning as the base cost above, not open-ended.

## 18. Config file for the verifier prompt template

`src/config/verifier_prompts.yaml` — `src/config/prompts.yaml` and `src/config/cot_prompts.yaml`
cover classification/description prompts and the CoT step chains respectively, but neither
owns the verifier's own prompt templates. Holds:
- **`rubrics`** / **`task_type_to_rubric`** — diagnostic reference material shown to the
  verifier only, shared by both `--baseline` modes.
- **`score_prompt`** — the per-candidate 1–20 scorer (rubric + question + candidate),
  `--baseline argmax` only.
- **`same_answer_prompt`** — Section 13's class-identity check, `--baseline belief` only
  (question + the two extracted `<answer>` strings).
- **`pairwise_prompt`** — `--baseline belief`'s tournament-ranking duels AND `src/verifier_
  lmv.py`'s own PPT baseline. Originally lived as a fixed Python string in `src/common/
  ppt_scoring.py` ("a citable reproduction of the paper's prompt, not meant to be edited") —
  moved here after a real run showed the paper's own template needed two fixes: (1) it showed
  the literal `<score_A>`/`<score_B>` tags right after the "analyze, then score" instruction,
  which caused the model to skip analysis almost entirely (99.6% empty analysis on a 20k-pair
  sample) rather than write free text first — the tags are no longer shown in this prompt at
  all, only ever supplied by pass 2's forced continuation (`src/common/ppt_scoring.py::
  build_score_prompt`); (2) the rating rule was lenient enough that 80.6% of a real sample
  scored *both* trajectories above 0.9 — added an explicit anti-tie instruction. So this is now
  an adapted version of the paper's template, not a verbatim reproduction — exactly why it now
  belongs in config (edit-without-redeploy, one citable record) rather than staying hardcoded.
  `src/common/ppt_scoring.py::pairwise_scoring_prompt` still keeps an internal fallback copy of
  this text for a caller that hasn't been wired up to load the YAML, but every real caller
  passes the YAML version explicitly.
- **`question_rephrase_prompt`** — Section 13.2's rephrase step, `--baseline belief` only,
  text-only (question in, reworded question out, no image — rephrasing text needs no visual
  input).
