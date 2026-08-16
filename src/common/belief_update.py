"""Bradley-Terry belief tracking via moment matching (assumed density
filtering / TrueSkill-style updates) — the math from
docs/verifier_score_belief_update.md, sections 2-8 and 13. Pure functions,
no vLLM/model dependency, shared by verifier_cot.py (the belief-tracking
round loop) and its argmax baseline for confidence binning.

Every candidate's quality belief is a Gaussian (mu, sigma2). A pairwise
verifier comparison is combined with that belief via moment matching (exact
mean/variance, then re-approximated as a Gaussian) rather than an exact
Bayesian update, since a Gaussian belief combined with a sigmoid-shaped
(Bradley-Terry) observation is not itself Gaussian. See the doc for the full
derivation — this module is deliberately just the formulas, not the why.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_pdf(z: float) -> float:
    """N(z) — standard normal density."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _norm_cdf(z: float) -> float:
    """Phi(z) — standard normal CDF, via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def correction_terms(z: float) -> tuple[float, float]:
    """v(z), w(z) — the truncated-normal mean/variance correction terms
    (doc Section 5). Both are always positive and grow as z -> -inf (a more
    surprising outcome shifts/shrinks the belief more). Guards the z -> -inf
    tail (Phi(z) underflowing to 0) by clamping the denominator, since a
    verifier upset after a very lopsided belief is rare but not impossible.
    """
    phi = _norm_cdf(z)
    phi = max(phi, 1e-12)
    v = _norm_pdf(z) / phi
    w = v * (v + z)
    return v, w


@dataclass(frozen=True)
class Belief:
    mu: float
    sigma2: float


def pairwise_update(winner: Belief, loser: Belief, beta: float) -> tuple[Belief, Belief]:
    """One Bradley-Terry moment-matching update (doc Section 5) given that
    `winner` beat `loser` in a single verifier comparison. beta is the
    per-comparison noise hyperparameter (bigger beta = trust any one
    verdict less). Returns (new_winner_belief, new_loser_belief) — sigma2
    strictly shrinks for both sides; mu moves toward the winner for both.
    """
    c = math.sqrt(winner.sigma2 + loser.sigma2 + 2.0 * beta * beta)
    z = (winner.mu - loser.mu) / c
    v, w = correction_terms(z)

    new_winner = Belief(
        mu=winner.mu + (winner.sigma2 / c) * v,
        sigma2=winner.sigma2 * (1.0 - (winner.sigma2 / (c * c)) * w),
    )
    new_loser = Belief(
        mu=loser.mu - (loser.sigma2 / c) * v,
        sigma2=loser.sigma2 * (1.0 - (loser.sigma2 / (c * c)) * w),
    )
    return new_winner, new_loser




def stop_probability(current: Belief, anchor: Belief, beta: float) -> float:
    """P(current belief > anchor belief) (doc Section 7/13) — same
    z-score-into-Phi construction as a pairwise update, just evaluated once
    instead of folded back into the belief. `anchor` is the frozen round-1
    belief; this measures whether accumulated reasoning has produced a
    meaningfully better answer than the single-shot response, not whether
    the latest round beat the one immediately before it.
    """
    c = math.sqrt(current.sigma2 + anchor.sigma2 + 2.0 * beta * beta)
    z = (current.mu - anchor.mu) / c
    return _norm_cdf(z)


def confidence_bin(p: float, high: float = 0.85, medium: float = 0.6) -> str:
    """Bucket a stop_probability (or, for the argmax baseline, a raw
    score/10) into "high" / "medium" / "low" — doc Section 13's stopping
    table. Always returns a label; never signals "keep going" itself — the
    round-cap/early-stop DECISION is the caller's, this is purely the
    reported bin.
    """
    if p >= high:
        return "high"
    if p >= medium:
        return "medium"
    return "low"


def should_stop(round_idx: int, p: float, min_round: int = 3, max_round: int = 10,
                 high: float = 0.85) -> bool:
    """True once round_idx >= min_round AND (p clears the high-confidence
    bin OR the hard round cap is reached) — doc Section 13's stopping rule.
    Never withholds a result past max_round regardless of p.
    """
    if round_idx >= max_round:
        return True
    return round_idx >= min_round and p >= high


if __name__ == "__main__":
    # Sanity check against docs/verifier_score_belief_update.md Section 6's
    # worked numerical example: mu1=mu2=0, sigma2=1, beta=0.5, candidate 1
    # wins -> mu1_new ~= 0.505, sigma1^2_new ~= 0.745.
    c1, c2 = Belief(0.0, 1.0), Belief(0.0, 1.0)
    new1, new2 = pairwise_update(winner=c1, loser=c2, beta=0.5)
    print(f"winner: mu={new1.mu:.3f} sigma2={new1.sigma2:.3f}  (expect ~0.505, ~0.745)")
    print(f"loser:  mu={new2.mu:.3f} sigma2={new2.sigma2:.3f}  (expect ~-0.505, ~0.745)")
    assert abs(new1.mu - 0.505) < 0.01 and abs(new1.sigma2 - 0.745) < 0.01
    assert abs(new2.mu + 0.505) < 0.01 and abs(new2.sigma2 - 0.745) < 0.01
    print("OK — matches the doc's worked example.")
