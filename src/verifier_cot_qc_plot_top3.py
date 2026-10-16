"""
Paper 2, Stage 5 — top3 QC plot. See docs/stage5_classification_verifier.md.
No GPU, no model calls, no network — reads an already-graded
`--verifier_type top3` checkpoint CSV (any `--top3_mode`: explicit,
resample, verifier_rubric) plus whatever real dataset images are already
cached locally under --images_root. Four outputs, written to
`<ckpt's own dir>/qc_plots_top3/`:

1. **Candidate gallery** (`candidate_gallery.png`, paginated) — random
   datasets x random images, each image next to all 3 candidate
   predictions (verifier_score per candidate, the winning one marked),
   the winner's own criterion_scores (bare numbers, in order — no
   question text), and the judge's verdict.
2. **Winner score correlation** (`winner_score_correlation.png`) — same
   box+jittered-strip/histogram split verifier-score-correlation-plot
   produces, reused directly against this checkpoint's WINNING
   candidate's verifier_score vs. judged_correct.
3. **Per-candidate score distribution** (`per_candidate_score_distribution.png`)
   — ALL 3 candidates per row (not just the winner), split by whether
   THAT candidate's own extracted answer text-matches ground_truth (a
   cheap proxy — only the winner is ever actually judged by the LLM
   judge, so this is NOT the same as judged_correct). Answers a broader
   question than (2): does verifier_score discriminate a right answer
   from a wrong one in general, across the whole pool, not just among
   winners?
4. **Pool summary** (`pool_summary.png`/`.json`) — pool_coverage (does
   the right answer appear in ANY candidate) and
   verifier_pick_accuracy_given_in_pool (when it does, does argmax
   actually land on it) — the same two numbers metrics.json's
   top3_pool_analysis reports, as a chart.

Usage:
    uv run python main.py verifier-cot-qc-plot-top3 \\
        --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic_top3.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re

from PIL import Image as PILImage

from src.detection_box_verifier_summary import _numbered_path
from src.verifier_cot_qc_plot import load_rows, resolve_image_paths, sample_rows_for_gallery
from src.verifier_score_correlation_plot import load_scored_rows, plot_correlation

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_COLOR_CORRECT = "seagreen"
_COLOR_INCORRECT = "firebrick"


def _winner_answer(prediction: str) -> str:
    m = _ANSWER_RE.search(prediction or "")
    return m.group(1).strip() if m else ""


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", (s or "").lower()).strip()


def plot_candidate_gallery(
    sampled: list[dict], image_paths: dict[tuple, str], out_path: Path,
    max_rows_per_figure: int = 12, title: str = "",
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = [sampled[i:i + max_rows_per_figure] for i in range(0, len(sampled), max_rows_per_figure)] or [[]]
    written: list[Path] = []

    for part, chunk in enumerate(chunks, start=1):
        n = len(chunk)
        fig, axes = plt.subplots(n, 2, figsize=(11, 3.4 * n), squeeze=False)
        for row_i, r in enumerate(chunk):
            ax_img, ax_text = axes[row_i]
            img_path = image_paths.get((r.get("agml_dataset", ""), r.get("image_name", "")))
            if img_path and Path(img_path).exists():
                ax_img.imshow(PILImage.open(img_path).convert("RGB"))
            else:
                ax_img.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=9, color="gray")
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f"{r.get('agml_dataset', '')} / {r.get('image_name', '')}", fontsize=8)

            try:
                cand_answers = json.loads(r.get("candidate_predictions") or "[]")
                cand_scores = json.loads(r.get("candidate_verifier_scores") or "[]")
            except json.JSONDecodeError:
                cand_answers, cand_scores = [], []
            winner_answer = _winner_answer(r.get("prediction", ""))
            try:
                criterion_scores = {int(k): int(v) for k, v in json.loads(r.get("criterion_scores") or "{}").items()}
            except (json.JSONDecodeError, ValueError):
                criterion_scores = {}
            ordered_scores = [criterion_scores[k] for k in sorted(criterion_scores)]

            judged = r.get("judged_correct", "")
            judge_verdict = {"1": "correct", "0": "incorrect"}.get(str(judged).strip(), "not judged")

            lines = [
                f"ground_truth: {r.get('ground_truth', r.get('class_label', ''))}",
                f"task_type: {r.get('task_type', '')}",
                f"judge verdict: {judge_verdict}",
                "",
                "candidates:",
            ]
            for idx, (ans, score) in enumerate(zip(cand_answers, cand_scores), start=1):
                mark = " [CHOSEN]" if _norm(ans) == _norm(winner_answer) else ""
                lines.append(f"  {idx}. {ans}  (verifier_score={score:.3f}){mark}")
            lines.append("")
            lines.append(f"criterion_scores (winner, in order): {ordered_scores}")

            ax_text.axis("off")
            ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=7.5, family="monospace",
                         wrap=True, transform=ax_text.transAxes)

        subtitle = f": {title}" if title else ""
        fig.suptitle(f"Top3 candidate gallery{subtitle} (part {part}/{len(chunks)})" if len(chunks) > 1
                     else f"Top3 candidate gallery{subtitle}", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        page_path = _numbered_path(out_path, part, len(chunks))
        page_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(page_path)

    return written


def plot_per_candidate_score_distribution(rows: list[dict], out_path: Path, title: str = "", seed: int = 42) -> dict:
    """Every candidate from every row, labeled correct/incorrect by a
    cheap text-match against ground_truth (NOT the LLM judge — only the
    winner is ever actually judged). Same box+strip/histogram shape as
    verifier_score_correlation_plot.py's plot_correlation, built locally
    here since the input is a flattened per-candidate list, not a
    per-row (score, judged_correct) pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import random as _random

    from src.common.bootstrap import point_biserial

    scores: list[float] = []
    binary: list[int] = []
    for r in rows:
        try:
            cand_answers = json.loads(r.get("candidate_predictions") or "[]")
            cand_scores = json.loads(r.get("candidate_verifier_scores") or "[]")
        except json.JSONDecodeError:
            continue
        gt = r.get("ground_truth", "")
        for ans, score in zip(cand_answers, cand_scores):
            scores.append(float(score))
            binary.append(int(_norm(ans) == _norm(gt) and bool(_norm(gt))))

    if not scores:
        raise SystemExit("ERROR: no rows with parseable candidate_predictions/candidate_verifier_scores found.")

    correct_scores = [s for s, c in zip(scores, binary) if c]
    incorrect_scores = [s for s, c in zip(scores, binary) if not c]
    r_value = point_biserial(scores, binary)
    mean_correct = sum(correct_scores) / len(correct_scores) if correct_scores else None
    mean_incorrect = sum(incorrect_scores) / len(incorrect_scores) if incorrect_scores else None

    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(12, 5.5))
    groups = [incorrect_scores, correct_scores]
    labels = [f"Text-mismatch (n={len(incorrect_scores)})", f"Text-match (n={len(correct_scores)})"]
    colors = [_COLOR_INCORRECT, _COLOR_CORRECT]
    bp = ax_box.boxplot(groups, tick_labels=labels, widths=0.5, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    rng = _random.Random(seed)
    for i, (group, color) in enumerate(zip(groups, colors), start=1):
        jitter = [i + rng.uniform(-0.15, 0.15) for _ in group]
        ax_box.scatter(jitter, group, color=color, alpha=0.4, s=10, zorder=3, edgecolors="none")
    ax_box.set_ylabel("verifier_score")
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.set_title("Per-candidate distribution (all 3 candidates/row)")
    ax_box.grid(alpha=0.3, axis="y")

    bins = [i / 20 for i in range(21)]
    ax_hist.hist(incorrect_scores, bins=bins, color=_COLOR_INCORRECT, alpha=0.5, label=labels[0], density=True)
    ax_hist.hist(correct_scores, bins=bins, color=_COLOR_CORRECT, alpha=0.5, label=labels[1], density=True)
    ax_hist.set_xlabel("verifier_score")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=8, loc="upper center")
    ax_hist.grid(alpha=0.3, axis="y")

    r_str = f"{r_value:.3f}" if r_value is not None else "n/a"
    mc_str = f"{mean_correct:.3f}" if mean_correct is not None else "n/a"
    mi_str = f"{mean_incorrect:.3f}" if mean_incorrect is not None else "n/a"
    subtitle = f": {title}" if title else ""
    fig.suptitle(
        f"verifier_score vs. text-match (ALL candidates){subtitle} — NOT the LLM judge\n"
        f"point-biserial r={r_str}   mean match={mc_str}   mean mismatch={mi_str}",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n": len(scores), "against": "ground_truth_text_match_all_candidates",
        "point_biserial_r": r_value, "mean_score_match": mean_correct, "mean_score_mismatch": mean_incorrect,
    }


def plot_pool_summary(rows: list[dict], out_path: Path, title: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ip_rows = [r for r in rows if r.get("answer_in_pool") not in (None, "")]
    if not ip_rows:
        raise SystemExit("ERROR: no rows with answer_in_pool — is this really a top3 checkpoint?")
    n_in_pool = sum(int(r["answer_in_pool"]) for r in ip_rows)
    pool_coverage = n_in_pool / len(ip_rows)
    picked_rows = [r for r in ip_rows if r.get("verifier_picked_correct") not in (None, "")]
    pick_accuracy = (
        sum(int(r["verifier_picked_correct"]) for r in picked_rows) / len(picked_rows) if picked_rows else None
    )

    fig, ax = plt.subplots(figsize=(5, 5.5))
    labels = ["pool_coverage", "verifier_pick_accuracy\n(given in pool)"]
    values = [pool_coverage, pick_accuracy if pick_accuracy is not None else 0.0]
    colors = ["dodgerblue", "mediumorchid"]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    subtitle = f": {title}" if title else ""
    ax.set_title(f"Top3 pool summary{subtitle}\n(n={len(ip_rows)})", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "n": len(ip_rows), "n_answer_in_pool": n_in_pool, "pool_coverage": round(pool_coverage, 4),
        "verifier_pick_accuracy_given_in_pool": round(pick_accuracy, 4) if pick_accuracy is not None else None,
    }


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    rows = [r for r in rows if r.get("candidate_predictions")]
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows with candidate_predictions — is this a "
                          f"--verifier_type top3 checkpoint (ckpt_<model>_top3.csv)?")

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "qc_plots_top3"

    sampled = sample_rows_for_gallery(rows, args.n_datasets, args.n_per_dataset, args.seed)
    image_paths = resolve_image_paths(sampled, Path(args.images_root))
    n_missing = sum(1 for r in sampled if (r.get("agml_dataset", ""), r.get("image_name", "")) not in image_paths)
    if n_missing:
        print(f"  [note] {n_missing}/{len(sampled)} sampled row(s) have no resolvable local image.")

    gallery_paths = plot_candidate_gallery(
        sampled, image_paths, out_dir / "candidate_gallery.png",
        max_rows_per_figure=args.max_rows_per_figure, title=args.title,
    )
    for p in gallery_paths:
        print(f"  Saved: {p}")

    scores, binary, judge_key = load_scored_rows(ckpt_path)
    if scores:
        corr_path = out_dir / "winner_score_correlation.png"
        corr_summary = plot_correlation(scores, binary, judge_key, corr_path, title=args.title, seed=args.seed)
        (out_dir / "winner_score_correlation.json").write_text(json.dumps(corr_summary, indent=2))
        print(f"  Saved: {corr_path}")
    else:
        print("  [note] no judged/correct rows found — skipping winner_score_correlation.png")

    per_cand_path = out_dir / "per_candidate_score_distribution.png"
    per_cand_summary = plot_per_candidate_score_distribution(rows, per_cand_path, title=args.title, seed=args.seed)
    (out_dir / "per_candidate_score_distribution.json").write_text(json.dumps(per_cand_summary, indent=2))
    print(f"  Saved: {per_cand_path}")

    pool_path = out_dir / "pool_summary.png"
    pool_summary = plot_pool_summary(rows, pool_path, title=args.title)
    (out_dir / "pool_summary.json").write_text(json.dumps(pool_summary, indent=2))
    print(f"  Saved: {pool_path}")
    print(f"  pool_coverage={pool_summary['pool_coverage']:.3f} "
          f"verifier_pick_accuracy_given_in_pool={pool_summary['verifier_pick_accuracy_given_in_pool']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A --verifier_type top3 checkpoint CSV (ckpt_<model>_top3.csv) — any "
                              "--top3_mode (explicit, resample, verifier_rubric).")
    parser.add_argument("--images_root", default="dataset/images",
                         help="Images are resolved by globbing <images_root>/<dataset>/*/<image_name> "
                              "against what's already cached on disk — never downloads/re-caches.")
    parser.add_argument("--out_dir", default=None, help="Default: <ckpt's own directory>/qc_plots_top3/")
    parser.add_argument("--n_datasets", type=int, default=10, help="Random datasets sampled for the gallery.")
    parser.add_argument("--n_per_dataset", type=int, default=3, help="Random rows per sampled dataset.")
    parser.add_argument("--max_rows_per_figure", type=int, default=12,
                         help="Gallery rows per PNG before paginating into _2/_3/... files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
