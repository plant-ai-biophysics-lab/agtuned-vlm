"""
Paper 2, Stage 3 — Verifier QC plot. See
docs/stage3_classification_verifier.md. No GPU, no model calls, no
network — reads an already-graded `verifier-cot` checkpoint CSV plus
whatever real dataset images are already cached locally under
--images_root (globbed directly, never re-downloaded/re-cached — this is
a read-only QC tool, not a generation script). Two outputs:

1. **Rubric gallery**: --n_datasets random datasets (--n_per_dataset
   images each) — one panel per sampled row, image on the left, the
   row's ground truth/verifier_score/criterion_scores (both the raw
   JSON and each criterion's own rubric question, so a reader doesn't
   have to cross-reference verifier_prompts.yaml by hand) on the right.
   Paginated (--max_rows_per_figure) so a 30-row gallery doesn't become
   one enormous image — same convention
   detection_box_verifier_summary.py's plot_crop_gallery/
   plot_refinement_timeline use.
2. **Criterion score distribution**: one small-multiple subplot per
   task_type actually present in --ckpt (task_types use DIFFERENT
   rubrics with different criteria — see verifier_prompts.yaml's
   task_type_to_rubric — so pooling "criterion 1" across task_types
   would silently mix unrelated questions). Each subplot is a stacked
   bar, one bar per criterion number, segments = how many rows scored
   1/2/3 on it.

Usage:
    uv run python main.py verifier-cot-qc-plot \\
        --ckpt outputs_final/verifier_cot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import json
import random
from collections import defaultdict

from PIL import Image as PILImage

from src.detection_box_verifier_summary import _numbered_path

# Ordinal 1/2/3 ramp (worst -> best), NOT a categorical hue cycle — same
# "status" convention as the box-verifier's confidence-distribution
# charts: a score of 1 is always this red regardless of which criterion
# or task_type it's drawn for.
_SCORE_COLORS = {1: "firebrick", 2: "goldenrod", 3: "seagreen"}


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _rubric_lines(rubric_text: str) -> dict[int, str]:
    """{criterion_number: question_text} from a verifier_prompts.yaml
    rubrics entry's raw numbered-line text."""
    lines: dict[int, str] = {}
    for line in rubric_text.strip().splitlines():
        line = line.strip()
        if not line or "." not in line:
            continue
        n_str, _, text = line.partition(".")
        if n_str.strip().isdigit():
            lines[int(n_str.strip())] = text.strip()
    return lines


def load_rows(ckpt_path: Path) -> list[dict]:
    with open(ckpt_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_image_paths(rows: list[dict], images_root: Path) -> dict[tuple, str]:
    """{(agml_dataset, image_name): path} for every dataset present in
    `rows`, resolved by globbing `images_root/<dataset>/*/<image_name>`
    directly against whatever's ALREADY cached on disk — deliberately not
    load_hf_samples() (which needs a manifest and would fall back to a
    live HF download/re-cache for anything missing); this is a read-only
    local QC tool, not a generation script, so it should never trigger
    network access or grow the local image cache."""
    by_key: dict[tuple, str] = {}
    for r in rows:
        ds_name, image_name = r.get("agml_dataset", ""), r.get("image_name", "")
        if not ds_name or not image_name or (ds_name, image_name) in by_key:
            continue
        matches = list((images_root / ds_name).glob(f"*/{image_name}"))
        if matches:
            by_key[(ds_name, image_name)] = str(matches[0])
    return by_key


def sample_rows_for_gallery(
    rows: list[dict], n_datasets: int, n_per_dataset: int, seed: int,
) -> list[dict]:
    """--n_datasets random datasets (deterministic given --seed), up to
    --n_per_dataset random rows each — fewer than requested if a dataset
    doesn't have that many rows in --ckpt."""
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dataset[r.get("agml_dataset", "")].append(r)
    rng = random.Random(seed)
    datasets = sorted(d for d in by_dataset if d)
    chosen_datasets = rng.sample(datasets, min(n_datasets, len(datasets)))
    sampled: list[dict] = []
    for ds_name in chosen_datasets:
        pool = by_dataset[ds_name]
        sampled.extend(rng.sample(pool, min(n_per_dataset, len(pool))))
    return sampled


def plot_rubric_gallery(
    sampled: list[dict], image_paths: dict[tuple, str], rubrics_by_task_type: dict[str, dict[int, str]],
    out_path: Path, max_rows_per_figure: int = 12, title: str = "",
) -> list[Path]:
    """One (image, text panel) row per sampled checkpoint row. Returns the
    list of PNG paths actually written (paginated)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = [sampled[i:i + max_rows_per_figure] for i in range(0, len(sampled), max_rows_per_figure)] or [[]]
    written: list[Path] = []

    for part, chunk in enumerate(chunks, start=1):
        n = len(chunk)
        fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
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

            task_type = r.get("task_type", "")
            rubric_map = rubrics_by_task_type.get(task_type, {})
            try:
                criterion_scores = {int(k): int(v) for k, v in json.loads(r.get("criterion_scores") or "{}").items()}
            except (json.JSONDecodeError, ValueError):
                criterion_scores = {}
            lines = [
                f"ground_truth: {r.get('ground_truth', r.get('class_label', ''))}",
                f"task_type: {task_type}",
                f"verifier_score: {r.get('verifier_score', '')}",
                "",
                f"criterion_scores (raw): {json.dumps(criterion_scores)}",
                "",
            ]
            for cn in sorted(criterion_scores):
                q = rubric_map.get(cn, "")
                q_short = (q[:70] + "…") if len(q) > 70 else q
                lines.append(f"  C{cn} [{criterion_scores[cn]}]  {q_short}")
            ax_text.axis("off")
            ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=7.5, family="monospace",
                         wrap=True, transform=ax_text.transAxes)

        subtitle = f": {title}" if title else ""
        fig.suptitle(f"Verifier rubric gallery{subtitle} (part {part}/{len(chunks)})" if len(chunks) > 1
                     else f"Verifier rubric gallery{subtitle}", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        page_path = _numbered_path(out_path, part, len(chunks))
        page_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(page_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(page_path)

    return written


def plot_criterion_score_distribution(
    rows: list[dict], rubrics_by_task_type: dict[str, dict[int, str]], out_path: Path, title: str = "",
) -> dict:
    """One small-multiple subplot per task_type present in `rows` — a
    stacked bar per criterion number, segments = count of rows scoring
    1/2/3 on it. Returns the counts dict also written alongside as JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # counts[task_type][criterion_n][score] -> count
    counts: dict[str, dict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: {1: 0, 2: 0, 3: 0}))
    for r in rows:
        task_type = r.get("task_type", "")
        try:
            criterion_scores = {int(k): int(v) for k, v in json.loads(r.get("criterion_scores") or "{}").items()}
        except (json.JSONDecodeError, ValueError):
            continue
        for cn, score in criterion_scores.items():
            if score in (1, 2, 3):
                counts[task_type][cn][score] += 1

    task_types = sorted(counts)
    if not task_types:
        raise SystemExit("ERROR: no rows with parseable criterion_scores found.")

    fig, axes = plt.subplots(1, len(task_types), figsize=(5.5 * len(task_types), 5), squeeze=False)
    axes = axes[0]
    summary: dict = {}
    for ax, task_type in zip(axes, task_types):
        per_criterion = counts[task_type]
        criterion_ns = sorted(per_criterion)
        bottoms = [0] * len(criterion_ns)
        for score in (1, 2, 3):
            vals = [per_criterion[cn][score] for cn in criterion_ns]
            ax.bar([f"C{cn}" for cn in criterion_ns], vals, bottom=bottoms, color=_SCORE_COLORS[score],
                   label=f"score={score}")
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_title(task_type, fontsize=10)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3, axis="y")
        summary[task_type] = {f"C{cn}": per_criterion[cn] for cn in criterion_ns}

    axes[-1].legend(fontsize=8, loc="upper right")
    subtitle = f": {title}" if title else ""
    fig.suptitle(f"Criterion score distribution{subtitle}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return summary


def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.ckpt)
    rows = load_rows(ckpt_path)
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows.")
    rows = [r for r in rows if r.get("criterion_scores")]
    if not rows:
        raise SystemExit(f"ERROR: {ckpt_path} has no rows with criterion_scores — was this graded "
                          f"(--verifier_type simple or top3, not --skip_judge-only generation)?")

    vprompts = load_yaml(Path(args.verifier_prompts))
    task_type_to_rubric = vprompts.get("task_type_to_rubric", {})
    rubrics_raw = vprompts.get("rubrics", {})
    rubrics_by_task_type = {
        task_type: _rubric_lines(rubrics_raw.get(rubric_key, ""))
        for task_type, rubric_key in task_type_to_rubric.items()
    }

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "qc_plots"

    sampled = sample_rows_for_gallery(rows, args.n_datasets, args.n_per_dataset, args.seed)
    image_paths = resolve_image_paths(sampled, Path(args.images_root))
    n_missing = sum(1 for r in sampled if (r.get("agml_dataset", ""), r.get("image_name", "")) not in image_paths)
    if n_missing:
        print(f"  [note] {n_missing}/{len(sampled)} sampled row(s) have no resolvable local image — "
              f"shown as \"image not found\" in the gallery.")

    gallery_paths = plot_rubric_gallery(
        sampled, image_paths, rubrics_by_task_type, out_dir / "rubric_gallery.png",
        max_rows_per_figure=args.max_rows_per_figure, title=args.title,
    )
    for p in gallery_paths:
        print(f"  Saved: {p}")

    dist_path = out_dir / "criterion_score_distribution.png"
    summary = plot_criterion_score_distribution(rows, rubrics_by_task_type, dist_path, title=args.title)
    json_path = out_dir / "criterion_score_distribution.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {dist_path}")
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True,
                         help="A verifier-cot checkpoint CSV already graded (has criterion_scores).")
    parser.add_argument("--images_root", default="dataset/images",
                         help="Images are resolved by globbing <images_root>/<dataset>/*/<image_name> "
                              "against what's already cached on disk — this is a read-only QC tool, "
                              "it never downloads/re-caches anything itself.")
    parser.add_argument("--verifier_prompts", default="src/config/verifier_prompts.yaml")
    parser.add_argument("--out_dir", default=None, help="Default: <ckpt's own directory>/qc_plots/")
    parser.add_argument("--n_datasets", type=int, default=10, help="Random datasets sampled for the gallery.")
    parser.add_argument("--n_per_dataset", type=int, default=3, help="Random rows per sampled dataset.")
    parser.add_argument("--max_rows_per_figure", type=int, default=12,
                         help="Gallery rows per PNG before paginating into _2/_3/... files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="", help="Optional subtitle (e.g. model name).")
    main(parser.parse_args())
