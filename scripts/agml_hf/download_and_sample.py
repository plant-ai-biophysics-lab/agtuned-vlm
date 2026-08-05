"""
Stage 0 — Download active datasets from Project-AgML (HuggingFace), take a
class-balanced subsample of each, and record exactly which images were chosen
to a JSON manifest for reproducibility.

Replaces the old `agml` pip package's implicit auto-download-on-first-use with
an explicit prefetch against https://huggingface.co/Project-AgML (259 datasets,
vs. the 25 previously hand-curated). Only classification datasets are relevant
to this pipeline — see scripts/agml_hf/migrate_datasets_yaml.py, which already
filtered datasets.yaml down to classification-only.

Sampling is a fixed cap per class (--n_per_class, adjustable), seeded for
reproducibility. The manifest records the exact HF row index selected per
(dataset, class) — re-running with --from_manifest reproduces the identical
image set without re-rolling randomness, and lets later consumers materialize
images by index without depending on random state.

Resuming after a partial/rate-limited run: pass --skip_existing to re-use any
dataset already recorded in --output (from a prior run) instead of
re-downloading/re-sampling it, and only process datasets missing or previously
failed. This only applies when --n_per_class matches the existing manifest's
recorded value — changing --n_per_class invalidates every already-capped
dataset (see caveat below), so --skip_existing is ignored (with a warning) if
the cap changed. HuggingFace Hub 429s (rate limiting) are retried per-dataset
with exponential backoff before being recorded as a failure.

Caveat when changing --n_per_class: only classes that exceed the cap are
subsampled at all (smaller classes are unaffected either way), but for classes
that DO get capped, a different --n_per_class is not guaranteed to select a
superset/subset of a previous run's choice — same seed, different sample size,
generally a different draw. Treat each --n_per_class value as producing its own
independent (but internally reproducible) selection, not an incremental one.

Usage:
  uv run python scripts/agml_hf/download_and_sample.py \
    --config src/config/datasets.yaml \
    --catalog src/config/hf_catalog.json \
    --n_per_class 50 \
    --output dataset/selected_images.json

  # Reproduce a previous selection exactly (e.g. after re-cataloging), no re-sampling:
  uv run python scripts/agml_hf/download_and_sample.py \
    --config src/config/datasets.yaml \
    --from_manifest dataset/selected_images.json \
    --output dataset/selected_images.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.common.dataset_config import active_datasets, DatasetConfig, subsample_class_balanced  # noqa: E402

TASK_COLOR = {
    "disease": "#4C72B0",
    "pest_damage": "#DD8452",
    "weed_identification": "#55A868",
    "crop_identification": "#C44E52",
}

RETRYABLE_STATUS = {429, 503}
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 10


def is_retryable(exc: Exception) -> bool:
    from huggingface_hub.utils import HfHubHTTPError
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None)
        if status in RETRYABLE_STATUS:
            return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def with_retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES or not is_retryable(e):
                raise
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"\n    rate-limited ({e}), retry {attempt}/{MAX_RETRIES} in {wait}s...",
                  end=" ", flush=True)
            time.sleep(wait)


def find_label_column(features) -> str:
    for key in ("label", "labels", "class", "class_label"):
        if key in features:
            return key
    raise KeyError(f"No recognizable label column in features: {list(features)}")


def resolve_class_names(ds, label_col: str) -> list[str] | None:
    feat = ds.features[label_col]
    return list(feat.names) if hasattr(feat, "names") else None


def process_dataset(agml_name: str, hf_config: str | None, catalog: dict,
                     n_per_class: int, seed: int,
                     fixed_indices: dict[str, list[int]] | None):
    from datasets import load_dataset

    config = hf_config or catalog.get(agml_name, {}).get("preferred_config")
    repo_id = f"Project-AgML/{agml_name}"
    ds = load_dataset(repo_id, config, split="train")
    label_col = find_label_column(ds.features)
    class_names = resolve_class_names(ds, label_col)

    def label_fn(i):
        raw = ds[i][label_col]
        return class_names[raw] if class_names is not None and isinstance(raw, int) else raw

    if fixed_indices is not None:
        selected = fixed_indices
    else:
        selected = subsample_class_balanced(list(range(len(ds))), label_fn, n_per_class, seed)

    return repo_id, config, selected


def make_charts(datasets: list[DatasetConfig], manifest: dict, chart_dir: Path) -> None:
    """Summary figures for a download_and_sample run: how many datasets/classes/images
    ended up in the selection, and how it breaks down by task type."""
    import matplotlib.pyplot as plt

    chart_dir.mkdir(parents=True, exist_ok=True)
    ds_by_name = {d.agml_name: d for d in datasets}
    task_type_by_name = {d.agml_name: d.task_type for d in datasets}
    ds_entries = manifest["datasets"]
    failed = manifest.get("failed", {})

    if not ds_entries:
        print("  No datasets in manifest — skipping charts.")
        return

    # ── 1. Datasets per task type ───────────────────────────────────────────
    task_counts = Counter(task_type_by_name.get(name, "unknown") for name in ds_entries)
    task_types = sorted(task_counts, key=lambda t: -task_counts[t])
    colors = [TASK_COLOR.get(t, "#888888") for t in task_types]

    fig, ax = plt.subplots(figsize=(max(6, len(task_types) * 1.5), 5))
    bars = ax.bar(task_types, [task_counts[t] for t in task_types],
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, task_types):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(task_counts[t]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of datasets")
    title = f"Datasets by task type (n={len(ds_entries)} datasets)"
    if failed:
        title += f"  —  {len(failed)} datasets FAILED, not shown (see summary.txt)"
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "datasets_per_task_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 1b. Classes per EFFECTIVE task type (class_overrides applied) ───────
    # Same idea as chart 1, but counting individual classes rather than whole
    # datasets — a pest/damage class nested inside an otherwise-"disease"
    # dataset counts under pest_damage here, via
    # DatasetConfig.effective_task_type() (src/common/dataset_config.py).
    # Only successfully-downloaded classes counted, same as every other chart.
    class_task_counts = Counter()
    for name, entry in ds_entries.items():
        cfg = ds_by_name.get(name)
        for cls in entry["classes"]:
            eff = cfg.effective_task_type(cls) if cfg else task_type_by_name.get(name, "unknown")
            class_task_counts[eff] += 1

    class_task_types = sorted(class_task_counts, key=lambda t: -class_task_counts[t])
    class_colors = [TASK_COLOR.get(t, "#888888") for t in class_task_types]

    fig, ax = plt.subplots(figsize=(max(6, len(class_task_types) * 1.5), 5))
    bars = ax.bar(class_task_types, [class_task_counts[t] for t in class_task_types],
                   color=class_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, class_task_types):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(class_task_counts[t]), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of classes")
    title = f"Classes by effective task type, overrides applied (n={sum(class_task_counts.values())} classes)"
    if failed:
        title += f"  —  {len(failed)} datasets FAILED, not shown"
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "classes_per_task_type.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 2. Classes per dataset (sorted, long-tail style) ────────────────────
    n_classes = {name: len(entry["classes"]) for name, entry in ds_entries.items()}
    names = sorted(n_classes, key=lambda n: -n_classes[n])
    counts = [n_classes[n] for n in names]
    colors2 = [TASK_COLOR.get(task_type_by_name.get(n, "unknown"), "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 5))
    ax.bar(range(len(names)), counts, color=colors2, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(names)))
    fontsize = max(4, 8 - len(names) // 20)
    ax.set_xticklabels(names, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylabel("Number of classes")
    ax.set_title("Classes per dataset")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TASK_COLOR.values()]
    ax.legend(handles, TASK_COLOR.keys(), fontsize=8, loc="upper right")
    plt.tight_layout()
    path = chart_dir / "classes_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 3. Images selected per dataset (sorted, long-tail style) ────────────
    n_images = {name: sum(len(v) for v in entry["classes"].values()) for name, entry in ds_entries.items()}
    names_by_img = sorted(n_images, key=lambda n: -n_images[n])
    img_counts = [n_images[n] for n in names_by_img]
    colors3 = [TASK_COLOR.get(task_type_by_name.get(n, "unknown"), "#888888") for n in names_by_img]

    fig, ax = plt.subplots(figsize=(max(14, len(names_by_img) * 0.35), 5))
    ax.bar(range(len(names_by_img)), img_counts, color=colors3, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(names_by_img)))
    fontsize = max(4, 8 - len(names_by_img) // 20)
    ax.set_xticklabels(names_by_img, rotation=90, ha="right", fontsize=fontsize)
    ax.set_ylabel("Images selected")
    ax.set_title(f"Images selected per dataset (n_per_class cap = {manifest.get('n_per_class')})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in TASK_COLOR.values()]
    ax.legend(handles, TASK_COLOR.keys(), fontsize=8, loc="upper right")
    plt.tight_layout()
    path = chart_dir / "images_selected_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 4. Distribution of images-per-class (shows how many classes hit the cap) ──
    per_class_counts = [len(idxs) for entry in ds_entries.values() for idxs in entry["classes"].values()]
    cap = manifest.get("n_per_class")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(per_class_counts, bins=min(30, max(cap, 1) if cap else 30),
            color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.5)
    if cap:
        ax.axvline(cap, color="#C44E52", linestyle="--", linewidth=1.5, label=f"cap = {cap}")
        ax.legend()
    ax.set_xlabel("Images selected per class")
    ax.set_ylabel("Number of classes")
    n_at_cap = sum(1 for c in per_class_counts if cap and c >= cap)
    ax.set_title(f"Images-per-class distribution ({n_at_cap}/{len(per_class_counts)} classes at cap)")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    path = chart_dir / "images_per_class_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── 5. Task-type split: datasets vs classes (after class_overrides) ─────
    # Same underlying counts as charts 1/1b, just as a side-by-side pie
    # comparison instead of two separate bar charts. class_task_counts
    # computed once, up in chart 1b, and reused here.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, counts, title in (
        (axes[0], task_counts, f"Datasets by task type (n={len(ds_entries)})"),
        (axes[1], class_task_counts, f"Classes by EFFECTIVE task type, overrides applied (n={sum(class_task_counts.values())})"),
    ):
        types = sorted(counts, key=lambda t: -counts[t])
        vals = [counts[t] for t in types]
        cols = [TASK_COLOR.get(t, "#888888") for t in types]
        ax.pie(vals, labels=types, colors=cols, autopct=lambda p: f"{p:.0f}%\n({round(p * sum(vals) / 100)})",
               startangle=90, wedgeprops={"edgecolor": "black", "linewidth": 0.5})
        ax.set_title(title, fontsize=10)
    fig.suptitle("Only successfully-downloaded datasets/classes counted"
                 + (f" — {len(failed)} failed dataset(s) excluded" if failed else ""), fontsize=9, y=0.02)
    plt.tight_layout()
    path = chart_dir / "task_type_split_pie.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Summary ──────────────────────────────────────────────────────────
    total_classes = sum(n_classes.values())
    total_images = sum(n_images.values())
    summary = (
        f"Datasets: {len(ds_entries)} succeeded, {len(failed)} failed\n"
        f"Classes:  {total_classes}\n"
        f"Images:   {total_images:,}\n"
        f"n_per_class cap: {cap}\n"
        f"By task type (datasets): {dict(task_counts)}\n"
        f"By task type (classes, overrides applied): {dict(class_task_counts)}\n"
    )
    if failed:
        summary += f"\nFailed datasets ({len(failed)}) — re-run with --skip_existing to retry:\n"
        for name in sorted(failed):
            err = failed[name]
            msg = err.get("error", err) if isinstance(err, dict) else err  # dict since the
                                                                            # traceback-capture
                                                                            # change; str for
                                                                            # any older manifest
            summary += f"  - {name}: {msg}\n"
    (chart_dir / "summary.txt").write_text(summary)
    print(f"  Saved: {chart_dir / 'summary.txt'}")
    print(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/config/datasets.yaml")
    ap.add_argument("--catalog", default="src/config/hf_catalog.json")
    ap.add_argument("--n_per_class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="dataset/selected_images.json")
    ap.add_argument("--include_skipped", action="store_true")
    ap.add_argument("--only", default=None,
                     help="optional regex filter on agml_name — run against a single dataset "
                          "(or a subset) instead of the full active list. Same convention as "
                          "catalog_datasets.py's --only. Combine with --skip_existing so a "
                          "one-dataset-at-a-time wrapper script (e.g. under an OS-level `timeout`, "
                          "for hangs that never raise a catchable exception) still accumulates "
                          "into the same manifest across separate invocations.")
    ap.add_argument("--from_manifest", default=None,
                     help="reproduce indices from an existing manifest instead of re-sampling")
    ap.add_argument("--skip_existing", action="store_true",
                     help="resume: reuse datasets already recorded (successfully) in --output "
                          "instead of re-downloading/re-sampling them; only process what's "
                          "missing or previously failed. Ignored (with a warning) if --n_per_class "
                          "doesn't match the existing manifest's recorded value.")
    ap.add_argument("--chart_dir", default="outputs/agml_hf/",
                     help="where to write summary figures (datasets/classes/images counts)")
    ap.add_argument("--no_charts", action="store_true", help="skip chart generation")
    ap.add_argument("--plot_only", action="store_true",
                     help="skip download/sampling entirely; just regenerate charts from the "
                          "existing manifest at --output")
    args = ap.parse_args()

    catalog: dict = json.loads(Path(args.catalog).read_text()) if Path(args.catalog).exists() else {}

    if args.include_skipped:
        from src.common.dataset_config import load_datasets
        datasets = load_datasets(Path(args.config))
    else:
        datasets = active_datasets(Path(args.config))

    if args.only:
        pattern = re.compile(args.only)
        datasets = [d for d in datasets if pattern.search(d.agml_name)]
        if not datasets:
            print(f"ERROR: --only {args.only!r} matched no active datasets.")
            sys.exit(1)

    if args.plot_only:
        out_path = Path(args.output)
        if not out_path.exists():
            print(f"ERROR: --plot_only given but {out_path} does not exist.")
            sys.exit(1)
        manifest = json.loads(out_path.read_text())
        make_charts(datasets, manifest, Path(args.chart_dir))
        return

    prior_manifest = json.loads(Path(args.from_manifest).read_text()) if args.from_manifest else None

    existing_manifest = None
    out_path = Path(args.output)
    if args.skip_existing and out_path.exists():
        existing_manifest = json.loads(out_path.read_text())
        if existing_manifest.get("n_per_class") != args.n_per_class:
            print(f"WARNING: --skip_existing requested but existing manifest was built with "
                  f"n_per_class={existing_manifest.get('n_per_class')} != requested "
                  f"{args.n_per_class} — every dataset's sample would be stale under the new "
                  f"cap, so ignoring --skip_existing and reprocessing everything.")
            existing_manifest = None

    manifest = {
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduced_from": args.from_manifest,
        "datasets": {},
        "failed": {},
    }
    if existing_manifest:
        manifest["datasets"] = dict(existing_manifest.get("datasets", {}))

    t0 = time.time()
    n_ok, n_err, n_skipped = 0, 0, 0
    for i, d in enumerate(datasets, 1):
        if existing_manifest and d.agml_name in manifest["datasets"]:
            print(f"[{i}/{len(datasets)}] {d.agml_name} — already done, skipping")
            n_skipped += 1
            continue

        print(f"[{i}/{len(datasets)}] {d.agml_name}", end=" ", flush=True)
        fixed = None
        if prior_manifest and d.agml_name in prior_manifest["datasets"]:
            fixed = prior_manifest["datasets"][d.agml_name]["classes"]
        try:
            repo_id, config, selected = with_retry(
                process_dataset, d.agml_name, d.hf_config, catalog, args.n_per_class, args.seed, fixed
            )
            manifest["datasets"][d.agml_name] = {
                "hf_repo_id": repo_id,
                "hf_config": config,
                "classes": selected,
            }
            manifest["failed"].pop(d.agml_name, None)
            n_total = sum(len(v) for v in selected.values())
            print(f"ok ({len(selected)} classes, {n_total} images)")
            n_ok += 1
        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR: {e}")
            print(tb)  # full traceback in the job log, not just str(e) — a bug like
                       # "bad operand type for abs(): 'str'" is unlocatable from the
                       # message alone; str/dict-shape kept backward compatible below,
                       # existing readers only ever do sorted(manifest["failed"]) (keys)
            manifest["failed"][d.agml_name] = {"error": str(e), "traceback": tb}
            n_err += 1

        if i % 5 == 0 or i == len(datasets):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    elapsed = time.time() - t0
    total_images = sum(
        len(idxs) for ds in manifest["datasets"].values() for idxs in ds["classes"].values()
    )
    print(f"\nDone in {elapsed:.0f}s. {n_ok} datasets ok, {n_skipped} skipped (already done), "
          f"{n_err} errors -> {args.output}")
    if manifest["failed"]:
        print(f"Failed datasets (re-run with --skip_existing to retry just these): "
              f"{sorted(manifest['failed'])}")
    print(f"Total images selected: {total_images:,}")

    if not args.no_charts:
        print(f"\n=== Generating summary charts -> {args.chart_dir} ===")
        make_charts(datasets, manifest, Path(args.chart_dir))


if __name__ == "__main__":
    main()
