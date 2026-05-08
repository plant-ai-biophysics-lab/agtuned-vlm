"""
AgriCoT Dataset Analysis
========================
Loads wenyb/AgriCoT from HuggingFace and produces:
  - Dataset metadata summary
  - Sample items (formatted)
  - Distributions: task dimension, sub-dimension, question type, answer, source
  - Text length stats: question length, reasoning length, number of options
  - All charts saved to --output_dir

The metadata (question/answer/reasoning) lives in VQRA.json on the Hub.
Images are not needed for this analysis and are skipped entirely.

Usage:
    python agricot.py
    python agricot.py --n_samples 500 --output_dir agricot_analysis
    python agricot.py --show_samples 5
"""

import argparse
import json
import os
import textwrap
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download


# ── Label maps (from README) ──────────────────────────────────────────────────

DIMENSION_LABELS = {
    1: "Object Detection",
    2: "Quantitative Analysis",
    3: "Disease Monitoring",
    4: "Spatial Understanding",
    5: "Environmental Management",
}

SUB_DIMENSION_LABELS: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def dim_label(did):
    return DIMENSION_LABELS.get(did, f"Dim {did}")


def subdim_label(sid):
    return SUB_DIMENSION_LABELS.get(sid, f"Sub-dim {sid}")


def count_options(options_field) -> int:
    if options_field is None:
        return 0
    if isinstance(options_field, dict):
        return len(options_field)
    if isinstance(options_field, (list, tuple)):
        return len(options_field)
    if isinstance(options_field, str):
        try:
            parsed = json.loads(options_field)
            return len(parsed) if isinstance(parsed, (dict, list)) else 0
        except Exception:
            return 0
    return 0


def format_options(options_field) -> str:
    if options_field is None:
        return "(no options)"
    if isinstance(options_field, dict):
        d = options_field
    elif isinstance(options_field, list):
        return "  ".join(str(x) for x in options_field)
    elif isinstance(options_field, str):
        try:
            d = json.loads(options_field)
            if isinstance(d, list):
                return "  ".join(str(x) for x in d)
        except Exception:
            return options_field
    else:
        return str(options_field)
    return "  ".join(f"{k}: {v}" for k, v in sorted(d.items()))


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ── Load ──────────────────────────────────────────────────────────────────────

def _find_json_records(zf) -> list[dict]:
    """
    Scan a ZipFile for JSON files containing a list of dicts with
    question/answer/reasoning fields. Returns the first match.
    """
    import zipfile

    names = zf.namelist()
    print(f"  ZIP contains {len(names)} files. Scanning for JSON metadata ...")

    # Prefer files whose name suggests QA data
    priority = sorted(names, key=lambda n: (
        0 if any(k in n.lower() for k in ("vqra", "qa", "anno", "data", "meta")) else 1,
        not n.endswith(".json"),
        n,
    ))

    for name in priority:
        if not name.endswith(".json"):
            continue
        info = zf.getinfo(name)
        if info.file_size > 500 * 1024 * 1024:  # skip files >500 MB
            print(f"  Skipping {name} (too large: {info.file_size // 1_000_000} MB)")
            continue
        print(f"  Trying: {name} ({info.file_size // 1_000} KB) ...")
        try:
            with zf.open(name) as f:
                data = json.load(f)
        except Exception as e:
            print(f"    Could not parse: {e}")
            continue

        # Unwrap common envelope formats
        if isinstance(data, dict):
            for key in ("data", "samples", "annotations", "items"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = list(data.values())[0] if len(data) == 1 else None

        if not isinstance(data, list) or not data:
            continue

        # Validate: first item should look like a QA record
        first = data[0]
        if isinstance(first, dict) and any(
            k in first for k in ("question", "answer", "reasoning", "item_id")
        ):
            print(f"  Found {len(data):,} records in {name}")
            return data

    raise RuntimeError(
        "No suitable JSON file found inside the ZIP. "
        "ZIP contents: " + ", ".join(names[:20])
    )


def load_records(n_samples: int | None) -> list[dict]:
    """
    Downloads AgriCoT_2.zip from the Hub (337 MB, cached after first run),
    extracts the JSON metadata, and returns a list of records.
    Falls back to AgriCoT.zip if needed.
    """
    import zipfile

    for filename in ("AgriCoT_2.zip", "AgriCoT.zip"):
        print(f"Downloading {filename} from wenyb/AgriCoT ...")
        try:
            zip_path = hf_hub_download(
                repo_id="wenyb/AgriCoT",
                filename=filename,
                repo_type="dataset",
            )
            print(f"  Cached at: {zip_path}")
        except Exception as e:
            print(f"  Could not download {filename}: {e}")
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                records = _find_json_records(zf)
            break
        except RuntimeError as e:
            print(f"  {e} — trying next archive ...")
            continue
    else:
        raise RuntimeError(
            "Could not load records from any AgriCoT ZIP archive.\n"
            "Try: huggingface-cli login  (if the dataset is gated)"
        )

    if n_samples and n_samples < len(records):
        records = records[:n_samples]
        print(f"  Subsampled to {len(records):,}")

    return records


# ── Sample printer ────────────────────────────────────────────────────────────

def print_samples(records: list[dict], n: int = 3):
    print(f"\n{'='*70}")
    print(f"  SAMPLE ITEMS  (showing {n})")
    print(f"{'='*70}")
    for i, row in enumerate(records[:n]):
        print(f"\n─── Sample #{row.get('item_id', i)} ───────────────────────────────")
        print(f"  Dim:        {dim_label(row.get('dimension_id'))}  "
              f"| Sub-dim: {subdim_label(row.get('sub_dimension_id'))}  "
              f"| Type: {row.get('type_id')}")
        print(f"  Image:      {row.get('image_path', 'N/A')}")
        print(f"  Question:")
        print(wrap(row.get("question", ""), indent="    "))
        opts = format_options(row.get("options"))
        if opts and opts != "(no options)":
            print(f"  Options:    {opts}")
        print(f"  Answer:     {row.get('answer', 'N/A')}")
        reasoning = str(row.get("reasoning", ""))
        short = reasoning[:400] + " [...]" if len(reasoning) > 400 else reasoning
        print(f"  Reasoning:")
        print(wrap(short, indent="    "))
    print()


# ── Chart helpers ─────────────────────────────────────────────────────────────

def bar_chart(counts: dict, title: str, xlabel: str, path: str,
              color: str = "steelblue", rotate: int = 0):
    labels = list(counts.keys())
    values = list(counts.values())
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9 + 1), 5))
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{v}\n({100*v/total:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    if rotate:
        ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=9)
    ax.set_ylim(0, max(values) * 1.18)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def hist_chart(values: list, title: str, xlabel: str, path: str,
               bins: int = 30, color: str = "teal"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(values), color="red", linestyle="--",
               linewidth=1.2, label=f"mean={np.mean(values):.0f}")
    ax.axvline(np.median(values), color="orange", linestyle="--",
               linewidth=1.2, label=f"median={np.median(values):.0f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def heatmap_chart(matrix: dict, title: str, path: str):
    row_keys = sorted({r for r, _ in matrix})
    col_keys = sorted({c for _, c in matrix})
    data = np.array([[matrix.get((r, c), 0) for c in col_keys] for r in row_keys])

    fig, ax = plt.subplots(figsize=(max(6, len(col_keys) * 0.8 + 1),
                                    max(4, len(row_keys) * 0.6 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(row_keys, fontsize=8)
    plt.colorbar(im, ax=ax, label="Count")
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            v = data[i, j]
            if v > 0:
                ax.text(j, i, str(int(v)), ha="center", va="center",
                        fontsize=7,
                        color="black" if v < data.max() * 0.7 else "white")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze the AgriCoT HuggingFace dataset")
    parser.add_argument("--output_dir",   type=str, default="agricot_analysis")
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="Limit to first N records (default: all ~4800)")
    parser.add_argument("--show_samples", type=int, default=3,
                        help="Number of sample items to print (default: 3)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.n_samples)

    # Print actual keys found so we know the real schema
    if records:
        print(f"  Keys in first record: {list(records[0].keys())}")

    print_samples(records, n=args.show_samples)

    # ── Collect distributions ──────────────────────────────────────────────────
    dim_counts:    Counter = Counter()
    subdim_counts: Counter = Counter()
    type_counts:   Counter = Counter()
    answer_counts: Counter = Counter()
    option_counts: Counter = Counter()
    dim_subdim:    dict    = defaultdict(int)

    q_lengths: list = []
    r_lengths: list = []

    for row in records:
        did  = row.get("dimension_id")
        sid  = row.get("sub_dimension_id")
        tid  = row.get("type_id")
        ans  = str(row.get("answer", "?")).strip().upper()
        opts = row.get("options")
        q    = str(row.get("question",  ""))
        r    = str(row.get("reasoning", ""))

        dim_counts[dim_label(did)]       += 1
        subdim_counts[subdim_label(sid)] += 1
        type_counts[f"Type {tid}"]       += 1
        answer_counts[ans]               += 1
        option_counts[count_options(opts)] += 1
        dim_subdim[(dim_label(did), subdim_label(sid))] += 1

        q_lengths.append(len(q))
        r_lengths.append(len(r))

    # ── Metadata summary ──────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("  METADATA SUMMARY")
    print(f"{'='*70}")
    print(f"  Total samples       : {len(records):,}")
    print(f"  Task dimensions     : {len(dim_counts)}")
    print(f"  Sub-dimensions      : {len(subdim_counts)}")
    print(f"  Question types      : {len(type_counts)}")
    print(f"  Unique answers      : {len(answer_counts)}")
    print(f"\n  Question length     : mean={np.mean(q_lengths):.0f}  "
          f"median={np.median(q_lengths):.0f}  "
          f"max={max(q_lengths)}  min={min(q_lengths)}")
    print(f"  Reasoning length    : mean={np.mean(r_lengths):.0f}  "
          f"median={np.median(r_lengths):.0f}  "
          f"max={max(r_lengths)}  min={min(r_lengths)}")

    print(f"\n  Dimension breakdown:")
    for k, v in sorted(dim_counts.items(), key=lambda x: -x[1]):
        print(f"    {k:<30} {v:>5}  ({100*v/len(records):.1f}%)")

    print(f"\n  Answer distribution:")
    for k, v in sorted(answer_counts.items()):
        print(f"    {k:<6} {v:>5}  ({100*v/len(records):.1f}%)")

    print(f"\n  Options per question:")
    for k, v in sorted(option_counts.items()):
        print(f"    {k} options: {v:>5}  ({100*v/len(records):.1f}%)")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "n_samples":           len(records),
        "dimension_counts":    dict(dim_counts),
        "subdimension_counts": dict(subdim_counts),
        "type_counts":         dict(type_counts),
        "answer_counts":       dict(answer_counts),
        "option_counts":       {str(k): v for k, v in option_counts.items()},
        "question_length": {
            "mean":   round(float(np.mean(q_lengths)), 1),
            "median": float(np.median(q_lengths)),
            "max":    int(max(q_lengths)),
            "min":    int(min(q_lengths)),
            "p90":    float(np.percentile(q_lengths, 90)),
        },
        "reasoning_length": {
            "mean":   round(float(np.mean(r_lengths)), 1),
            "median": float(np.median(r_lengths)),
            "max":    int(max(r_lengths)),
            "min":    int(min(r_lengths)),
            "p90":    float(np.percentile(r_lengths, 90)),
        },
    }
    summary_path = os.path.join(args.output_dir, "agricot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON: {summary_path}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GENERATING CHARTS")
    print(f"{'='*70}")

    bar_chart(
        dict(sorted(dim_counts.items(), key=lambda x: -x[1])),
        title="AgriCoT — Samples by Task Dimension",
        xlabel="Dimension",
        path=os.path.join(args.output_dir, "dist_dimension.png"),
        color="steelblue", rotate=15,
    )

    bar_chart(
        dict(sorted(subdim_counts.items(), key=lambda x: -x[1])),
        title="AgriCoT — Samples by Sub-Dimension",
        xlabel="Sub-Dimension",
        path=os.path.join(args.output_dir, "dist_subdimension.png"),
        color="darkorange", rotate=30,
    )

    bar_chart(
        dict(sorted(type_counts.items())),
        title="AgriCoT — Samples by Question Type",
        xlabel="Question Type",
        path=os.path.join(args.output_dir, "dist_type.png"),
        color="mediumseagreen",
    )

    bar_chart(
        dict(sorted(answer_counts.items())),
        title="AgriCoT — Answer Label Distribution",
        xlabel="Answer",
        path=os.path.join(args.output_dir, "dist_answer.png"),
        color="mediumpurple",
    )

    hist_chart(
        q_lengths,
        title="AgriCoT — Question Length Distribution (chars)",
        xlabel="Characters",
        path=os.path.join(args.output_dir, "hist_question_length.png"),
        color="steelblue",
    )

    hist_chart(
        r_lengths,
        title="AgriCoT — Reasoning Length Distribution (chars)",
        xlabel="Characters",
        path=os.path.join(args.output_dir, "hist_reasoning_length.png"),
        color="teal",
    )

    heatmap_chart(
        dim_subdim,
        title="AgriCoT — Dimension × Sub-Dimension Co-occurrence",
        path=os.path.join(args.output_dir, "heatmap_dim_subdim.png"),
    )

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
