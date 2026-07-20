"""
Track A — Multi-layer Cosine Alignment Experiment (desc_aug)

For each (image, desc_aug prediction) pair in our verifier subsample (~1,800
images), computes:

  φ_vision     = mean-pooled post-projector vision embedding
                 (same hook as probe.py → embed_vision → early-exit)
  ψ_lang_L{n}  = mean-pooled hidden states at layer n over prediction tokens

  align_score_L{n} = cos_sim(φ_vision, ψ_lang_L{n})   ∈ [-1, 1]

Probes multiple layer depths (default: 10, 20, 30, last) and produces:
  - per-layer box plots and per-dataset scatters
  - summary bar chart of point-biserial r vs layer depth

Reuses from probe.py: load_gemma4, resolve_mm_root, resolve_projector

Usage:
  uv run python src/alignment_belief.py \\
    --model google/gemma-4-26B-A4B-it \\
    --output_dir outputs/alignment_belief/

  # Resume from checkpoints:
  uv run python src/alignment_belief.py ... --from_ckpt

  # Plot only:
  uv run python src/alignment_belief.py ... --plot_only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Probe helpers (reused as-is) ──────────────────────────────────────────────

from src.linear_probe.probe import (
    load_gemma4,
    resolve_mm_root,
    resolve_projector,
)

# ── cot_vision helpers ────────────────────────────────────────────────────────

from src.cot_vision import (
    _sys_msg,
    _cls_question,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DESC_AUG_CSV       = "outputs/desc_augmented_vision/aug_cls_all_google_gemma-4-26B-A4B-it.csv"
VERIFIER_CKPT      = "outputs/verifier_vision/ckpt_candidates.csv"
DATASETS_YAML      = "src/config/datasets.yaml"

CKPT_ITEMS        = "ckpt_items.csv"
CKPT_PHI_VISION   = "ckpt_phi_vision.npz"   # {image_key → φ_vision vector}
CKPT_SCORED       = "ckpt_scored.csv"        # legacy single-layer run
CKPT_SCORED_MULTI = "ckpt_scored_layers.csv" # multi-layer run

DEFAULT_FRACS = [0.25, 0.5, 0.75, 1.0]  # fractions of total depth, resolved at runtime

BASE_FIELDS = [
    "agml_dataset", "image_name", "image_path",
    "task_type", "crop", "class_label",
    "prediction", "judged_correct",
]


def _frac_col(frac: float) -> str:
    return f"align_score_pct{int(round(frac * 100))}"


def _scored_fields(fracs: list[float]) -> list[str]:
    return BASE_FIELDS + [_frac_col(f) for f in fracs]

# ── CSV helpers ───────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path} ({len(rows)} rows)")


# ── Step 1 — Build evaluation items ──────────────────────────────────────────

def build_items(
    desc_aug_csv: Path,
    verifier_ckpt: Path,
    datasets_yaml: Path,
    output_path: Path,
) -> list[dict]:
    """
    Join desc_aug predictions with image paths from the verifier subsample.

    The verifier ckpt_candidates.csv already selected the right 50-per-class
    subset (same test_split, same 9 datasets). We use it as the image source
    and join to desc_aug on (agml_dataset, image_name).
    """
    import yaml

    with open(datasets_yaml) as f:
        cfg = yaml.safe_load(f)
    ds_info = {d["agml_name"]: d for d in cfg["datasets"]}

    # Build image_path lookup from verifier candidates
    # (one entry per image, just take the first occurrence)
    path_lookup: dict[tuple[str, str], str] = {}
    task_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for row in _read_csv(verifier_ckpt):
        key = (row["agml_dataset"], row["image_name"])
        if key not in path_lookup:
            path_lookup[key] = row["image_path"]
            task_lookup[key] = (row["task_type"], row.get("crop", ""))

    print(f"Verifier subsample: {len(path_lookup)} unique images across "
          f"{len(set(k[0] for k in path_lookup))} datasets")

    # Load desc_aug rows and join
    items: list[dict] = []
    missing = 0
    for row in _read_csv(desc_aug_csv):
        key = (row["agml_dataset"], row["image_name"])
        if key not in path_lookup:
            missing += 1
            continue
        task_type, crop = task_lookup[key]
        items.append({
            "agml_dataset":  row["agml_dataset"],
            "image_name":    row["image_name"],
            "image_path":    path_lookup[key],
            "task_type":     task_type,
            "crop":          crop,
            "class_label":   row["class_label"],
            "prediction":    row["prediction"],
            "judged_correct": row["judged_correct"],
        })

    print(f"Matched: {len(items)} rows  |  unmatched: {missing}")

    # Deduplicate: desc_aug may have one row per image; keep that
    seen: set[tuple] = set()
    deduped = []
    for r in items:
        k = (r["agml_dataset"], r["image_name"])
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    print(f"After dedup: {len(deduped)} unique (image, prediction) pairs")

    _write_csv(deduped, output_path, list(deduped[0].keys()))
    return deduped


# ── Step 2a — φ_vision: post-projector vision embeddings ─────────────────────

def extract_phi_vision(
    items: list[dict],
    processor,
    model,
    mm_root,
    ckpt_path: Path,
) -> dict[str, np.ndarray]:
    """
    Return {image_key: φ_vision} for all unique images in items.

    Hooks the embed_vision projector (same approach as probe.py) with an early
    exit so we never run the full language model. Mean-pools over visual tokens.
    Saves to ckpt_phi_vision.npz keyed by "dataset/image_name" for resumability.
    """
    from PIL import Image as PILImage
    from tqdm import tqdm

    # Check our own per-image cache first
    image_key_fn = lambda r: f"{r['agml_dataset']}/{r['image_name']}"

    if ckpt_path.exists():
        data = np.load(str(ckpt_path), allow_pickle=True)
        cached = {k: data[k] for k in data.files}
        keys_needed = {image_key_fn(r) for r in items}
        if keys_needed.issubset(cached.keys()):
            print(f"  φ_vision: loaded all {len(keys_needed)} from {ckpt_path.name}")
            return {k: cached[k] for k in keys_needed}
        already = keys_needed & cached.keys()
        print(f"  φ_vision: {len(already)} cached, "
              f"{len(keys_needed) - len(already)} to extract")
        phi_vision = dict(cached)
    else:
        phi_vision: dict[str, np.ndarray] = {}

    # Resolve projector (same as probe.py)
    proj_module, proj_name = resolve_projector(mm_root)
    if proj_module is None:
        raise RuntimeError("Could not resolve projector module in model")
    print(f"  Projector: {proj_name}")

    from src.linear_probe.probe import _EarlyExit

    # Extract for images not yet cached
    todo = [r for r in items if image_key_fn(r) not in phi_vision]
    # Deduplicate by image key (same image may appear multiple times in items)
    seen: set[str] = set()
    unique_todo = []
    for r in todo:
        k = image_key_fn(r)
        if k not in seen:
            seen.add(k)
            unique_todo.append(r)

    n_todo = len(unique_todo)
    print(f"  Extracting φ_vision for {n_todo} images...", flush=True)
    model_device = next(model.parameters()).device
    log_every = max(1, n_todo // 20)   # print ~20 updates

    for i, row in enumerate(tqdm(unique_todo, desc="φ_vision (post-proj)", unit="img")):
        if i % log_every == 0 or i == n_todo - 1:
            print(f"  [φ_vision] {i+1}/{n_todo}  {row['agml_dataset']}", flush=True)

        image = PILImage.open(row["image_path"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": "."},
        ]}]
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            inputs = processor(text="<image>", images=[image], return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        captured: list[torch.Tensor] = []

        def _hook(module, input, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if isinstance(t, torch.Tensor):
                captured.append(t.detach().float().cpu())
            raise _EarlyExit

        handle = proj_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(**inputs)
        except _EarlyExit:
            pass
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError(f"Projector hook captured nothing for {row['image_path']}")

        feat = captured[0]
        if feat.dim() == 3:
            feat = feat[0]              # (batch, seq, dim) → (seq, dim)
        phi_vision[image_key_fn(row)] = feat.mean(dim=0).numpy()

    # Save per-image cache
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(ckpt_path), **phi_vision)
    print(f"  Saved φ_vision cache: {ckpt_path.name} ({len(phi_vision)} images)")

    return phi_vision


# ── Steps 2b + 3 — ψ_lang extraction + cosine similarity (inline) ────────────

def extract_psi_lang_and_score(
    items: list[dict],
    phi_vision: dict[str, np.ndarray],
    processor,
    model,
    batch_size: int,
    ckpt_path: Path,
    layer_fracs: list[float],
) -> tuple[list[dict], list[float]]:
    """
    For each (image, prediction) row:
      1. Run a full forward pass with output_hidden_states=True.
      2. For each fraction f, resolve to layer index int(round(f*(n_layers-1)))
         and extract mean-pooled hidden states over prediction tokens.
      3. Compute cos_sim(φ_vision, ψ_lang) inline per fraction.
      4. Append to ckpt_scored_layers.csv after every batch.

    Column names are align_score_pct{int(f*100)} — model-agnostic, so the same
    fractions are directly comparable across different model depths.
    Returns (rows, fracs_used).

    Progress: watch  wc -l <output_dir>/ckpt_scored_layers.csv
    """
    from PIL import Image as PILImage
    from tqdm import tqdm

    # fracs used in this run; inferred from existing CSV on resume
    fracs_used: list[float] = []

    # Resume: skip rows already scored
    done_keys: set[tuple[str, str]] = set()
    rows_out: list[dict] = []
    if ckpt_path.exists():
        existing = _read_csv(ckpt_path)
        if existing:
            # Infer fractions from column names (align_score_pct25 → 0.25)
            fracs_used = [
                int(c[len("align_score_pct"):]) / 100
                for c in existing[0].keys()
                if c.startswith("align_score_pct")
            ]
            for r in existing:
                done_keys.add((r["agml_dataset"], r["image_name"]))
            rows_out.extend(existing)
            print(f"  Resuming — {len(done_keys)}/{len(items)} already scored  "
                  f"(fracs: {fracs_used})")

    todo = [r for r in items
            if (r["agml_dataset"], r["image_name"]) not in done_keys]
    if not todo:
        print("  All items already scored.")
        return rows_out, fracs_used

    n_total = len(items)
    n_done_start = len(done_keys)
    model_device = next(model.parameters()).device
    n_batches = (len(todo) + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 20)

    for b_idx, start in enumerate(tqdm(range(0, len(todo), batch_size),
                                       total=n_batches,
                                       desc="ψ_lang + align_score", unit="batch")):
        if b_idx % log_every == 0 or b_idx == n_batches - 1:
            n_scored_so_far = n_done_start + b_idx * batch_size
            print(f"  [ψ_lang] batch {b_idx+1}/{n_batches}  "
                  f"({min(n_scored_so_far, n_total)}/{n_total} images)", flush=True)

        batch = todo[start:start + batch_size]
        batch_rows: list[dict] = []

        for row in batch:
            img_key = f"{row['agml_dataset']}/{row['image_name']}"
            if img_key not in phi_vision:
                continue

            image = PILImage.open(row["image_path"]).convert("RGB")
            task_type  = row["task_type"]
            crop       = row["crop"] or None
            prediction = row["prediction"]

            prefix_msgs = _sys_msg() + [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": _cls_question(task_type, crop)},
            ]}]
            prefix_text = processor.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=True
            )
            full_msgs = prefix_msgs + [{"role": "assistant", "content": prediction}]
            full_text = processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )

            try:
                prefix_inputs = processor(
                    text=[prefix_text], images=[image], return_tensors="pt"
                )
                full_inputs = processor(
                    text=[full_text], images=[image], return_tensors="pt"
                )
            except Exception as e:
                print(f"  [warn] processor error for {row['image_name']}: {e}")
                continue

            n_prefix = prefix_inputs["input_ids"].shape[1]
            n_full   = full_inputs["input_ids"].shape[1]
            n_pred   = max(1, n_full - n_prefix - 1)

            full_inputs_gpu = {k: v.to(model_device) for k, v in full_inputs.items()}
            with torch.no_grad():
                outputs = model(**full_inputs_gpu, output_hidden_states=True)

            # Resolve fractions to absolute layer indices once on the first image
            if not fracs_used:
                n_layers = len(outputs.hidden_states)
                fracs_used = list(layer_fracs)
                frac_to_idx = {
                    f: max(0, min(int(round(f * (n_layers - 1))), n_layers - 1))
                    for f in fracs_used
                }
                print(f"  Model has {n_layers} hidden states; "
                      f"fracs → layers: { {f: frac_to_idx[f] for f in fracs_used} }",
                      flush=True)

            phi = torch.tensor(phi_vision[img_key], dtype=torch.float32)
            out_row: dict = {**row}
            for frac in fracs_used:
                li = frac_to_idx[frac]
                hidden = outputs.hidden_states[li]   # (1, seq_len, hidden_dim)
                psi = hidden[0, n_prefix:n_prefix + n_pred, :].float().mean(dim=0).cpu()
                out_row[_frac_col(frac)] = float(
                    F.cosine_similarity(phi.unsqueeze(0), psi.unsqueeze(0))
                )

            batch_rows.append(out_row)

        rows_out.extend(batch_rows)

        if batch_rows:
            fields = _scored_fields(fracs_used)
            write_header = not ckpt_path.exists()
            with open(ckpt_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerows(batch_rows)

    return rows_out, fracs_used


# ── Step 4 — Plots ────────────────────────────────────────────────────────────

def plot_results(scored: list[dict], output_dir: Path, fracs_used: list[float]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats

    output_dir.mkdir(parents=True, exist_ok=True)
    correct = np.array([int(r["judged_correct"]) for r in scored])

    frac_rs: list[float] = []
    frac_ps: list[float] = []
    pct_labels: list[str] = []

    for frac in fracs_used:
        col = _frac_col(frac)
        pct = int(round(frac * 100))
        scores = np.array([float(r[col]) for r in scored])

        # ── Box plot ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 5))
        data = [scores[correct == 1], scores[correct == 0]]
        bp = ax.boxplot(data, labels=["Correct", "Incorrect"],
                        patch_artist=True, medianprops={"color": "black", "lw": 2})
        bp["boxes"][0].set_facecolor("#2ca02c")
        bp["boxes"][1].set_facecolor("#d62728")
        ax.set_ylabel(f"cos(φ_vision, ψ_lang @ {pct}% depth)")
        ax.set_title(f"Alignment Score vs Correctness — {pct}% depth (desc_aug)")
        ax.yaxis.grid(True, alpha=0.3)
        r, p = stats.pointbiserialr(correct, scores)
        ax.text(0.97, 0.03, f"r = {r:.3f}  p = {p:.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        plt.tight_layout()
        fig.savefig(output_dir / f"align_boxplot_pct{pct}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {pct}% depth — point-biserial r = {r:.4f}, p = {p:.4g}")

        frac_rs.append(r)
        frac_ps.append(p)
        pct_labels.append(f"{pct}%")

        # ── Per-dataset scatter ────────────────────────────────────────────
        by_ds: dict[str, dict] = defaultdict(lambda: {"scores": [], "correct": []})
        for row in scored:
            by_ds[row["agml_dataset"]]["scores"].append(float(row[col]))
            by_ds[row["agml_dataset"]]["correct"].append(int(row["judged_correct"]))

        ds_names = sorted(by_ds)
        ds_align = [np.mean(by_ds[d]["scores"])  for d in ds_names]
        ds_acc   = [np.mean(by_ds[d]["correct"]) for d in ds_names]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(ds_align, ds_acc, s=100, zorder=3)
        for i, ds in enumerate(ds_names):
            short = ds.replace("_classification", "").replace("_", " ")
            ax.annotate(short, (ds_align[i], ds_acc[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
        r_ds, p_ds = stats.pearsonr(ds_align, ds_acc)
        ax.set_xlabel(f"Mean alignment score ({pct}% depth)")
        ax.set_ylabel("Mean accuracy (judged_correct)")
        ax.set_title(f"Per-dataset alignment vs accuracy — {pct}% depth (desc_aug)")
        ax.text(0.97, 0.03, f"r = {r_ds:.3f}  p = {p_ds:.3g}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"align_per_dataset_pct{pct}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary: r vs layer depth ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ca02c" if p < 0.05 else "#aec7e8" for p in frac_ps]
    ax.bar(pct_labels, frac_rs, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer depth (% of total)")
    ax.set_ylabel("Point-biserial r  (align_score vs judged_correct)")
    ax.set_title("Vision-language alignment signal by layer depth (desc_aug)")
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(handles=[
        Patch(color="#2ca02c", label="p < 0.05"),
        Patch(color="#aec7e8", label="p ≥ 0.05"),
    ], fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "align_r_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots saved to {output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track A cosine alignment experiment (desc_aug)"
    )
    parser.add_argument("--model",       required=True, help="HF model ID or local path")
    parser.add_argument("--output_dir",  default="outputs/alignment_belief/")
    parser.add_argument("--desc_aug_csv", default=DESC_AUG_CSV)
    parser.add_argument("--verifier_ckpt", default=VERIFIER_CKPT,
                        help="ckpt_candidates.csv from verifier_vision (image path source)")
    parser.add_argument("--config",      default=DATASETS_YAML)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--quantize",     default=None, choices=["4bit", "8bit"])
    parser.add_argument("--probe_fracs", default=",".join(str(f) for f in DEFAULT_FRACS),
                        help="Comma-separated fractions of model depth to probe (0.0–1.0). "
                             f"Default: {DEFAULT_FRACS}")
    parser.add_argument("--from_ckpt",   action="store_true")
    parser.add_argument("--plot_only",   action="store_true")
    args = parser.parse_args()

    layer_fracs = [float(x) for x in args.probe_fracs.split(",")]

    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_items      = output_dir / CKPT_ITEMS
    ckpt_phi_vision = output_dir / CKPT_PHI_VISION
    ckpt_scored     = output_dir / CKPT_SCORED_MULTI

    # ── Plot only ──────────────────────────────────────────────────────────
    if args.plot_only:
        if not ckpt_scored.exists():
            raise FileNotFoundError(f"No scored CSV at {ckpt_scored}")
        scored = _read_csv(ckpt_scored)
        fracs_used = [
            int(c[len("align_score_pct"):]) / 100
            for c in scored[0].keys() if c.startswith("align_score_pct")
        ]
        scored = [r for r in scored if all(
            r.get(_frac_col(f)) not in ("", None) for f in fracs_used
        )]
        plot_results(scored, output_dir, fracs_used)
        return

    # ── Step 1 — items ─────────────────────────────────────────────────────
    if args.from_ckpt and ckpt_items.exists():
        print("\n=== Step 1: items [SKIPPED — loading from ckpt] ===")
        items = _read_csv(ckpt_items)
    else:
        print("\n=== Step 1: Build evaluation items ===")
        items = build_items(
            Path(args.desc_aug_csv),
            Path(args.verifier_ckpt),
            Path(args.config),
            ckpt_items,
        )

    print(f"Items: {len(items)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\n=== Loading model: {args.model} ===")
    processor, model, mm_root = load_gemma4(args.model, device="cuda", quantize=args.quantize)

    # ── Step 2a — φ_vision ─────────────────────────────────────────────────
    print("\n=== Step 2a: φ_vision (post-projector embeddings) ===")
    phi_vision = extract_phi_vision(
        items, processor, model, mm_root,
        ckpt_path=ckpt_phi_vision,
    )

    # ── Steps 2b + 3 — ψ_lang + align_score per layer ─────────────────────
    print(f"\n=== Steps 2b+3: ψ_lang + cosine scores (fracs {layer_fracs}) ===")
    print(f"  Progress: wc -l {ckpt_scored}  (1 row = 1 image, target {len(items)})")
    scored, fracs_used = extract_psi_lang_and_score(
        items, phi_vision, processor, model,
        batch_size=args.batch_size,
        ckpt_path=ckpt_scored,
        layer_fracs=layer_fracs,
    )
    print(f"Scored: {len(scored)} rows  (fracs: {fracs_used})")

    # ── Step 4 — Plots ─────────────────────────────────────────────────────
    print("\n=== Step 4: Plots ===")
    plot_results(scored, output_dir, fracs_used)


if __name__ == "__main__":
    main()
