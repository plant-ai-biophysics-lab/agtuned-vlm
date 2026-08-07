"""
Pre-check: does each candidate model expose a vision tower this pipeline can
actually hook and extract from — before spending time running the full
linear-probe pipeline (dataset loading, per-image extraction across every
active dataset, probe training) on a model that turns out to be architecturally
incompatible.

Reuses probe.py's own model-loading and vision-encoder-resolution logic
(load_gemma4, resolve_mm_root, resolve_projector) rather than re-implementing
it — this check must fail/pass in agreement with what a real probe.py run
would do, not by a separate, potentially-drifting heuristic.

Loads each model for real (there's no way to introspect a HF model's actual
runtime module structure without instantiating it), but does none of the
per-dataset embedding extraction or probe training probe.py's main pipeline
does — just: load, resolve pre-projector hook module, resolve post-projector
(informational), unload, move to the next model. Independent of any dataset/
manifest — this is purely about model architecture, not data.

Usage:
  uv run python src/linear_probe/check_vision_towers.py \\
    --models google/gemma-4-26B-A4B-it Qwen/Qwen3.5-9B some/other-vlm

  # Defaults to whatever "models" is set to in probe_config.yaml if --models
  # is omitted, so you can check the exact same list you're about to probe:
  uv run python src/linear_probe/check_vision_towers.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import torch.nn as nn

from src.linear_probe.probe import (
    load_gemma4, resolve_mm_root, resolve_projector, unload,
    _load_layer_selections, _LAYER_SELECTIONS_PATH,
)


def _out_shape(module: nn.Module) -> str:
    """Static output shape from weight attributes — no forward pass needed,
    same approach the (now-removed) layer_explorer.py app used."""
    if isinstance(module, nn.Linear):
        return str(module.out_features)
    if isinstance(module, nn.Conv2d):
        return f"C={module.out_channels}"
    if isinstance(module, nn.Embedding):
        return str(module.embedding_dim)
    if isinstance(module, nn.LayerNorm):
        return str(tuple(module.normalized_shape))
    if isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        return f"C={module.num_features if hasattr(module, 'num_features') else module.num_channels}"
    if isinstance(module, nn.MultiheadAttention):
        return str(module.embed_dim)
    return "—"


def _last_leaf(vision_tower: nn.Module) -> tuple[str, str, str]:
    """(name, type, out_shape) of the last leaf submodule (no children) inside
    vision_tower, via depth-first traversal — an approximation, without a real
    forward pass, of what a forward hook on vision_tower itself is actually
    capturing. Useful for spot-checking whether two same-family models (e.g.
    a 9B and 27B variant) end their vision tower on comparably-shaped,
    comparably-normalized output, or something structurally different."""
    last_name, last_mod = None, None
    for name, sub in vision_tower.named_modules():
        if name and next(sub.children(), None) is None:
            last_name, last_mod = name, sub
    if last_mod is None:
        return "(no leaf found)", type(vision_tower).__name__, "—"
    return last_name, type(last_mod).__name__, _out_shape(last_mod)


def check_model(model_id: str, device: str, quantize: str | None = None,
                 trust_remote_code: bool = True) -> dict:
    """Load `model_id` for real, attempt to resolve its pre-/post-projector
    hook modules the same way probe.py's main pipeline does, then unload.
    Returns a result dict — never raises; any failure is captured in the
    "ok"/"error" fields so one bad model doesn't abort checking the rest."""
    result: dict = {"model_id": model_id}
    model = None
    try:
        # load_gemma4() calls resolve_mm_root() internally (it's part of what
        # it returns), so any failure lands in this one try/except either
        # way — but WHICH exception type distinguishes where it happened:
        # resolve_mm_root() only ever raises AttributeError (its own
        # explicit "Could not locate vision encoder" message); anything else
        # (ValueError for an unrecognized architecture, missing
        # trust_remote_code, network errors, ...) necessarily happened
        # earlier, in from_pretrained() itself, before resolve_mm_root ever
        # ran. Reported separately below — telling someone to go fix
        # resolve_mm_root()'s vision_tower/visual check for a model that
        # never even finished loading is actively misleading.
        _processor, model, mm_root = load_gemma4(
            model_id, device, quantize=quantize, trust_remote_code=trust_remote_code,
        )
        result["ok"] = True
        result["vision_attr"] = "vision_tower" if hasattr(mm_root, "vision_tower") else "?"
        result["vision_tower_type"] = type(mm_root.vision_tower).__name__
        leaf_name, leaf_type, leaf_out = _last_leaf(mm_root.vision_tower)
        result["last_leaf_name"] = leaf_name
        result["last_leaf_type"] = leaf_type
        result["last_leaf_out"] = leaf_out

        proj, proj_name = resolve_projector(mm_root)
        result["has_postproj"] = proj is not None
        result["postproj_name"] = proj_name

        layer_selections = _load_layer_selections()
        result["has_manual_layer_selection"] = model_id in layer_selections
    except AttributeError as e:
        result["ok"] = False
        result["stage"] = "resolve_mm_root"
        result["error"] = str(e)
    except Exception as e:
        result["ok"] = False
        result["stage"] = "load"
        result["error"] = str(e)
    finally:
        if model is not None:
            unload(model)
    return result


def print_report(results: list[dict]) -> None:
    print(f"\n{'='*110}")
    print(f"{'Model':<40} {'Pre-proj':<10} {'Post-proj':<12} {'Manual sel.':<12} {'Last leaf (type, out)':<30}")
    print("-" * 110)
    for r in results:
        if r["ok"]:
            pre = f"OK ({r['vision_tower_type']})"
            post = r["postproj_name"] if r["has_postproj"] else "not found"
            manual = "yes" if r["has_manual_layer_selection"] else "heuristic"
            leaf = f"{r['last_leaf_name']}: {r['last_leaf_type']} ({r['last_leaf_out']})"
            print(f"{r['model_id']:<40} {pre:<10} {post:<12} {manual:<12} {leaf:<30}")
        else:
            print(f"{r['model_id']:<40} FAILED ({r['stage']}): {r['error'][:60]}")
    print("=" * 110)

    ok = [r for r in results if r["ok"]]
    failed_resolve = [r for r in results if not r["ok"] and r["stage"] == "resolve_mm_root"]
    failed_load = [r for r in results if not r["ok"] and r["stage"] == "load"]
    print(f"\n{len(ok)}/{len(results)} model(s) have a resolvable vision tower.")

    if failed_resolve:
        print(f"\n{len(failed_resolve)} loaded fine but FAILED at vision-tower "
              f"resolution — resolve_mm_root() in probe.py checks for a 'vision_tower' "
              f"or 'visual' top-level attribute; this architecture uses neither, so it "
              f"needs a one-line addition there:")
        for r in failed_resolve:
            print(f"  - {r['model_id']}: {r['error']}")

    if failed_load:
        print(f"\n{len(failed_load)} FAILED before loading even completed — NOT a "
              f"vision_tower/visual naming issue, resolve_mm_root() never ran for "
              f"these. Usually one of: transformers too old for this architecture "
              f"(try `uv add -U transformers`), or the repo needs "
              f"trust_remote_code=True (runs arbitrary code from that HF repo — only "
              f"opt into this for repos/orgs you actually trust):")
        for r in failed_load:
            print(f"  - {r['model_id']}: {r['error']}")
    no_postproj = [r for r in ok if not r["has_postproj"]]
    if no_postproj:
        print(f"\n{len(no_postproj)} model(s) have no resolvable post-projector — "
              f"fine for the paper's pre-projector-only setup, but the optional "
              f"post-projector side analysis won't run for these:")
        for r in no_postproj:
            print(f"  - {r['model_id']}")
    no_manual = [r for r in ok if not r["has_manual_layer_selection"]]
    if no_manual:
        print(f"\n{len(no_manual)} model(s) have no entry in {_LAYER_SELECTIONS_PATH} — "
              f"they'll use the heuristic fallback (vision_tower's own final output), "
              f"which is what the paper's setup describes, so this is expected/fine "
              f"unless you specifically want a non-final layer for one of them.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None,
                     help="Model IDs to check. Defaults to probe_config.yaml's 'models' list.")
    ap.add_argument("--probe_config", default="src/linear_probe/probe_config.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quantize", choices=["4bit", "8bit"], default=None)
    ap.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=None,
                     help="Allow loading models with custom modeling code on their HF repo "
                          "(executes arbitrary Python from that repo). Defaults to "
                          "probe_config.yaml's 'trust_remote_code' (itself True by default) "
                          "when not passed explicitly here.")
    args = ap.parse_args()

    cfg: dict = {}
    cfg_path = Path(args.probe_config)
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    models = args.models
    if models is None:
        models = cfg.get("models")
        if not models:
            print(f"ERROR: --models not given and no 'models' list found in {cfg_path}.")
            sys.exit(1)
        print(f"Using models from {cfg_path}: {models}")

    trust_remote_code = args.trust_remote_code
    if trust_remote_code is None:
        trust_remote_code = cfg.get("trust_remote_code", True)

    results = []
    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Checking {model_id} ...")
        results.append(check_model(model_id, args.device, quantize=args.quantize,
                                    trust_remote_code=trust_remote_code))

    print_report(results)


if __name__ == "__main__":
    main()
