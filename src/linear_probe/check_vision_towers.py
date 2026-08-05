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

from src.linear_probe.probe import (
    load_gemma4, resolve_mm_root, resolve_projector, unload,
    _load_layer_selections, _LAYER_SELECTIONS_PATH,
)


def check_model(model_id: str, device: str, quantize: str | None = None) -> dict:
    """Load `model_id` for real, attempt to resolve its pre-/post-projector
    hook modules the same way probe.py's main pipeline does, then unload.
    Returns a result dict — never raises; any failure is captured in the
    "ok"/"error" fields so one bad model doesn't abort checking the rest."""
    result: dict = {"model_id": model_id}
    model = None
    try:
        _processor, model, mm_root = load_gemma4(model_id, device, quantize=quantize)
        result["ok"] = True
        result["vision_attr"] = "vision_tower" if hasattr(mm_root, "vision_tower") else "?"
        result["vision_tower_type"] = type(mm_root.vision_tower).__name__

        proj, proj_name = resolve_projector(mm_root)
        result["has_postproj"] = proj is not None
        result["postproj_name"] = proj_name

        layer_selections = _load_layer_selections()
        result["has_manual_layer_selection"] = model_id in layer_selections
    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)
    finally:
        if model is not None:
            unload(model)
    return result


def print_report(results: list[dict]) -> None:
    print(f"\n{'='*78}")
    print(f"{'Model':<40} {'Pre-proj':<10} {'Post-proj':<12} {'Manual sel.':<12}")
    print("-" * 78)
    for r in results:
        if r["ok"]:
            pre = f"OK ({r['vision_tower_type']})"
            post = r["postproj_name"] if r["has_postproj"] else "not found"
            manual = "yes" if r["has_manual_layer_selection"] else "heuristic"
            print(f"{r['model_id']:<40} {pre:<10} {post:<12} {manual:<12}")
        else:
            print(f"{r['model_id']:<40} FAILED: {r['error']}")
    print("=" * 78)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(ok)}/{len(results)} model(s) have a resolvable vision tower.")
    if failed:
        print(f"{len(failed)} FAILED — resolve_mm_root() in probe.py checks for a "
              f"'vision_tower' or 'visual' top-level attribute; a different naming "
              f"convention needs a one-line addition there. Failing models:")
        for r in failed:
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
    args = ap.parse_args()

    models = args.models
    if models is None:
        import yaml
        cfg_path = Path(args.probe_config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            models = cfg.get("models")
        if not models:
            print(f"ERROR: --models not given and no 'models' list found in {cfg_path}.")
            sys.exit(1)
        print(f"Using models from {cfg_path}: {models}")

    results = []
    for i, model_id in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Checking {model_id} ...")
        results.append(check_model(model_id, args.device, quantize=args.quantize))

    print_report(results)


if __name__ == "__main__":
    main()
