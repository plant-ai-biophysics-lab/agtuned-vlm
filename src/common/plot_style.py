"""Shared bar-chart styling for backbone-comparison plots — used by
src/linear_probe/probe.py's summary_by_task.png, src/vision_baseline_collate.py's
stage2_f1_by_task.png, and src/linear_probe/collate_seeds.py's seed-collated
chart, so the same backbone always reads the same color across every chart
in the pipeline rather than each script picking its own palette.
"""
from __future__ import annotations

import colorsys
import re
from collections import defaultdict

import matplotlib.colors as mcolors

FAMILY_BASE_COLORS = {
    "dinov3": "#1f77b4",
    "yolo": "#ff7f0e",
    "qwen": "#2ca02c",
    "gemma": "#9467bd",
}
DEFAULT_FAMILY_COLOR = "#7f7f7f"

VISION_ONLY_MARKERS = ("yolo", "dino")


def is_vision_only(backbone: str) -> bool:
    """No language model — a frozen classifier/self-supervised tower (YOLO,
    DINO) rather than a VLM. Used to hatch these bars distinctly."""
    name = backbone.lower()
    return any(marker in name for marker in VISION_ONLY_MARKERS)


def backbone_family(name: str) -> str:
    """Group a backbone label into a model family so same-family variants
    (e.g. Qwen3.5-9B / Qwen3.5-27B) get shades of the same base color."""
    n = name.lower()
    for family in FAMILY_BASE_COLORS:
        if family in n:
            return family
    return n


def param_count_b(name: str) -> float:
    """Extract the largest '<number>B' parameter count in a backbone label
    (e.g. "gemma-4-26B-A4B-it" -> 26.0), 0.0 if none found. Used to rank
    same-family variants from smallest to largest for shading."""
    matches = re.findall(r"(\d+(?:\.\d+)?)b", name.lower())
    return max((float(m) for m in matches), default=0.0)


def family_shaded_colors(backbones: list[str]) -> dict[str, tuple]:
    """Assign each backbone a color: same base hue per family, with more
    parameters -> darker shade. Single-member families get a mid shade."""
    families: dict[str, list[str]] = defaultdict(list)
    for b in backbones:
        families[backbone_family(b)].append(b)

    colors: dict[str, tuple] = {}
    for family, members in families.items():
        base_hex = FAMILY_BASE_COLORS.get(family, DEFAULT_FAMILY_COLOR)
        h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_hex))
        params = {b: param_count_b(b) for b in members}
        lo, hi = min(params.values()), max(params.values())
        for b in members:
            if hi > lo:
                frac = (params[b] - lo) / (hi - lo)  # 0 = smallest, 1 = largest
            else:
                frac = 0.5  # single member (or tied params) -> mid shade
            lightness = 0.68 - frac * 0.38  # smallest ~0.68 (light) -> largest ~0.30 (dark)
            colors[b] = colorsys.hls_to_rgb(h, lightness, s)
    return colors
