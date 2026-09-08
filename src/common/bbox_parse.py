"""Bounding-box extraction from VLM text output — Stage 5's spatial analog of
verifier_cot.py's extract_answer_tag() (open-ended <answer> tag parsing).

Convention: <box>x1,y1,x2,y2</box>optional_category_name</box> per instance,
one per line, or the literal <box>none</box> for "no instances visible" — a
model outputting NOTHING is a valid negative prediction, not a parse error
(same "found=False, not an exception" spirit as extract_answer_tag)."""
from __future__ import annotations

import json
import re

_BOX_COORDS_RE = re.compile(
    r"<box>\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*,\s*([\d.\-]+)\s*</box>",
    re.IGNORECASE,
)
# Optional trailing category label immediately after a box tag (same line,
# before any other tag) — e.g. "<box>1,2,3,4</box>leaf_spot". Matching just
# the box coordinates first, then trying this separately on the remainder,
# avoids the fragile single-regex lookahead this used to use (broke as soon
# as a box tag was followed by anything other than another <box> or
# end-of-string — e.g. the verifier's <answer><box>...</box></answer>
# wrapping, see verifier_prompts_detection.yaml).
_TRAILING_LABEL_RE = re.compile(r"^([^<\n]+)")
_NONE_RE = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)


def extract_boxes(text: str) -> list[dict]:
    """Parse every <box>x1,y1,x2,y2</box>[category] occurrence in `text`.

    Returns [{"bbox": [x1,y1,x2,y2], "category_name": str}, ...] — bbox
    values are floats, NOT yet clamped to any image's bounds (see
    clamp_box()). A malformed individual box (non-numeric coordinate) is
    skipped rather than aborting the whole parse — mirrors
    extract_answer_tag's "don't let one bad piece break everything" spirit,
    just applied per-box instead of per-whole-answer. Returns [] for
    <box>none</box> or no matches at all — both are valid "zero instances
    predicted" outcomes, not parse failures; callers distinguish "explicitly
    none" from "nothing matched" via found_none_tag() if that distinction
    matters (e.g. logging malformed-output rate)."""
    if _NONE_RE.search(text):
        return []
    boxes = []
    for m in _BOX_COORDS_RE.finditer(text):
        try:
            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        except ValueError:
            continue
        label_m = _TRAILING_LABEL_RE.match(text[m.end():])
        category = label_m.group(1).strip() if label_m else ""
        boxes.append({"bbox": [x1, y1, x2, y2], "category_name": category})
    return boxes


_BOX_2D_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def extract_boxes_json(text: str) -> list[dict]:
    """Parse Gemma's NATIVE grounding output format — a JSON array of
    {"box_2d": [y_min, x_min, y_max, x_max], "label": str} objects, on the
    same 0-1000 normalized scale as the <box> tag format (still needs
    denormalize_box_1000 same as extract_boxes()'s output) but Y-FIRST,
    not the [x1,y1,x2,y2] convention this repo's bbox_convention/
    bbox_metrics.py otherwise use throughout.

    Root-cause note (2026-08-25): Gemma's actual pretraining grounding
    format is this exact box_2d/y-first JSON shape (see Google's Gemma
    vision documentation) — not the free-text <box>x1,y1,x2,y2</box> tag
    extract_boxes() parses. Asking for the tag format (see that function's
    docstring) mostly works, but on some images the model reverts to its
    native y,x-first ordering INSIDE the requested tag with no warning,
    silently swapping every predicted box's axes — confirmed near-total
    (19/20 images) on QuinceSet_detection via a direct swap-and-recompute-
    F1 check against ground truth. Asking for the model's own native
    format instead of fighting it removes that ambiguity. Selected at
    runtime via uses_json_box_format(model_id) — this is Gemma-specific,
    see that function's docstring for why other model families keep using
    extract_boxes() instead.

    Returns [{"bbox": [x1,y1,x2,y2], "category_name": str}, ...] — ALREADY
    reordered from box_2d's y-first convention into this repo's
    x1,y1,x2,y2 convention, still on the 0-1000 scale. A response with no
    JSON array at all, or an individual malformed object, contributes
    nothing rather than raising — same "don't let one bad piece break
    everything" spirit as extract_boxes(); an empty array ("[]", the
    format this repo's prompts ask for on a "nothing visible" answer)
    correctly returns [] the same way "no match found" does, since both
    mean zero predicted instances."""
    m = _BOX_2D_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    boxes = []
    for item in items:
        if not isinstance(item, dict) or "box_2d" not in item:
            continue
        coords = item["box_2d"]
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        try:
            y1, x1, y2, x2 = (float(c) for c in coords)
        except (TypeError, ValueError):
            continue
        category = str(item.get("label", ""))
        boxes.append({"bbox": [x1, y1, x2, y2], "category_name": category})
    return boxes


def uses_json_box_format(model_id: str) -> bool:
    """True for model families whose NATIVE grounding output is the
    box_2d/y-first JSON shape extract_boxes_json() parses, rather than the
    free-text <box>x1,y1,x2,y2</box> tag extract_boxes() parses — see
    extract_boxes_json()'s docstring for the root-cause evidence. Same
    "gemma" condition as needs_1000_scale_boxes() below (both stem from
    the same underlying native training convention), kept as a separate
    flag/function rather than reusing needs_1000_scale_boxes() directly,
    since the two could diverge for a future model family that needs one
    fix but not the other. Qwen2-VL-family models are NOT included —
    unaffected by the axis-order failure mode this exists to route around."""
    return "gemma" in model_id.lower()


def denormalize_box_1000(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """Convert a box reported on Gemma's native 0-1000 normalized grounding
    scale (confirmed via Google's own PaliGemma/Gemma vision demo code —
    ai.google.dev/gemma/docs/capabilities/vision/image) into real pixel
    coordinates for `img_w` x `img_h`. This is NOT a Qwen convention —
    Qwen2-VL-family models report boxes in real image pixels natively, so
    this must only be applied for the model families confirmed to need it
    (see needs_1000_scale_boxes() below), never applied unconditionally.

    Root-cause note (2026-08-24): every prompt in this repo asks the model
    for boxes as literal pixel coordinates ("Coordinates are pixel
    positions in the image as shown"), but a full 1,844-image
    detection_baseline run showed 100% of Gemma's predicted box
    coordinates <= 1000 regardless of each image's actual (often much
    larger) resolution — the unmistakable signature of the model ignoring
    that instruction and falling back to its own trained 0-1000 scale
    instead. clamp_box() alone cannot fix this: it only clips
    out-of-range values into an image's bounds, it never rescales, so any
    image larger than ~1000px on a side was scored against the wrong
    coordinate frame regardless of how accurate the model's own internal
    localization actually was."""
    x1, y1, x2, y2 = bbox
    return [x1 / 1000.0 * img_w, y1 / 1000.0 * img_h, x2 / 1000.0 * img_w, y2 / 1000.0 * img_h]


def needs_1000_scale_boxes(model_id: str) -> bool:
    """True for model families empirically confirmed to report boxes on
    Gemma's native 0-1000 normalized scale rather than real pixels,
    regardless of what the prompt asks for — see denormalize_box_1000's
    docstring. Only "gemma" is confirmed as of this writing (measured
    directly against real detection_baseline.py output); Qwen2-VL-family
    models are NOT included here since they report real pixel coordinates
    natively. If a new model family is added to this repo's detection
    stage, verify its own box-coordinate convention the same way (compare
    each row's max predicted coordinate against that image's actual
    width/height across a real run) before assuming either behavior."""
    return "gemma" in model_id.lower()


def found_none_tag(text: str) -> bool:
    """True if the text explicitly used <box>none</box> — distinguishes
    "model correctly said nothing is here" from "model's output didn't
    parse at all" (both return [] from extract_boxes)."""
    return bool(_NONE_RE.search(text))


def clamp_box(bbox: list[float], img_w: int, img_h: int) -> list[float]:
    """Clip a predicted box into [0, img_w] x [0, img_h] — models do
    hallucinate out-of-range or negative coordinates. Also normalizes a
    swapped corner order (x2<x1 or y2<y1) by sorting, so downstream IoU
    math (bbox_metrics.iou) never sees a degenerate/negative-area box from
    a model that emitted corners in the wrong order."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    return [x1, y1, x2, y2]
