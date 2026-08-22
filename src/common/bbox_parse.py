"""Bounding-box extraction from VLM text output — Stage 5's spatial analog of
verifier_cot.py's extract_answer_tag() (open-ended <answer> tag parsing).

Convention: <box>x1,y1,x2,y2</box>optional_category_name</box> per instance,
one per line, or the literal <box>none</box> for "no instances visible" — a
model outputting NOTHING is a valid negative prediction, not a parse error
(same "found=False, not an exception" spirit as extract_answer_tag)."""
from __future__ import annotations

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
