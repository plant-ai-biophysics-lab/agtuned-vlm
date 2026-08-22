"""Bounding-box IoU + pred-vs-ground-truth matching — Stage 5. No IoU/box-
matching function existed anywhere in this repo before (the only prior bbox
code, experiments/alignment_test/check_alignment.py, only ever compares a
heatmap to a box MASK, never box-vs-box)."""
from __future__ import annotations


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Standard axis-aligned IoU on [x1,y1,x2,y2] boxes. 0.0 for
    non-overlapping or degenerate (zero-area) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    pred: list[dict], gt: list[dict], iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy descending-IoU matching between pred and gt box lists (each a
    list of {"bbox": [...], ...} dicts — only "bbox" is read here). Greedy,
    not Hungarian-optimal — the simplest thing that works, matching this
    repo's own "iterate only if wrong" pattern (see verifier_cot.py's
    module docstring history); revisit only if greedy's known failure mode
    (a suboptimal early match blocking a better one) shows up in practice.

    Returns (matches, unmatched_pred_idx, unmatched_gt_idx) where matches is
    [(pred_idx, gt_idx, iou), ...], each index used at most once, only pairs
    with iou >= iou_threshold included."""
    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            score = iou(p["bbox"], g["bbox"])
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(key=lambda t: -t[0])

    matched_pred, matched_gt = set(), set()
    matches: list[tuple[int, int, float]] = []
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, score))

    unmatched_pred = [i for i in range(len(pred)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt)) if i not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def precision_recall_f1(
    matches: list[tuple[int, int, float]], unmatched_pred: list[int], unmatched_gt: list[int],
) -> dict:
    """Standard detection precision/recall/F1 from match_boxes' output, plus
    mean_iou over the matched pairs only (undefined / 0.0 if nothing
    matched — callers should check n_matched before trusting mean_iou)."""
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = sum(s for _, _, s in matches) / tp if tp else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "mean_iou_matched": round(mean_iou, 4), "n_matched": tp, "n_pred": tp + fp, "n_gt": tp + fn,
    }
