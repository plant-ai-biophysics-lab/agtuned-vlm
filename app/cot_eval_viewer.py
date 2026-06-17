"""
CoT Eval Viewer — explore cot_eval results from class_log.jsonl.

Usage:
    streamlit run app/cot_eval_viewer.py -- --output_dir outputs/cot_eval

Views:
  1. Overview   — F1 + judge accuracy by condition, overall / by task / by dataset
  2. Class Log  — per-class sample browser: CoT steps, all condition responses, judge scores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output_dir", default="outputs/cot_eval")
    args, _ = p.parse_known_args()
    return args

_args = _parse_args()
_OUT = Path(_args.output_dir)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_results(output_dir: str) -> dict:
    p = Path(output_dir) / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_class_log(output_dir: str) -> list[dict]:
    p = Path(output_dir) / "class_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


@st.cache_data
def load_judge_log(output_dir: str) -> list[dict]:
    p = Path(output_dir) / "judge_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

# ── Colours ───────────────────────────────────────────────────────────────────

_COND_COLOR = {
    "base":         "#5b8dd9",
    "cot":          "#e07b54",
    "thinking":     "#2ecc71",
    "cot_thinking": "#9b59b6",
}
_COND_LABEL = {
    "base":         "Base",
    "cot":          "Structured CoT",
    "thinking":     "Native Thinking",
    "cot_thinking": "CoT + Thinking",
}
CONDITIONS = list(_COND_LABEL)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{text}</span>'
    )


def _score_color(v: float) -> str:
    if v >= 0.6:
        return "#2ecc71"
    if v >= 0.3:
        return "#f39c12"
    return "#e74c3c"


def _metric_pill(label: str, value: float) -> str:
    color = _score_color(value)
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78em;margin-right:4px">'
        f'{label}: {value:.3f}</span>'
    )


def _cond_pill(cond: str) -> str:
    return _badge(_COND_LABEL[cond], _COND_COLOR[cond])

# ── Overview tab ──────────────────────────────────────────────────────────────

def render_overview(results: dict) -> None:
    if not results:
        st.warning("No results.json found. Run the experiment first.")
        return

    ov = results.get("overall", {})

    # ── Summary cards ─────────────────────────────────────────────────────────
    st.subheader("Overall")
    cols = st.columns(len(CONDITIONS))
    for col, cond in zip(cols, CONDITIONS):
        f1    = ov.get(f"{cond}_f1", 0.0)
        judge = ov.get(f"{cond}_judge_acc", 0.0)
        gain  = ov.get(f"{cond}_gain", 0.0) if cond != "base" else 0.0
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {_COND_COLOR[cond]};"
                f"padding:10px 14px;border-radius:4px;background:#1e1e1e'>"
                f"<div style='font-size:0.75em;color:#aaa'>{_COND_LABEL[cond]}</div>"
                f"<div style='font-size:1.6em;font-weight:700'>{f1:.3f}</div>"
                f"<div style='font-size:0.78em;color:#aaa'>F1</div>"
                f"<div style='font-size:0.95em;color:#ccc;margin-top:4px'>"
                f"Judge {judge:.3f}"
                + (f"  <span style='color:{'#2ecc71' if gain>0 else '#e74c3c'}'>"
                   f"{'↑' if gain>0 else '↓'}{abs(gain):.3f}</span>" if cond != "base" else "")
                + f"</div></div>",
                unsafe_allow_html=True,
            )

    st.caption(f"n={ov.get('n_classes', 0)} classes evaluated")

    # ── By task ───────────────────────────────────────────────────────────────
    by_task = results.get("by_task", {})
    if by_task:
        st.subheader("By task type")
        rows = []
        for tt, v in sorted(by_task.items()):
            row = {"task_type": tt}
            for cond in CONDITIONS:
                row[f"{_COND_LABEL[cond]} F1"]    = v.get(f"{cond}_f1", 0.0)
                row[f"{_COND_LABEL[cond]} judge"]  = v.get(f"{cond}_judge_acc", 0.0)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("task_type")
        st.dataframe(
            df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
        )

    # ── By dataset ────────────────────────────────────────────────────────────
    by_ds = results.get("by_dataset", {})
    if by_ds:
        st.subheader("By dataset")
        rows = []
        for ds, v in sorted(by_ds.items()):
            row = {"dataset": ds, "n": v.get("n_classes", 0)}
            for cond in CONDITIONS:
                row[f"{_COND_LABEL[cond]} F1"]   = round(v.get(f"{cond}_f1", 0.0), 3)
                row[f"{_COND_LABEL[cond]} judge"] = round(v.get(f"{cond}_judge_acc", 0.0), 3)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("dataset")
        f1_cols = [f"{_COND_LABEL[c]} F1" for c in CONDITIONS]
        st.dataframe(
            df.style.background_gradient(subset=f1_cols, cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
        )

# ── Class log tab ─────────────────────────────────────────────────────────────

def render_class_log(log: list[dict], judge_log: list[dict]) -> None:
    if not log:
        st.warning("No class_log.jsonl found yet.")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    datasets = sorted({e["dataset"] for e in log})
    sel_ds   = st.sidebar.selectbox("Dataset", ["(all)"] + datasets)

    filtered = log if sel_ds == "(all)" else [e for e in log if e["dataset"] == sel_ds]
    classes  = sorted({e["class_label"] for e in filtered})
    sel_cls  = st.sidebar.selectbox("Class", classes)

    entry = next((e for e in filtered if e["class_label"] == sel_cls), None)
    if entry is None:
        st.info("No entry found.")
        return

    # ── Class header ──────────────────────────────────────────────────────────
    st.markdown(
        f"### {entry['class_label']}  "
        f"<span style='color:#aaa;font-size:0.8em'>{entry['dataset']} · "
        f"{entry.get('task_type','')} · crop: {entry.get('crop','?')}</span>",
        unsafe_allow_html=True,
    )

    # Metric pills
    pills = ""
    for cond in CONDITIONS:
        f1    = entry.get(f"{cond}_f1",        0.0)
        judge = entry.get(f"{cond}_judge_acc",  0.0)
        pills += (
            f"<div style='display:inline-block;margin:4px 6px 4px 0;"
            f"border-left:3px solid {_COND_COLOR[cond]};padding:4px 10px;"
            f"border-radius:4px;background:#1e1e1e'>"
            f"<span style='font-size:0.75em;color:#aaa'>{_COND_LABEL[cond]}</span><br>"
            f"<b>F1 {f1:.3f}</b> &nbsp; judge {judge:.3f}"
            + (f" &nbsp;<span style='color:{'#2ecc71' if (g:=entry.get(f'{cond}_gain',0))>0 else '#e74c3c'}'>"
               f"{'↑' if g>0 else '↓'}{abs(g):.3f}</span>" if cond != "base" else "")
            + "</div>"
        )
    st.markdown(pills, unsafe_allow_html=True)
    st.divider()

    # ── Sample picker ─────────────────────────────────────────────────────────
    n_samples = len(entry.get("base_responses", []))
    if n_samples == 0:
        st.info("No responses recorded.")
        return

    sample_idx = st.slider("Sample image index", 0, n_samples - 1, 0)

    # ── CoT step-by-step ─────────────────────────────────────────────────────
    cot_grid = entry.get("cot_step_grid", [])
    ct_grid  = entry.get("cot_thinking_step_grid", [])

    if cot_grid:
        st.markdown("#### CoT chain")
        for step_i, step_answers in enumerate(cot_grid):
            ans_cot = step_answers[sample_idx] if sample_idx < len(step_answers) else "—"
            ans_ct  = ct_grid[step_i][sample_idx] if (ct_grid and sample_idx < len(ct_grid[step_i])) else "—"

            with st.expander(f"Step {step_i + 1}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"{_badge('Structured CoT', _COND_COLOR['cot'])}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"> {ans_cot}")
                with c2:
                    st.markdown(
                        f"{_badge('CoT + Thinking', _COND_COLOR['cot_thinking'])}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"> {ans_ct}")

    # ── Final classification answers ──────────────────────────────────────────
    st.markdown("#### Final classification")

    def _get(key: str) -> str:
        lst = entry.get(key, [])
        return lst[sample_idx] if sample_idx < len(lst) else "—"

    cols = st.columns(len(CONDITIONS))
    for col, cond in zip(cols, CONDITIONS):
        response = _get(f"{cond}_responses")
        f1       = entry.get(f"{cond}_f1", 0.0)
        judge    = entry.get(f"{cond}_judge_acc", 0.0)
        with col:
            st.markdown(
                f"{_badge(_COND_LABEL[cond], _COND_COLOR[cond])}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:8px;padding:10px;background:#1e1e1e;"
                f"border-radius:6px;min-height:60px;font-size:0.95em'>{response or '—'}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<small style='color:#aaa'>F1 {f1:.3f} &nbsp; judge {judge:.3f}</small>",
                unsafe_allow_html=True,
            )

    # ── Native thinking full blob ─────────────────────────────────────────────
    thinking_full    = entry.get("thinking_full", [])
    ct_full          = entry.get("cot_thinking_full", [])

    if thinking_full or ct_full:
        st.markdown("#### Reasoning blobs")
        t1, t2 = st.tabs(["Native Thinking", "CoT + Thinking"])
        with t1:
            blob = thinking_full[sample_idx] if sample_idx < len(thinking_full) else "—"
            st.text_area("thinking_full", blob, height=300, label_visibility="collapsed")
        with t2:
            blob = ct_full[sample_idx] if sample_idx < len(ct_full) else "—"
            st.text_area("cot_thinking_full", blob, height=300, label_visibility="collapsed")

    # ── Judge details for this sample ─────────────────────────────────────────
    cls_key = f"{entry['dataset']}/{entry['class_label']}"
    sample_judge = [
        r for r in judge_log
        if r.get("cls_key") == cls_key and r.get("sample_idx") == sample_idx
    ]

    if sample_judge:
        st.markdown("#### Judge reasoning")
        for r in sample_judge:
            cond = r.get("condition", "?")
            correct = r.get("judged_correct", 0)
            color = "#2ecc71" if correct else "#e74c3c"
            icon  = "✓" if correct else "✗"
            with st.expander(
                f"{_COND_LABEL.get(cond, cond)}  —  {icon}  pred: \"{r.get('response','')}\"",
                expanded=False,
            ):
                st.markdown(
                    f"**GT:** `{r.get('gt','')}` &nbsp;&nbsp; "
                    f"**Pred:** `{r.get('response','')}` &nbsp;&nbsp; "
                    f"<span style='color:{color}'><b>{icon} judged_correct={correct}</b></span>",
                    unsafe_allow_html=True,
                )
                reasoning = r.get("judge_reasoning", "")
                # Strip markdown code fences if present
                reasoning = reasoning.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = json.loads(reasoning)
                    st.json(parsed)
                except Exception:
                    st.text(reasoning)

    # ── Raw entry dump ────────────────────────────────────────────────────────
    with st.expander("Raw entry (JSON)", expanded=False):
        # Show without the bulky response arrays
        slim = {k: v for k, v in entry.items()
                if k not in ("base_responses", "cot_responses", "thinking_responses",
                             "cot_thinking_responses", "thinking_full", "cot_thinking_full",
                             "cot_step_grid", "cot_thinking_step_grid")}
        st.json(slim)

# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="CoT Eval Viewer",
        page_icon="🌱",
        layout="wide",
    )
    st.title("🌱 CoT Eval Viewer")
    st.caption(f"Output dir: `{_OUT}`")

    results   = load_results(str(_OUT))
    class_log = load_class_log(str(_OUT))
    judge_log = load_judge_log(str(_OUT))

    if not results and not class_log:
        st.error(f"No results found in `{_OUT}`. Run the experiment first.")
        st.code(
            "uv run python main.py cot-eval \\\n"
            "  --model google/gemma-4-26B-A4B-it \\\n"
            "  --split test \\\n"
            "  --output_dir outputs/cot_eval",
            language="bash",
        )
        return

    tab_overview, tab_log = st.tabs(["📊 Overview", "🔍 Class Log"])

    with tab_overview:
        render_overview(results)

    with tab_log:
        render_class_log(class_log, judge_log)


if __name__ == "__main__":
    main()
