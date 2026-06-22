"""
Results Viewer — browse per-class outputs from language and vision assessments.

Usage:
    streamlit run app/results_viewer.py -- \
        --language_dir outputs/language_assessment/ \
        --vision_dir   outputs/vision_assessment/ \
        --results_csv  outputs/results_table.csv

Tabs:
  1. Summary      — collated results_table.csv with gradient colouring
  2. Class Explorer — per-class/model table: metrics, responses, correct flags
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--language_dir", default="outputs/language_assessment/")
    p.add_argument("--vision_dir",   default="outputs/vision_assessment/")
    p.add_argument("--results_csv",  default="outputs/results_table.csv")
    args, _ = p.parse_known_args()
    return args

_args = _parse_args()
_LANG_DIR = Path(_args.language_dir)
_VIS_DIR  = Path(_args.vision_dir)
_RES_CSV  = Path(_args.results_csv)

# ── Data loading ──────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _model_name(stem: str, prefix: str) -> str:
    """cls_google__gemma-4-26B-A4B-it → google/gemma-4-26B-A4B-it"""
    raw = stem[len(prefix):]
    return raw.replace("__", "/")


@st.cache_data
def load_results_table(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def load_assessment_data(lang_dir: str, vis_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (lang_df, vis_df) as flat DataFrames with a 'model' column."""
    lang_rows: list[dict] = []
    vis_rows:  list[dict] = []

    ld, vd = Path(lang_dir), Path(vis_dir)

    # ── Language CSVs ─────────────────────────────────────────────────────────
    lang_cls_files  = sorted(ld.glob("cls_*.csv"))
    lang_desc_files = sorted(ld.glob("desc_*.csv"))

    cls_map  = {_model_name(p.stem, "cls_"):  p for p in lang_cls_files}
    desc_map = {_model_name(p.stem, "desc_"): p for p in lang_desc_files}
    all_lang_models = sorted(set(cls_map) | set(desc_map))

    for model in all_lang_models:
        cls_rows_raw  = _read_csv(cls_map[model])  if model in cls_map  else []
        desc_rows_raw = _read_csv(desc_map[model]) if model in desc_map else []

        cls_by_key: dict[tuple, dict] = {}
        for r in cls_rows_raw:
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            cls_by_key[key] = r

        desc_by_key: dict[tuple, dict] = {}
        for r in desc_rows_raw:
            key = (r.get("agml_dataset", ""), r.get("class_label", ""))
            desc_by_key[key] = r

        all_keys = sorted(set(cls_by_key) | set(desc_by_key))
        for key in all_keys:
            c = cls_by_key.get(key, {})
            d = desc_by_key.get(key, {})
            is_judged = "judged_correct" in c
            lang_rows.append({
                "model":                    model,
                "agml_dataset":             key[0],
                "class_label":              key[1],
                "task_type":                c.get("task_type") or d.get("task_type", ""),
                # classification
                "lang_cls_response":        c.get("prediction", ""),
                "lang_cls_resolved":        c.get("resolved_class", ""),
                "lang_cls_resolvable":      _to_bool(c.get("resolvable")),
                "lang_cls_contains":        _to_bool(c.get("contains_match")),
                "lang_cls_resolved_correct": _to_bool(c.get("correct")),
                "lang_cls_judged_correct":   _to_bool(c.get("judged_correct")) if is_judged else None,
                "lang_cls_judged":          is_judged,
                # description
                "lang_desc_response":       d.get("prediction", ""),
                "lang_desc_gt":             d.get("ground_truth", ""),
                "lang_rouge_ss":            _to_float(d.get("rouge_ss")),
                "lang_rouge1":              _to_float(d.get("rouge1")),
            })

    lang_df = pd.DataFrame(lang_rows) if lang_rows else pd.DataFrame()

    # ── Vision CSVs ───────────────────────────────────────────────────────────
    vis_cls_files  = sorted(vd.glob("cls_*.csv"))
    vis_desc_files = sorted(vd.glob("desc_*.csv"))

    cls_vmap  = {_model_name(p.stem, "cls_"):  p for p in vis_cls_files}
    desc_vmap = {_model_name(p.stem, "desc_"): p for p in vis_desc_files}
    all_vis_models = sorted(set(cls_vmap) | set(desc_vmap))

    for model in all_vis_models:
        cls_rows_raw  = _read_csv(cls_vmap[model])  if model in cls_vmap  else []
        desc_rows_raw = _read_csv(desc_vmap[model]) if model in desc_vmap else []

        desc_by_img: dict[tuple, dict] = {}
        for r in desc_rows_raw:
            k = (r.get("agml_dataset", ""), r.get("class_label", ""), r.get("image_name", ""))
            desc_by_img[k] = r

        for r in cls_rows_raw:
            img = r.get("image_name", "")
            dk = (r.get("agml_dataset", ""), r.get("class_label", ""), img)
            d = desc_by_img.get(dk, {})
            is_judged = "judged_correct" in r
            vis_rows.append({
                "model":                    model,
                "agml_dataset":             r.get("agml_dataset", ""),
                "class_label":              r.get("class_label", ""),
                "task_type":                r.get("task_type", ""),
                "crop":                     r.get("crop", ""),
                "image_name":               img,
                # classification
                "vis_cls_response":         r.get("prediction", ""),
                "vis_cls_resolved":         r.get("resolved_class", ""),
                "vis_cls_resolvable":       _to_bool(r.get("resolvable")),
                "vis_cls_contains":         _to_bool(r.get("contains_match")),
                "vis_cls_resolved_correct": _to_bool(r.get("correct")),
                "vis_cls_judged_correct":   _to_bool(r.get("judged_correct")) if is_judged else None,
                "vis_cls_judged":           is_judged,
                # description
                "vis_desc_response":        d.get("prediction", ""),
                "vis_desc_gt":              d.get("ground_truth", ""),
                "vis_rouge_ss":             _to_float(d.get("rouge_ss")),
                "vis_rouge1":               _to_float(d.get("rouge1")),
            })

    vis_df = pd.DataFrame(vis_rows) if vis_rows else pd.DataFrame()
    return lang_df, vis_df


@st.cache_data
def load_metrics_json(lang_dir: str, vis_dir: str) -> tuple[dict, dict]:
    def _load(path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}
    return (
        _load(Path(lang_dir) / "language_assessment_metrics.json"),
        _load(Path(vis_dir)  / "vision_assessment_metrics.json"),
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bool(val) -> bool | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    try:
        return bool(int(val))
    except (ValueError, TypeError):
        return None


def _to_float(val) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fmt_bool(val: bool | None) -> str:
    if val is None:
        return "—"
    return "✓" if val else "✗"


def _score_color(v: float | None) -> str:
    if v is None:
        return "#555"
    if v >= 0.6:
        return "#2ecc71"
    if v >= 0.3:
        return "#f39c12"
    return "#e74c3c"


def _pill(label: str, value, fmt: str = ".3f", color: str | None = None) -> str:
    if value is None:
        return ""
    c = color or _score_color(float(value))
    v_str = format(float(value), fmt) if isinstance(value, (int, float)) else str(value)
    return (
        f'<span style="background:{c};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.78em;margin-right:4px">'
        f'{label}: {v_str}</span>'
    )


def _ds_judge_acc(df: pd.DataFrame, ds: str, model: str,
                  judged_col: str, ds_col: str = "agml_dataset") -> float | None:
    """Compute judge accuracy for a (dataset, model) pair from CSV rows."""
    if df.empty or judged_col not in df.columns:
        return None
    sub = df[(df[ds_col] == ds) & (df["model"] == model)]
    vals = sub[judged_col].dropna()
    if vals.empty:
        return None
    return round(vals.astype(float).mean(), 3)


def _correct_badge(resolved: bool | None, judged: bool | None) -> str:
    """Render separate resolved and judged correct badges."""
    parts = []
    # resolved
    if resolved is not None:
        color = "#2ecc71" if resolved else "#e74c3c"
        icon = "✓" if resolved else "✗"
        parts.append(
            f'<span style="color:{color};font-weight:bold">{icon}</span>'
            f'<span style="color:#aaa;font-size:0.8em"> resolved</span>'
        )
    # judged
    if judged is not None:
        color = "#3498db" if judged else "#e67e22"
        icon = "✓" if judged else "✗"
        parts.append(
            f'<span style="color:{color};font-weight:bold">{icon}</span>'
            f'<span style="color:#aaa;font-size:0.8em"> judged</span>'
        )
    return " &nbsp; ".join(parts) if parts else "<span style='color:#888'>—</span>"


def _desc_pill(label: str, rouge_ss: float | None, baseline: float | None) -> str:
    """Pill colored green/red based on whether rouge_ss beats the baseline."""
    if rouge_ss is None:
        return ""
    if baseline is not None:
        color = "#2ecc71" if rouge_ss > baseline else "#e74c3c"
    else:
        color = _score_color(rouge_ss)
    return _pill(label, rouge_ss, color=color)


# ── Summary tab ───────────────────────────────────────────────────────────────

def render_summary(results_csv: str) -> None:
    df = load_results_table(results_csv)
    if df is None:
        st.warning(
            f"No results table found at `{results_csv}`. "
            "Run `python main.py collate` first."
        )
        return

    rename = {
        "type": "Type", "model": "Model", "params": "Params",
        "lp_f1": "LP F1",
        "lang_cls_f1": "Lang F1", "lang_desc_rss": "Lang ROUGE-SS",
        "vis_cls_f1":  "Vision F1", "vis_desc_rss":  "Vision ROUGE-SS",
    }
    df = df.rename(columns=rename)

    numeric_cols = ["LP F1", "Lang F1", "Lang ROUGE-SS", "Vision F1", "Vision ROUGE-SS"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    present = [c for c in numeric_cols if c in df.columns]
    styled = (
        df.style
        .background_gradient(subset=present, cmap="RdYlGn", vmin=0, vmax=1)
        .format({c: "{:.3f}" for c in present}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(f"Source: `{results_csv}`")


# ── Class Explorer tab ────────────────────────────────────────────────────────

def _per_class_acc(df: pd.DataFrame, correct_col: str) -> float | None:
    if correct_col not in df.columns:
        return None
    vals = df[correct_col].dropna()
    if vals.empty:
        return None
    return round(vals.astype(float).mean(), 3)


def render_class_explorer(
    lang_df: pd.DataFrame,
    vis_df: pd.DataFrame,
    lang_metrics: dict,
    vis_metrics: dict,
) -> None:
    if lang_df.empty and vis_df.empty:
        st.warning("No assessment CSV files found. Run language/vision assessments first.")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    all_datasets = sorted(
        set(lang_df["agml_dataset"].unique().tolist() if not lang_df.empty else [])
        | set(vis_df["agml_dataset"].unique().tolist() if not vis_df.empty else [])
    )
    sel_dataset = st.sidebar.selectbox("Dataset", ["(all)"] + all_datasets)

    all_models = sorted(
        set(lang_df["model"].unique().tolist() if not lang_df.empty else [])
        | set(vis_df["model"].unique().tolist() if not vis_df.empty else [])
    )
    sel_model = st.sidebar.selectbox("Model", ["(all)"] + all_models)

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if sel_dataset != "(all)":
            df = df[df["agml_dataset"] == sel_dataset]
        if sel_model != "(all)":
            df = df[df["model"] == sel_model]
        return df

    lf = _filter(lang_df)
    vf = _filter(vis_df)

    # ── Naive description baselines from metrics JSON ─────────────────────────
    lang_naive_rss: dict[str, float] = {}
    vis_naive_rss:  dict[str, float] = {}
    for ds, m in lang_metrics.get("naive", {}).get("description_per_dataset", {}).items():
        v = _to_float(m.get("rouge_ss") if isinstance(m, dict) else None)
        if v is not None:
            lang_naive_rss[ds] = v
    for ds, m in vis_metrics.get("naive", {}).get("description_per_dataset", {}).items():
        v = _to_float(m.get("rouge_ss") if isinstance(m, dict) else None)
        if v is not None:
            vis_naive_rss[ds] = v

    # ── Build per-class summary table ─────────────────────────────────────────
    lang_keys = set(zip(lf["agml_dataset"], lf["class_label"], lf["model"])) if not lf.empty else set()
    vis_keys  = set(zip(vf["agml_dataset"], vf["class_label"], vf["model"])) if not vf.empty else set()
    all_keys  = sorted(lang_keys | vis_keys)

    if not all_keys:
        st.info("No data for the selected filters.")
        return

    summary_rows = []
    for ds, cls, model in all_keys:
        l_sub = lf[(lf["agml_dataset"] == ds) & (lf["class_label"] == cls) & (lf["model"] == model)] if not lf.empty else pd.DataFrame()
        v_sub = vf[(vf["agml_dataset"] == ds) & (vf["class_label"] == cls) & (vf["model"] == model)] if not vf.empty else pd.DataFrame()

        l_row = l_sub.iloc[0] if not l_sub.empty else {}

        desc_gt = (l_row.get("lang_desc_gt") if isinstance(l_row, dict) else l_row.get("lang_desc_gt", "")) or ""
        if not desc_gt and not v_sub.empty:
            desc_gt = v_sub.iloc[0].get("vis_desc_gt", "") or ""

        # Dataset-level F1 from metrics JSON
        lang_ds_f1 = lang_metrics.get("models", {}).get(model, {}).get("classification_per_dataset", {}).get(ds, {}).get("f1_macro")
        vis_ds_f1  = vis_metrics.get("models", {}).get(model, {}).get("classification_per_dataset", {}).get(ds, {}).get("f1_macro")

        # Dataset-level judged F1 computed from CSV rows (more reliable than JSON)
        lang_ds_jf1 = _ds_judge_acc(lang_df, ds, model, "lang_cls_judged_correct")
        vis_ds_jf1  = _ds_judge_acc(vis_df,  ds, model, "vis_cls_judged_correct")

        # Per-class accuracy
        lang_cls_acc  = _per_class_acc(l_sub, "lang_cls_resolved_correct") if not l_sub.empty else None
        lang_cls_jacc = _per_class_acc(l_sub, "lang_cls_judged_correct")   if not l_sub.empty else None
        vis_cls_acc   = _per_class_acc(v_sub, "vis_cls_resolved_correct")  if not v_sub.empty else None
        vis_cls_jacc  = _per_class_acc(v_sub, "vis_cls_judged_correct")    if not v_sub.empty else None

        # ROUGE-SS
        lang_rss = l_row.get("lang_rouge_ss") if not isinstance(l_row, dict) else (l_row.get("lang_rouge_ss") if l_row else None)
        vis_rss  = round(v_sub["vis_rouge_ss"].dropna().astype(float).mean(), 3) if not v_sub.empty and "vis_rouge_ss" in v_sub else None

        summary_rows.append({
            "dataset": ds, "class": cls, "model": model,
            "_desc_gt": desc_gt,
            "lang_ds_f1":   round(lang_ds_f1, 3)  if lang_ds_f1  is not None else None,
            "lang_ds_jf1":  lang_ds_jf1,
            "vis_ds_f1":    round(vis_ds_f1, 3)   if vis_ds_f1   is not None else None,
            "vis_ds_jf1":   vis_ds_jf1,
            "lang_cls_acc":  lang_cls_acc,
            "lang_cls_jacc": lang_cls_jacc,
            "vis_cls_acc":   vis_cls_acc,
            "vis_cls_jacc":  vis_cls_jacc,
            "lang_rss": round(lang_rss, 3) if lang_rss is not None else None,
            "vis_rss":  vis_rss,
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── Render summary table ──────────────────────────────────────────────────
    st.markdown("### Class overview")
    st.caption(f"{len(summary_df)} class × model combinations")

    display_cols = {
        "dataset":       "Dataset",
        "class":         "Class",
        "model":         "Model",
        "lang_ds_f1":    "Lang F1",
        "lang_ds_jf1":   "Lang F1 (judged)",
        "vis_ds_f1":     "Vision F1",
        "vis_ds_jf1":    "Vision F1 (judged)",
        "lang_cls_acc":  "Lang cls acc.",
        "lang_cls_jacc": "Lang cls acc. (judged)",
        "vis_cls_acc":   "Vision cls acc.",
        "vis_cls_jacc":  "Vision cls acc. (judged)",
        "lang_rss":      "Lang ROUGE-SS",
        "vis_rss":       "Vision ROUGE-SS",
    }
    disp = summary_df[[c for c in display_cols if c in summary_df.columns]].rename(columns=display_cols)

    numeric_disp = [v for k, v in display_cols.items() if k not in ("dataset", "class", "model")]
    numeric_disp = [c for c in numeric_disp if c in disp.columns]
    for col in numeric_disp:
        disp[col] = pd.to_numeric(disp[col], errors="coerce")

    st.dataframe(
        disp.style.background_gradient(subset=numeric_disp, cmap="RdYlGn", vmin=0, vmax=1)
            .format({c: "{:.3f}" for c in numeric_disp}, na_rep="—"),
        use_container_width=True,
    )

    st.divider()

    # ── Class detail view ─────────────────────────────────────────────────────
    st.markdown("### Class detail")

    all_classes = sorted(summary_df["class"].unique())
    sel_cls = st.selectbox("Class", all_classes, key="cls_sel")

    cls_models = sorted(summary_df[summary_df["class"] == sel_cls]["model"].unique())
    sel_cls_model = st.selectbox("Model", cls_models, key="cls_model_sel")

    row_info = summary_df[(summary_df["class"] == sel_cls) & (summary_df["model"] == sel_cls_model)]
    if row_info.empty:
        st.info("No data for this combination.")
        return
    row_info = row_info.iloc[0]
    ds_label = row_info["dataset"]

    st.markdown(
        f"**{sel_cls}** &nbsp; "
        f"<span style='color:#aaa;font-size:0.85em'>{ds_label} · {sel_cls_model}</span>",
        unsafe_allow_html=True,
    )

    # Metric pills
    pills = (
        _pill("Lang F1",              row_info.get("lang_ds_f1"))
        + _pill("Lang F1 judged",     row_info.get("lang_ds_jf1"), color="#3498db")
        + _pill("Lang cls acc.",      row_info.get("lang_cls_acc"))
        + _pill("Lang cls judged",    row_info.get("lang_cls_jacc"), color="#3498db")
        + _pill("Lang ROUGE-SS",      row_info.get("lang_rss"))
        + " &nbsp;&nbsp; "
        + _pill("Vis F1",             row_info.get("vis_ds_f1"))
        + _pill("Vis F1 judged",      row_info.get("vis_ds_jf1"), color="#3498db")
        + _pill("Vis cls acc.",       row_info.get("vis_cls_acc"))
        + _pill("Vis cls judged",     row_info.get("vis_cls_jacc"), color="#3498db")
        + _pill("Vis ROUGE-SS",       row_info.get("vis_rss"))
    )
    st.markdown(pills, unsafe_allow_html=True)

    desc_gt = row_info.get("_desc_gt", "")
    if desc_gt:
        with st.expander("Reference description (ground truth)", expanded=False):
            st.markdown(desc_gt)

    st.divider()

    # ── Language row ─────────────────────────────────────────────────────────
    l_sub = lf[(lf["agml_dataset"] == ds_label) & (lf["class_label"] == sel_cls) & (lf["model"] == sel_cls_model)] if not lf.empty else pd.DataFrame()

    st.markdown("#### Language Assessment")
    if l_sub.empty:
        st.info("No language assessment data.")
    else:
        l_row = l_sub.iloc[0]
        lang_baseline_rss = lang_naive_rss.get(ds_label)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Classification**")
            st.markdown(
                f"<div style='padding:10px;background:#1e1e1e;border-radius:6px;"
                f"min-height:50px'>{l_row.get('lang_cls_response') or '—'}</div>",
                unsafe_allow_html=True,
            )
            resolved_ok = _to_bool(l_row.get("lang_cls_resolved_correct"))
            judged_ok   = _to_bool(l_row.get("lang_cls_judged_correct"))
            st.markdown(
                "<small>" + _correct_badge(resolved_ok, judged_ok) + "</small>",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Description**")
            st.markdown(
                f"<div style='padding:10px;background:#1e1e1e;border-radius:6px;"
                f"min-height:50px'>{l_row.get('lang_desc_response') or '—'}</div>",
                unsafe_allow_html=True,
            )
            rss = _to_float(l_row.get("lang_rouge_ss"))
            r1  = _to_float(l_row.get("lang_rouge1"))
            baseline_note = f" vs baseline {lang_baseline_rss:.3f}" if lang_baseline_rss is not None else ""
            st.markdown(
                f"<small style='color:#aaa'>"
                + _desc_pill("ROUGE-SS" + baseline_note, rss, lang_baseline_rss)
                + (_pill("ROUGE-1", r1) if r1 is not None else "")
                + "</small>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Vision rows (one per image) ───────────────────────────────────────────
    v_sub = vf[(vf["agml_dataset"] == ds_label) & (vf["class_label"] == sel_cls) & (vf["model"] == sel_cls_model)] if not vf.empty else pd.DataFrame()

    st.markdown("#### Vision Assessment")
    if v_sub.empty:
        st.info("No vision assessment data.")
    else:
        vis_baseline_rss = vis_naive_rss.get(ds_label)

        n_resolved = int(v_sub["vis_cls_resolved_correct"].dropna().astype(float).sum()) if "vis_cls_resolved_correct" in v_sub else 0
        n_judged   = int(v_sub["vis_cls_judged_correct"].dropna().astype(float).sum())   if "vis_cls_judged_correct" in v_sub else 0
        n_has_judge = int(v_sub["vis_cls_judged_correct"].notna().sum())                  if "vis_cls_judged_correct" in v_sub else 0
        n_total = len(v_sub)

        caption = f"{n_total} images · {n_resolved}/{n_total} correct (resolved)"
        if n_has_judge:
            caption += f" · {n_judged}/{n_has_judge} correct (judged)"
        st.caption(caption)

        for _, vrow in v_sub.iterrows():
            img_name    = vrow.get("image_name", "—")
            resolved_ok = _to_bool(vrow.get("vis_cls_resolved_correct"))
            judged_ok   = _to_bool(vrow.get("vis_cls_judged_correct"))

            # Expander icon: prefer judged if available, else resolved
            primary_ok = judged_ok if judged_ok is not None else resolved_ok
            icon  = "✓" if primary_ok else ("✗" if primary_ok is not None else "?")
            color = "#2ecc71" if primary_ok else ("#e74c3c" if primary_ok is not None else "#888")

            with st.expander(f"{icon}  {img_name}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Classification**")
                    st.markdown(
                        f"<div style='padding:10px;background:#1e1e1e;border-radius:6px;"
                        f"min-height:40px'>{vrow.get('vis_cls_response') or '—'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<small>" + _correct_badge(resolved_ok, judged_ok) + "</small>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown("**Description**")
                    desc_resp = vrow.get("vis_desc_response") or "—"
                    st.markdown(
                        f"<div style='padding:10px;background:#1e1e1e;border-radius:6px;"
                        f"min-height:40px'>{desc_resp}</div>",
                        unsafe_allow_html=True,
                    )
                    rss = _to_float(vrow.get("vis_rouge_ss"))
                    r1  = _to_float(vrow.get("vis_rouge1"))
                    baseline_note = f" vs {vis_baseline_rss:.3f}" if vis_baseline_rss is not None else ""
                    st.markdown(
                        "<small style='color:#aaa'>"
                        + _desc_pill("ROUGE-SS" + baseline_note, rss, vis_baseline_rss)
                        + (_pill("ROUGE-1", r1) if r1 is not None else "")
                        + "</small>",
                        unsafe_allow_html=True,
                    )


# ── Descriptions tab ──────────────────────────────────────────────────────────

def render_descriptions(lang_df: pd.DataFrame) -> None:
    """Fast, image-free browser for per-class language description responses."""
    if lang_df.empty:
        st.warning("No language assessment CSV files found. Run the language stage first.")
        return

    desc_cols = ["model", "agml_dataset", "class_label", "task_type",
                 "lang_desc_response", "lang_desc_gt", "lang_rouge_ss", "lang_rouge1"]
    missing = [c for c in desc_cols if c not in lang_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    df = lang_df[desc_cols].copy()
    df = df[df["lang_desc_response"].notna() & (df["lang_desc_response"] != "")]
    if df.empty:
        st.info("No description responses found in language assessment data.")
        return

    # ── Sidebar filters ───────────────────────────────────────────────────────
    all_models   = sorted(df["model"].unique())
    all_datasets = sorted(df["agml_dataset"].unique())
    all_tasks    = sorted(df["task_type"].dropna().unique())

    sel_model   = st.sidebar.selectbox("Model (desc)",   ["(all)"] + all_models,   key="desc_model")
    sel_dataset = st.sidebar.selectbox("Dataset (desc)", ["(all)"] + all_datasets, key="desc_dataset")
    sel_task    = st.sidebar.selectbox("Task (desc)",    ["(all)"] + all_tasks,    key="desc_task")

    mask = pd.Series(True, index=df.index)
    if sel_model   != "(all)": mask &= df["model"]        == sel_model
    if sel_dataset != "(all)": mask &= df["agml_dataset"] == sel_dataset
    if sel_task    != "(all)": mask &= df["task_type"]    == sel_task
    df = df[mask].reset_index(drop=True)

    if df.empty:
        st.info("No rows match the selected filters.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown(
        f"### Description responses &nbsp; "
        f"<span style='color:#aaa;font-size:0.8em'>{len(df)} classes</span>",
        unsafe_allow_html=True,
    )

    disp = df[["agml_dataset", "class_label", "model", "task_type",
               "lang_rouge_ss", "lang_rouge1"]].copy()
    disp.columns = ["Dataset", "Class", "Model", "Task", "ROUGE-SS", "ROUGE-1"]
    for col in ("ROUGE-SS", "ROUGE-1"):
        disp[col] = pd.to_numeric(disp[col], errors="coerce")

    st.dataframe(
        disp.style
            .background_gradient(subset=["ROUGE-SS", "ROUGE-1"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"ROUGE-SS": "{:.3f}", "ROUGE-1": "{:.3f}"}, na_rep="—"),
        use_container_width=True,
        height=300,
    )

    st.divider()

    # ── Detail view ───────────────────────────────────────────────────────────
    st.markdown("### Browse by class")

    all_classes = sorted(df["class_label"].unique())
    sel_cls = st.selectbox("Class", all_classes, key="desc_cls")

    cls_rows = df[df["class_label"] == sel_cls]
    cls_models = sorted(cls_rows["model"].unique())
    sel_cls_model = st.selectbox("Model", cls_models, key="desc_cls_model")

    row = cls_rows[cls_rows["model"] == sel_cls_model]
    if row.empty:
        st.info("No data for this combination.")
        return
    row = row.iloc[0]

    rss = _to_float(row.get("lang_rouge_ss"))
    r1  = _to_float(row.get("lang_rouge1"))
    st.markdown(
        f"**{sel_cls}** &nbsp; "
        f"<span style='color:#aaa;font-size:0.85em'>"
        f"{row['agml_dataset']} · {sel_cls_model}</span> &nbsp; "
        + _pill("ROUGE-SS", rss) + _pill("ROUGE-1", r1),
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model response**")
        st.markdown(
            f"<div style='padding:12px;background:#1e1e1e;border-radius:6px;"
            f"min-height:80px;white-space:pre-wrap'>"
            f"{row.get('lang_desc_response') or '—'}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Ground truth**")
        st.markdown(
            f"<div style='padding:12px;background:#1e1e1e;border-radius:6px;"
            f"min-height:80px;white-space:pre-wrap'>"
            f"{row.get('lang_desc_gt') or '—'}</div>",
            unsafe_allow_html=True,
        )


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AgML Results Viewer",
        page_icon="🌾",
        layout="wide",
    )
    st.title("🌾 AgML Results Viewer")

    lang_df, vis_df           = load_assessment_data(str(_LANG_DIR), str(_VIS_DIR))
    lang_metrics, vis_metrics = load_metrics_json(str(_LANG_DIR), str(_VIS_DIR))

    tab_summary, tab_explorer, tab_desc = st.tabs(
        ["📊 Summary", "🔍 Class Explorer", "📝 Descriptions"]
    )

    with tab_summary:
        render_summary(str(_RES_CSV))

    with tab_explorer:
        render_class_explorer(lang_df, vis_df, lang_metrics, vis_metrics)

    with tab_desc:
        render_descriptions(lang_df)


if __name__ == "__main__":
    main()
