"""
Classification Viewer — browse agml_classification.jsonl records: which
dataset/class each came from, the grounding sources used, and the final
synthesized description.

Usage:
    streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Classification Viewer")

_args = sys.argv[1:]
_DEFAULT_OUTPUT_DIR = "outputs/data_generation"
if "--output_dir" in _args:
    _idx = _args.index("--output_dir")
    _DEFAULT_OUTPUT_DIR = _args[_idx + 1] if _idx + 1 < len(_args) else _DEFAULT_OUTPUT_DIR


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _ref_db_key(agml_dataset: str, class_label: str, crop: str) -> str:
    return f"{_slug(agml_dataset)}__{_slug(class_label)}__{_slug(crop)}"


@st.cache_data
def load_classification_jsonl(path: str):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                meta = r.get("metadata", {})
                rows.append({
                    "Dataset": meta.get("agml_dataset", ""),
                    "Crop": meta.get("crop", ""),
                    "Class": meta.get("class_label", ""),
                    "Task type": meta.get("task_type", ""),
                    "Grounding status": meta.get("grounding_status", ""),
                    "Description": meta.get("synthesized_description", ""),
                    "Source URLs": meta.get("source_urls", []),
                    "Image names": meta.get("image_names", []),
                })
        return pd.DataFrame(rows)
    except FileNotFoundError:
        return None


@st.cache_data
def load_reference_db(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("entries", {})
    except FileNotFoundError:
        return {}


def _find_sample_image(output_dir: Path, agml_dataset: str, class_label: str):
    path = output_dir / "sample_images" / agml_dataset / f"{_slug(class_label)}.jpg"
    return path if path.exists() else None


# --- Sidebar ---
st.sidebar.header("Output directory")
output_dir_input = st.sidebar.text_input(
    "outputs/ path", value=_DEFAULT_OUTPUT_DIR,
    help="Directory containing agml_classification.jsonl and reference_db.json",
)
output_dir = Path(output_dir_input)
cls_path = str(output_dir / "agml_classification.jsonl")
ref_path = str(output_dir / "reference_db.json")

df = load_classification_jsonl(cls_path)
reference_db = load_reference_db(ref_path)

if df is None:
    st.warning(f"File not found: {cls_path}")
    st.stop()

st.sidebar.caption(f"`agml_classification.jsonl` → `{cls_path}`")
st.sidebar.caption(f"`reference_db.json` → `{ref_path}`")

all_datasets = sorted(df["Dataset"].dropna().unique().tolist())
all_tasks = sorted(df["Task type"].dropna().unique().tolist())
all_statuses = sorted(df["Grounding status"].dropna().unique().tolist())

sel_datasets = st.sidebar.multiselect("Dataset", all_datasets, default=[])
sel_tasks = st.sidebar.multiselect("Task type", all_tasks, default=all_tasks)
sel_statuses = st.sidebar.multiselect("Grounding status", all_statuses, default=all_statuses)

mask = df["Task type"].isin(sel_tasks) & df["Grounding status"].isin(sel_statuses)
if sel_datasets:
    mask = mask & df["Dataset"].isin(sel_datasets)
df_filtered = df[mask].reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total classes", len(df))
c2.metric("Shown", len(df_filtered))
c3.metric("Grounded", (df["Grounding status"] == "grounded").sum())

st.markdown("---")

show_cols = ["Dataset", "Crop", "Class", "Task type", "Grounding status"]
event = st.dataframe(
    df_filtered[show_cols],
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

sel_rows = event.selection.rows
if sel_rows:
    row = df_filtered.iloc[sel_rows[0]]
    st.markdown("---")

    img_col, info_col = st.columns([1, 3])
    with img_col:
        img_path = _find_sample_image(output_dir, row["Dataset"], row["Class"])
        if img_path:
            st.image(str(img_path), use_container_width=True)
        else:
            st.caption("No sample image")
    with info_col:
        st.subheader(f"{row['Crop']} — {row['Class']}")
        st.caption(f"{row['Dataset']}  ·  {row['Task type']}  ·  status: {row['Grounding status']}")

    st.markdown("**Synthesized description**")
    if row["Description"]:
        st.markdown(
            f'<div style="background:#f0f0f0;padding:10px;border-radius:6px">{row["Description"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Empty — needs a manual fill (see grounding status above).")

    st.markdown("**Sources**")
    ref_key = _ref_db_key(row["Dataset"], row["Class"], row["Crop"])
    entry = reference_db.get(ref_key)
    sources = entry.get("sources", []) if entry else []
    if not sources:
        st.caption("No sources recorded for this entry.")
    for src in sources:
        title = src.get("title", "No title")
        org = src.get("organization", "Unknown")
        url = src.get("url", "")
        excerpt = src.get("excerpt", "")
        with st.expander(f"[{org}] {title}"):
            if url:
                st.markdown(f"[{url}]({url})")
            if excerpt:
                st.text_area("Excerpt", value=excerpt, height=150, disabled=True,
                              key=f"exc_{ref_key}_{url}")
