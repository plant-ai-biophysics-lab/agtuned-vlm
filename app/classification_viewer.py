"""
Classification Viewer — browse agml_classification.jsonl records: which
dataset/class each came from, the grounding sources used, and the final
synthesized description.

Works even before Step 2 has run: if agml_classification.jsonl doesn't exist
yet, falls back to building the table straight from reference_db.json (Step
1's output alone) — Dataset/Crop/Class/Status/Sources, no description yet.

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


def rows_from_reference_db(entries: dict):
    """Step-1-only fallback table: no synthesized description yet, but
    Dataset/Crop/Class/status/source-count are all available straight from
    reference_db.json without needing agml_classification.jsonl at all."""
    rows = []
    for entry in entries.values():
        rows.append({
            "Dataset": entry.get("agml_dataset", ""),
            "Crop": entry.get("crop", ""),
            "Class": entry.get("class_label", ""),
            "Task type": entry.get("task_type", ""),
            "Grounding status": entry.get("status", ""),
            "Description": "",
            "Source URLs": [s.get("url", "") for s in entry.get("sources", [])],
            "Image names": [],
        })
    return pd.DataFrame(rows)


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

step2_available = df is not None
if not step2_available:
    if not reference_db:
        st.warning(f"Neither file found:\n- `{cls_path}`\n- `{ref_path}`\n\nRun Step 1 (build-reference-db) at least.")
        st.stop()
    st.info("`agml_classification.jsonl` not found yet (Step 2 hasn't run) — showing Step 1's "
            "reference_db.json only. Sources are available; descriptions aren't yet.")
    df = rows_from_reference_db(reference_db)

st.sidebar.caption(f"`agml_classification.jsonl` → `{cls_path}`"
                    + ("" if step2_available else "  (not found)"))
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

grounded_statuses = {"grounded", "grounded_web_and_vision", "grounded_vision_only", "grounded_web_only"}
c1, c2, c3 = st.columns(3)
c1.metric("Total classes", len(df))
c2.metric("Shown", len(df_filtered))
c3.metric("Grounded" if step2_available else "Found (Step 1)",
          df["Grounding status"].isin(grounded_statuses | {"found"}).sum())

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
    elif not step2_available:
        st.info("Not generated yet — run Step 2 (data-gen) to synthesize descriptions.")
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
