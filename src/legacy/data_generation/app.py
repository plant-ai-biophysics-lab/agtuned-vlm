import json
import re
import string
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="Description Verifier")

# Allow --output_dir to be passed on the command line:
#   streamlit run app.py -- --output_dir /path/to/outputs
_DEFAULT_OUTPUT_DIR = "outputs/data_generation"
_args = sys.argv[1:]
if "--output_dir" in _args:
    _idx = _args.index("--output_dir")
    _DEFAULT_OUTPUT_DIR = _args[_idx + 1] if _idx + 1 < len(_args) else _DEFAULT_OUTPUT_DIR

_output_dir = Path(_DEFAULT_OUTPUT_DIR)
DEFAULT_REF_DB = str(_output_dir / "reference_db.json")
DEFAULT_CSV = str(_output_dir / "descriptions_corrected.csv")


@st.cache_data
def load_reference_db(path: str):
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("entries", {})
    except FileNotFoundError:
        return None


@st.cache_data
def load_descriptions_csv(path: str):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def highlight_words_in_text(text: str, word_set: set, tag: str = "mark", style: str = "") -> str:
    """Wrap words from word_set found in text with the given HTML tag."""
    if not text:
        return ""
    tokens = re.split(r"(\W+)", str(text))
    result = []
    for token in tokens:
        clean = token.strip(string.punctuation).lower()
        if clean and clean in word_set:
            if tag == "mark":
                result.append(f'<mark style="background:#ffe066">{token}</mark>')
            else:
                result.append(f"<b>{token}</b>")
        else:
            result.append(token)
    return "".join(result)


def _find_sample_image(output_dir: Path, agml_dataset: str, class_label: str):
    slug = re.sub(r"[^a-z0-9]+", "_", class_label.strip().lower()).strip("_")
    path = output_dir / "sample_images" / agml_dataset / f"{slug}.jpg"
    return path if path.exists() else None


def word_set_from(text: str) -> set:
    if not text or (isinstance(text, float)):
        return set()
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
                 "at", "to", "for", "and", "or", "but", "with", "as", "by", "it",
                 "its", "this", "that", "which", "from", "be", "has", "have", "had"}
    return {w for w in words if w not in stopwords and len(w) > 2}


# --- Sidebar ---
st.sidebar.header("Output Directory")
output_dir_input = st.sidebar.text_input(
    "outputs/ path", value=str(_output_dir),
    help="Directory containing reference_db.json and descriptions_corrected.csv",
)
_resolved = Path(output_dir_input)
ref_path = str(_resolved / "reference_db.json")
csv_path = str(_resolved / "descriptions_corrected.csv")
st.sidebar.caption(f"`reference_db.json` → `{ref_path}`")
st.sidebar.caption(f"`descriptions_corrected.csv` → `{csv_path}`")

tab1, tab2, tab3 = st.tabs(["Reference Database", "Corrected Descriptions", "Stats"])

# ===========================================================================
# TAB 1: Reference Database
# ===========================================================================
with tab1:
    entries = load_reference_db(ref_path)
    if entries is None:
        st.warning(f"File not found: {ref_path}")
    else:
        rows = []
        for key, e in entries.items():
            rows.append({
                "_key": key,
                "Dataset": e.get("agml_dataset", ""),
                "Crop": e.get("crop", ""),
                "Class": e.get("class_label", ""),
                "Task": e.get("task_type", ""),
                "Status": e.get("status", ""),
                "# Sources": len(e.get("sources", [])),
            })
        df_ref = pd.DataFrame(rows)

        all_tasks = sorted(df_ref["Task"].dropna().unique().tolist())
        all_statuses = sorted(df_ref["Status"].dropna().unique().tolist())

        sel_tasks = st.sidebar.multiselect("Task type (Ref DB)", all_tasks, default=all_tasks, key="ref_task")
        sel_statuses = st.sidebar.multiselect("Status (Ref DB)", all_statuses, default=all_statuses, key="ref_status")

        mask = df_ref["Task"].isin(sel_tasks) & df_ref["Status"].isin(sel_statuses)
        df_filtered = df_ref[mask].reset_index(drop=True)

        display_cols = ["Dataset", "Crop", "Class", "Task", "Status", "# Sources"]
        event = st.dataframe(
            df_filtered[display_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        sel_rows = event.selection.rows
        if sel_rows:
            idx = sel_rows[0]
            key = df_filtered.iloc[idx]["_key"]
            entry = entries[key]
            st.markdown("---")
            img_col, info_col = st.columns([1, 3])
            with img_col:
                img_path = _find_sample_image(_resolved, entry.get("agml_dataset", ""), entry.get("class_label", ""))
                if img_path:
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.caption("No sample image")
            with info_col:
                st.subheader(f"{entry.get('crop', '')} — {entry.get('class_label', '')}")
            sources = entry.get("sources", [])
            if not sources:
                st.info("No sources found for this entry.")
            for i, src in enumerate(sources):
                org = src.get("organization", "Unknown")
                title = src.get("title", "No title")
                url = src.get("url", "")
                excerpt = src.get("excerpt", "")
                with st.expander(f"{title}"):
                    st.markdown(
                        f'<span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;font-size:0.8em">{org}</span>',
                        unsafe_allow_html=True,
                    )
                    if url:
                        st.markdown(f"[{url}]({url})")
                    if excerpt:
                        st.text_area("Excerpt", value=excerpt, height=150, disabled=True, key=f"exc_{key}_{i}")

# ===========================================================================
# TAB 2: Corrected Descriptions
# ===========================================================================
with tab2:
    df_desc = load_descriptions_csv(csv_path)
    if df_desc is None:
        st.warning(f"File not found: {csv_path}")
    else:
        all_tasks2 = sorted(df_desc["task_type"].dropna().unique().tolist()) if "task_type" in df_desc.columns else []
        all_statuses2 = sorted(df_desc["status"].dropna().unique().tolist()) if "status" in df_desc.columns else []

        sel_tasks2 = st.sidebar.multiselect("Task type (Descriptions)", all_tasks2, default=all_tasks2, key="desc_task")
        sel_statuses2 = st.sidebar.multiselect("Status (Descriptions)", all_statuses2, default=all_statuses2, key="desc_status")

        mask2 = pd.Series([True] * len(df_desc))
        if "task_type" in df_desc.columns and sel_tasks2:
            mask2 = mask2 & df_desc["task_type"].isin(sel_tasks2)
        if "status" in df_desc.columns and sel_statuses2:
            mask2 = mask2 & df_desc["status"].isin(sel_statuses2)

        df_desc_filtered = df_desc[mask2].reset_index(drop=True)

        hide_cols = {"excerpt_used", "source_urls"}
        show_cols = [c for c in df_desc_filtered.columns if c not in hide_cols]

        event2 = st.dataframe(
            df_desc_filtered[show_cols],
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        sel_rows2 = event2.selection.rows
        if sel_rows2:
            idx2 = sel_rows2[0]
            row = df_desc_filtered.iloc[idx2]

            st.markdown("---")
            img_col2, info_col2 = st.columns([1, 3])
            with img_col2:
                img_path2 = _find_sample_image(_resolved, str(row.get("agml_dataset", "")), str(row.get("class_label", "")))
                if img_path2:
                    st.image(str(img_path2), use_container_width=True)
                else:
                    st.caption("No sample image")
            with info_col2:
                st.subheader(f"{row.get('crop', '')} — {row.get('class_label', '')}")

            original = str(row.get("compressed_description", "")) if pd.notna(row.get("compressed_description")) else ""
            corrected = str(row.get("corrected_description", "")) if pd.notna(row.get("corrected_description")) else ""
            excerpt = str(row.get("excerpt_used", "")) if pd.notna(row.get("excerpt_used")) else ""
            source_urls = str(row.get("source_urls", "")) if pd.notna(row.get("source_urls")) else ""

            col_orig, col_corr = st.columns(2)

            with col_orig:
                st.markdown("**Original description**")
                st.markdown(
                    f'<div style="background:#f0f0f0;padding:10px;border-radius:6px">{original}</div>',
                    unsafe_allow_html=True,
                )

            with col_corr:
                st.markdown("**Corrected description**")
                excerpt_words = word_set_from(excerpt)
                highlighted_corrected = highlight_words_in_text(corrected, excerpt_words, tag="mark")
                st.markdown(
                    f'<div style="padding:10px;border:1px solid #ddd;border-radius:6px">{highlighted_corrected}</div>',
                    unsafe_allow_html=True,
                )

            if excerpt:
                st.markdown("**Source excerpt** (words matching corrected description are **bolded**)")
                corrected_words = word_set_from(corrected)
                highlighted_excerpt = highlight_words_in_text(excerpt, corrected_words, tag="b")
                st.markdown(
                    f'<div style="background:#f9f9e8;padding:10px;border-radius:6px;font-size:0.9em">{highlighted_excerpt}</div>',
                    unsafe_allow_html=True,
                )

            if source_urls:
                st.markdown("**Sources**")
                for url in source_urls.split(","):
                    url = url.strip()
                    if url:
                        st.markdown(f"- [{url}]({url})")

# ===========================================================================
# TAB 3: Stats
# ===========================================================================
with tab3:
    df_stats = load_descriptions_csv(csv_path)
    if df_stats is None:
        st.warning(f"File not found: {csv_path}")
    else:
        total = len(df_stats)
        n_corrected = (df_stats["status"] == "corrected").sum() if "status" in df_stats.columns else 0
        n_no_source = (df_stats["status"] == "no_source").sum() if "status" in df_stats.columns else 0
        n_error = (df_stats["status"] == "error").sum() if "status" in df_stats.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total entries", total)
        c2.metric("Corrected", n_corrected)
        c3.metric("No source", n_no_source)
        c4.metric("Error", n_error)

        st.markdown("---")

        if "task_type" in df_stats.columns and "status" in df_stats.columns:
            st.subheader("Count by Task Type x Status")
            pivot = df_stats.groupby(["task_type", "status"]).size().unstack(fill_value=0)
            st.bar_chart(pivot)

        if "agml_dataset" in df_stats.columns:
            st.subheader("Count by AgML Dataset")
            dataset_counts = df_stats["agml_dataset"].value_counts()
            st.bar_chart(dataset_counts)

        st.markdown("---")
        st.subheader("Paper-ready summary")
        pct_corrected = n_corrected / total * 100 if total else 0
        pct_no_source = n_no_source / total * 100 if total else 0

        lines = [
            f"Total classes: {total}",
            f"Corrected via reference sources: {n_corrected} ({pct_corrected:.1f}%)",
            f"No source found (original kept): {n_no_source} ({pct_no_source:.1f}%)",
        ]
        if "task_type" in df_stats.columns and "status" in df_stats.columns:
            lines.append("")
            lines.append("By task type:")
            for tt, grp in df_stats.groupby("task_type"):
                n_c = (grp["status"] == "corrected").sum()
                n_t = len(grp)
                lines.append(f"  {tt}: {n_c}/{n_t} corrected ({n_c/n_t*100:.1f}%)")
        if "agml_dataset" in df_stats.columns and "status" in df_stats.columns:
            lines.append("")
            lines.append("By dataset:")
            for ds, grp in df_stats.groupby("agml_dataset"):
                n_c = (grp["status"] == "corrected").sum()
                n_t = len(grp)
                lines.append(f"  {ds}: {n_c}/{n_t} corrected ({n_c/n_t*100:.1f}%)")

        st.code("\n".join(lines), language=None)
