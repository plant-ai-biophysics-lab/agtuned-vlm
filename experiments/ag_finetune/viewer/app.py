"""
Training data viewer — Streamlit app.

Browse processed datasets by stage and purpose, with before/after toggle
showing raw vs converted samples.

Usage:
  streamlit run experiments/ag_finetune/viewer/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

OUT_DIR = Path(__file__).parent.parent / "outputs"

# ── Stage metadata ──────────────────────────────────────────────────────────
STAGES = {
    "stage1_agrillm_filtered.jsonl": {
        "label":   "Stage 1a — AgriLLM Filtered",
        "purpose": "SFT Seed (Q&A)",
        "day":     "Day 1",
        "description": (
            "Filtered and deduplicated agrillm-train-146k. "
            "Non-agricultural content removed, length-filtered, prompt scaffolding stripped."
        ),
        "before_fields": ["system_prompt", "turns"],
        "after_fields":  ["question", "answer", "system"],
    },
    "stage2_chi_dpo_pairs.jsonl": {
        "label":   "Stage 1c — Chi-Metrics DPO Pairs",
        "purpose": "DPO Alignment",
        "day":     "Day 1",
        "description": (
            "Preference pairs from Farmer.Chat interaction logs. "
            "High-score (≥4) responses paired against low-score (≤2) responses per topic."
        ),
        "before_fields": ["original_message", "message_response", "overall_score"],
        "after_fields":  ["chosen_prompt", "chosen_response", "rejected_response",
                          "chosen_score", "rejected_score", "topic"],
    },
    "stage3_gardian_passages.jsonl": {
        "label":   "Stage 1b — GARDIAN Passages",
        "purpose": "RAG QA Corpus",
        "day":     "Day 1",
        "description": (
            "Chunked passages from CGIAR GARDIAN research publications. "
            "Used for document-grounded QA generation (requires gated dataset access)."
        ),
        "before_fields": ["title", "text"],
        "after_fields":  ["passage", "chunk_id", "crop", "region"],
    },
    "stage4_reasoning_chains.jsonl": {
        "label":   "Stage 2a — Reasoning Chains",
        "purpose": "SFT Reasoning (STaR)",
        "day":     "Day 2",
        "description": (
            "CoT reasoning chains bootstrapped from agrillm Q&A pairs via frontier model. "
            "Each row adds a step-by-step chain between question and answer."
        ),
        "before_fields": ["question", "answer"],
        "after_fields":  ["question", "chain", "answer"],
    },
    "stage5_verifier_data.jsonl": {
        "label":   "Stage 3 — Verifier Data",
        "purpose": "Verifier / Critic Training",
        "day":     "Day 3",
        "description": (
            "Multi-turn (question, wrong_pred → correction) triples for verifier training. "
            "Includes hard negatives generated from agrillm seed Q&A."
        ),
        "before_fields": ["question", "wrong_pred", "correct"],
        "after_fields":  ["messages"],
    },
}

PURPOSE_COLORS = {
    "SFT Seed (Q&A)":           "#d4edda",
    "DPO Alignment":             "#fff3cd",
    "RAG QA Corpus":             "#d1ecf1",
    "SFT Reasoning (STaR)":     "#e2d9f3",
    "Verifier / Critic Training":"#f8d7da",
}


@st.cache_data
def load_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def render_field(label: str, value) -> None:
    if isinstance(value, list):
        st.markdown(f"**{label}**")
        for i, item in enumerate(value):
            with st.expander(f"Turn {i}"):
                st.json(item)
    elif isinstance(value, dict):
        st.markdown(f"**{label}**")
        st.json(value)
    else:
        text = str(value)
        if len(text) > 300:
            st.markdown(f"**{label}**")
            st.text_area("", text, height=120, key=f"{label}_{hash(text)}")
        else:
            st.markdown(f"**{label}:** {text}")


def main() -> None:
    st.set_page_config(
        page_title="AgriLLM Training Data Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("AgriLLM Training Data Viewer")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        # Stage selector
        available = {k: v for k, v in STAGES.items() if (OUT_DIR / k).exists()}
        missing   = {k: v for k, v in STAGES.items() if not (OUT_DIR / k).exists()}

        if not available:
            st.warning(f"No stage output files found in:\n`{OUT_DIR}`\n\nRun the data_prep scripts first.")
            return

        stage_labels = {v["label"]: k for k, v in available.items()}
        selected_label = st.selectbox("Stage", list(stage_labels.keys()))
        selected_file  = stage_labels[selected_label]
        meta = STAGES[selected_file]

        # Purpose filter
        st.markdown("---")
        st.markdown(f"**Purpose:** {meta['purpose']}")
        st.markdown(f"**Day:** {meta['day']}")

        # Before/After toggle
        st.markdown("---")
        view_mode = st.radio("View", ["After (processed)", "Before (raw)", "Side by side"])

        # Sample navigation
        st.markdown("---")
        rows = load_file(OUT_DIR / selected_file)
        n = len(rows)
        st.markdown(f"**{n:,} rows**")

        if n == 0:
            st.warning("File is empty.")
            return

        idx = st.number_input("Row index", min_value=0, max_value=n - 1, value=0, step=1)
        st.button("Random sample", on_click=lambda: None)  # trigger re-render

        # Missing stages notice
        if missing:
            st.markdown("---")
            st.markdown("**Not yet generated:**")
            for k, v in missing.items():
                st.markdown(f"- {v['label']}")

    # ── Main panel ───────────────────────────────────────────────────────────
    color = PURPOSE_COLORS.get(meta["purpose"], "#f0f0f0")
    st.markdown(
        f'<div style="background:{color};padding:12px;border-radius:8px;margin-bottom:12px">'
        f'<b>{meta["label"]}</b> &nbsp;·&nbsp; {meta["purpose"]} &nbsp;·&nbsp; {meta["day"]}<br/>'
        f'<small>{meta["description"]}</small></div>',
        unsafe_allow_html=True,
    )

    row = rows[int(idx)]

    if view_mode == "After (processed)":
        st.subheader("Processed Sample")
        for field in meta["after_fields"]:
            if field in row:
                render_field(field, row[field])

    elif view_mode == "Before (raw)":
        st.subheader("Raw Sample")
        raw = row.get("raw", {})
        if raw:
            for field in meta["before_fields"]:
                if field in raw:
                    render_field(field, raw[field])
        else:
            st.info("No raw data stored for this stage (not available from streaming source).")
            for field in meta["before_fields"]:
                if field in row:
                    render_field(field, row[field])

    else:  # Side by side
        col_before, col_after = st.columns(2)
        with col_before:
            st.subheader("Before (raw)")
            raw = row.get("raw", {})
            source = raw if raw else row
            for field in meta["before_fields"]:
                if field in source:
                    render_field(field, source[field])
        with col_after:
            st.subheader("After (processed)")
            for field in meta["after_fields"]:
                if field in row:
                    render_field(field, row[field])

    # ── Stats strip ──────────────────────────────────────────────────────────
    st.markdown("---")
    stats_file = OUT_DIR / selected_file.replace(".jsonl", "_stats.json").replace("stage", "stage")
    # Try matching stats file by stage number
    stage_num = selected_file.split("_")[0]
    stats_candidates = list(OUT_DIR.glob(f"{stage_num}_*stats*.json"))
    if stats_candidates:
        stats = json.loads(stats_candidates[0].read_text())
        st.subheader("Processing Stats")
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k.replace("_", " ").title(), f"{v:,}" if isinstance(v, int) else v)


if __name__ == "__main__":
    main()
