"""ReAct Baseline Viewer — browse react-baseline's own output (src/baselines/
react_baseline.py): the per-model checkpoint CSV (one row per episode) and
the shared trajectory_log.jsonl (one line per model turn, tagged by
episode_key) it writes under <output_dir>/<safe_model_name>/.

Auto-discovers every ckpt_*.csv under --output_dir (rglob, not a fixed
single-level layout) — works whether you point it straight at a real
outputs_final/react_baseline/ root (one subfolder per model) or at some
other wrapper directory containing that same shape nested inside it.

Optional image preview: tries, in order, (1) --images_root/<agml_dataset>/
<class_label>/<image_name> (dataset/images/'s own real convention), then
(2) a reference_images/<agml_dataset>_<class_label>_<image_name> file next
to --output_dir — the naming a hand-built demo folder might use. Neither
existing is fine; the viewer just skips the image.

Usage:
    streamlit run app/react_baseline_viewer.py -- --output_dir outputs_final/react_baseline/
    streamlit run app/react_baseline_viewer.py -- --output_dir /path/to/some/folder --images_root dataset/images
"""
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.baselines.tool_call_formats import TOOL_CALL_EXTRACTORS  # pure stdlib (json/re) —
                                                                    # safe to import here, no vllm/torch pulled in

st.set_page_config(layout="wide", page_title="ReAct Baseline Viewer")

_args = sys.argv[1:]


def _flag(name: str, default: str) -> str:
    if name in _args:
        i = _args.index(name)
        return _args[i + 1] if i + 1 < len(_args) else default
    return default


_DEFAULT_OUTPUT_DIR = _flag("--output_dir", "outputs_final/react_baseline")
_DEFAULT_IMAGES_ROOT = _flag("--images_root", "dataset/images")

REACT_CKPT_FIELDS = [
    "agml_dataset", "class_label", "image_name", "task", "result", "summary", "justification",
    "stop_reason", "n_steps", "tools_used", "nudges_used", "wrap_up_nudged", "auto_verified", "error",
    "tool_call_format", "elapsed_seconds", "judge_gt_hint", "judged_correct", "judge_reasoning",
]


def episode_key(agml_dataset: str, class_label: str, image_name: str) -> str:
    """Same construction as react_baseline.py's own main() loop
    (`"/".join(filter(None, [...]))`) — MUST stay in sync with it, since
    this is how trajectory_log.jsonl lines get matched back to a
    checkpoint row (there's no foreign-key column, just this shared
    convention)."""
    return "/".join(filter(None, [agml_dataset, class_label, image_name]))


def _canonical_ckpt_path(model_dir: Path) -> Path:
    """react_baseline.py always writes the canonical checkpoint as
    ckpt_<model_dir.name>.csv (safe_model_name(args.model) == the model's
    own output subfolder name, by construction) — targeting that exact
    filename, not a glob("ckpt_*.csv"), matters because --rerun's own
    archives (ckpt_<model>.<timestamp>.bak.csv) ALSO match that glob
    pattern (starts with "ckpt_", ends with ".csv"); a plain glob's first
    result can be an archived, stale checkpoint instead of the current
    one — confirmed as a real bug here, not hypothetical."""
    return model_dir / f"ckpt_{model_dir.name}.csv"


@st.cache_data
def discover_model_dirs(output_dir: str) -> dict[str, str]:
    """{model_dir_name: model_dir_path} for every folder under output_dir
    (searched recursively) that contains a canonical ckpt_<name>.csv — so
    this works whether output_dir IS a model's own folder, the
    react_baseline/ root above several model folders, or some other
    wrapper directory. Deliberately does NOT match --rerun's own
    ckpt_<model>.<timestamp>.bak.csv archives — see _canonical_ckpt_path."""
    root = Path(output_dir)
    if not root.exists():
        return {}
    found = {}
    for ckpt_path in sorted(root.rglob("ckpt_*.csv")):
        model_dir = ckpt_path.parent
        if ckpt_path == _canonical_ckpt_path(model_dir):
            found[model_dir.name] = str(model_dir)
    return found


@st.cache_data
def load_episodes(model_dir: str) -> pd.DataFrame:
    ckpt_path = _canonical_ckpt_path(Path(model_dir))
    if not ckpt_path.exists():
        return pd.DataFrame(columns=REACT_CKPT_FIELDS)
    df = pd.read_csv(ckpt_path, dtype=str).fillna("")
    for col in REACT_CKPT_FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["episode_key"] = df.apply(
        lambda r: episode_key(r["agml_dataset"], r["class_label"], r["image_name"]), axis=1,
    )
    return df


@st.cache_data
def load_trajectory(model_dir: str) -> dict:
    """{episode_key: [ {step, raw_text, ...}, ... ]} — turns for an
    episode_key not present at all (e.g. checkpoint row exists but the log
    predates it, or vice versa) just come back as an empty list; the
    detail view below handles that by saying so rather than crashing."""
    path = Path(model_dir) / "trajectory_log.jsonl"
    turns: dict = {}
    if not path.exists():
        return turns
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns.setdefault(obj.get("episode_key", ""), []).append(obj)
    for key in turns:
        turns[key].sort(key=lambda o: o.get("step", 0))
    return turns


def find_image(agml_dataset: str, class_label: str, image_name: str, output_dir: str, images_root: str):
    if agml_dataset and class_label and image_name:
        candidate = Path(images_root) / agml_dataset / class_label / image_name
        if candidate.exists():
            return candidate
    if image_name:
        fallback_name = "_".join(filter(None, [agml_dataset, class_label, image_name]))
        candidate = Path(output_dir).parent / "reference_images" / fallback_name
        if candidate.exists():
            return candidate
    return None


# ── raw_text parsing for the Trajectory tab ─────────────────────────────────
# Reuses the SAME extractors (tool_call_formats.py) the loop itself calls at
# runtime, so what the viewer shows as "the tool calls this turn made" is
# guaranteed consistent with what react_agent.py actually parsed — not a
# second, independently-maintained parser that could silently drift from it.

# Gemma-4's thinking channel is always the first thing in a turn's raw_text
# (either the model emits <|channel>thought\n itself, or — when a prior turn
# ended in a tool response — the prompt already opened it implicitly and
# raw_text starts directly with the thinking prose) — so this only needs to
# match from the start of the string, not search anywhere inside it.
_GEMMA4_THINK_RE = re.compile(r"^(?:<\|channel>thought\n?)?(.*?)<channel\|>", re.DOTALL)
_HERMES_THINK_RE = re.compile(r"^<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_SPAN_RES = {
    "gemma4": re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    "hermes": re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
}


def parse_turn(raw_text: str, tool_call_format: str) -> dict:
    """{"thinking": str, "calls": [{"name", "arguments"}, ...], "prose": str}
    — prose is whatever's left after removing the thinking block and every
    matched tool_call span, e.g. a nudge-recovered explanation, or the full
    answer text on a turn that skipped calling `finish` (stop_reason=
    no_tool_call)."""
    think_re = _GEMMA4_THINK_RE if tool_call_format == "gemma4" else _HERMES_THINK_RE
    m = think_re.match(raw_text)
    thinking, remainder = (m.group(1).strip(), raw_text[m.end():]) if m else ("", raw_text)

    extractor = TOOL_CALL_EXTRACTORS.get(tool_call_format)
    calls = extractor(raw_text) if extractor else []

    span_re = _TOOL_CALL_SPAN_RES.get(tool_call_format)
    prose = span_re.sub("", remainder).strip() if span_re else remainder.strip()
    return {"thinking": thinking, "calls": calls, "prose": prose}


st.title("ReAct Baseline Viewer")

with st.sidebar:
    st.header("Source")
    output_dir = st.text_input("Output dir", value=_DEFAULT_OUTPUT_DIR)
    images_root = st.text_input("Images root (optional)", value=_DEFAULT_IMAGES_ROOT)
    model_dirs = discover_model_dirs(output_dir)
    if not model_dirs:
        st.warning(f"No ckpt_*.csv found under {output_dir!r} (searched recursively).")
        st.stop()
    model_name = st.selectbox("Model", sorted(model_dirs), index=0)
    model_dir = model_dirs[model_name]

episodes = load_episodes(model_dir)
trajectory = load_trajectory(model_dir)

if episodes.empty:
    st.info(f"{model_dir}: checkpoint exists but has no rows yet.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    stop_reasons = ["(all)"] + sorted(episodes["stop_reason"].unique().tolist())
    stop_reason_filter = st.selectbox("stop_reason", stop_reasons, index=0)
    judged_filter = st.selectbox("judged", ["(all)", "correct", "incorrect", "unjudged"], index=0)
    search = st.text_input("Search task/result contains")

filtered = episodes
if stop_reason_filter != "(all)":
    filtered = filtered[filtered["stop_reason"] == stop_reason_filter]
if judged_filter == "correct":
    filtered = filtered[filtered["judged_correct"] == "1"]
elif judged_filter == "incorrect":
    filtered = filtered[filtered["judged_correct"] == "0"]
elif judged_filter == "unjudged":
    filtered = filtered[filtered["judged_correct"] == ""]
if search:
    mask = filtered["task"].str.contains(search, case=False, na=False) | filtered["result"].str.contains(
        search, case=False, na=False
    )
    filtered = filtered[mask]

st.caption(f"{len(filtered)} / {len(episodes)} episode(s) — {model_dir}")

n_finished = (episodes["stop_reason"] == "finished").sum()
n_no_tool_call = (episodes["stop_reason"] == "no_tool_call").sum()
n_max_iter = (episodes["stop_reason"] == "max_iterations").sum()
n_errors = episodes["stop_reason"].isin(["context_overflow", "error"]).sum()
elapsed_numeric = pd.to_numeric(episodes["elapsed_seconds"], errors="coerce")
judged_mask = episodes["judged_correct"].isin(["0", "1"])
n_judged = int(judged_mask.sum())
judge_acc = (
    f"{100 * (episodes.loc[judged_mask, 'judged_correct'] == '1').mean():.0f}% ({n_judged})"
    if n_judged else "—"
)
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Total episodes", len(episodes))
c2.metric("Finished", int(n_finished))
c3.metric("No tool call", int(n_no_tool_call))
c4.metric("Max iterations", int(n_max_iter))
c5.metric("Errors", int(n_errors))
c6.metric("Judge accuracy", judge_acc)
c7.metric("Avg. time/episode", f"{elapsed_numeric.mean():.1f}s" if elapsed_numeric.notna().any() else "—")

overview_cols = [
    "agml_dataset", "class_label", "image_name", "stop_reason", "judged_correct", "n_steps",
    "elapsed_seconds", "tools_used", "nudges_used", "wrap_up_nudged", "auto_verified", "result",
]
display_df = filtered[overview_cols].copy()
display_df["result"] = display_df["result"].str.slice(0, 120)

selection = st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = selection["selection"]["rows"] if selection and selection.get("selection") else []
if not selected_rows:
    st.info("Select a row above to see its full task, result, and turn-by-turn trajectory.")
    st.stop()

row = filtered.iloc[selected_rows[0]]
st.divider()
st.subheader(f"{row['agml_dataset'] or '(ad-hoc)'} / {row['class_label'] or '—'} / {row['image_name']}")

col_img, col_info = st.columns([1, 2])
with col_img:
    img_path = find_image(row["agml_dataset"], row["class_label"], row["image_name"], output_dir, images_root)
    if img_path:
        st.image(str(img_path), caption=str(img_path))
    else:
        st.caption("(no image found — checked --images_root and a sibling reference_images/ folder)")
with col_info:
    judge_badge = (
        "✅ correct" if row["judged_correct"] == "1"
        else "❌ incorrect" if row["judged_correct"] == "0"
        else "⋯ unjudged"
    )
    st.markdown(f"**stop_reason:** `{row['stop_reason']}`  |  **n_steps:** {row['n_steps']}  |  "
                f"**elapsed:** {row['elapsed_seconds'] or '?'}s  |  "
                f"**tool_call_format:** `{row['tool_call_format']}`  |  "
                f"**tools_used:** {row['tools_used'] or '(none)'}  |  "
                f"**nudges_used:** {row['nudges_used'] or '0'}  |  "
                f"**wrap_up_nudged:** {row['wrap_up_nudged'] or 'False'}  |  "
                f"**auto_verified:** {row['auto_verified'] or 'False'}  |  "
                f"**judged:** {judge_badge}")
    if row["error"]:
        st.error(row["error"])

tab_task, tab_result, tab_judge, tab_trajectory = st.tabs(["Task", "Result", "Judge", "Trajectory"])
with tab_task:
    st.text(row["task"])
with tab_result:
    st.text(row["result"] or "(no result — episode did not finish)")
    if row["summary"] or row["justification"]:
        st.divider()
        st.markdown("**Summary (what was investigated):**")
        st.text(row["summary"] or "(none)")
        st.markdown("**Justification (why this answer):**")
        st.text(row["justification"] or "(none)")
with tab_judge:
    if not row["class_label"]:
        st.caption("No class_label on this episode (ad-hoc/text-only) — nothing to judge against.")
    elif row["judged_correct"] == "":
        st.caption("Not judged yet — run react-baseline again without --skip_judge to backfill.")
    else:
        if row["judged_correct"] == "1":
            st.success("✅ Judged correct")
        else:
            st.error("❌ Judged incorrect")
        st.markdown(f"**Ground truth shown to judge (judge_gt_hint):** {row['judge_gt_hint']}")
        st.markdown("**Judge's reasoning:**")
        st.text(row["judge_reasoning"] or "(no reasoning recorded)")
with tab_trajectory:
    key = row["episode_key"]
    turns = trajectory.get(key, [])
    if not turns:
        st.warning(f"No trajectory_log.jsonl turns found for episode_key {key!r}.")

    # "tool_response" entries (added alongside "generation" — see
    # react_agent.py's own log_path docstring) render nested under their
    # matching step below, not as their own top-level card. An OLDER
    # trajectory_log.jsonl (predating this) has no "type" key at all —
    # every one of its lines is implicitly "generation", so
    # observations_by_step just comes back empty and nothing breaks.
    observations_by_step: dict[int, list] = {}
    for turn in turns:
        if turn.get("type") == "tool_response":
            observations_by_step[turn.get("step")] = turn.get("observations", [])

    for turn in turns:
        if turn.get("type") == "tool_response":
            continue
        step = turn.get("step")
        raw_text = turn.get("raw_text", "")
        parsed = parse_turn(raw_text, row["tool_call_format"])
        st.markdown(f"**Step {step}**")

        if parsed["thinking"]:
            with st.expander("🤔 Thinking"):
                st.text(parsed["thinking"])

        for call in parsed["calls"]:
            if call["name"] == "finish":
                st.success(f"✅ **finish**\n\n{call['arguments'].get('result', '')}")
            elif call["name"] == "verify":
                st.info(f"🔎 **verify:** {call['arguments'].get('claim', '')}")
            else:
                args_str = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
                st.code(f"{call['name']}({args_str})", language=None)

        for obs in observations_by_step.get(step, []):
            with st.expander(f"↳ {obs.get('name')} response"):
                st.text(obs.get("content", ""))

        if parsed["prose"]:
            st.text(parsed["prose"])
        if not parsed["calls"] and not parsed["prose"] and not parsed["thinking"]:
            st.caption("(empty turn)")

        with st.expander("Raw text"):
            st.text(raw_text)
        st.divider()
