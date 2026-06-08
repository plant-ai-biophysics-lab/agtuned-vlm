"""
Layer Explorer — browse a HuggingFace VLM's full module tree and save pre/post-projector
layer selections to src/config/layer_selections.yaml for use by the linear probe.

Usage:
    streamlit run app/layer_explorer.py

Tabs:
  1. Table   — searchable flat list of all modules with type, shape, param counts
  2. Diagram — interactive tree diagram; click any node to assign it to pre/post slot

Memory:  ~200 MB RAM — model loaded on "meta" device (shapes only, no weights, no GPU).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import yaml
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

_CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "layer_selections.yaml"

# Slot colours (consistent across both tabs)
_SLOT_COLORS = {"pre": "#4C72B0", "post": "#DD8452"}  # blue / orange


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_selections() -> dict:
    if _CONFIG_PATH.exists():
        return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    return {}


def _save_selections(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))


# ── Layer introspection ────────────────────────────────────────────────────────

def _shape_info(module: nn.Module) -> tuple[str, str]:
    """Return (in_shape, out_shape) strings from static weight attributes — no forward pass."""
    if isinstance(module, nn.Linear):
        return str(module.in_features), str(module.out_features)
    if isinstance(module, nn.Conv2d):
        ks = f"{module.kernel_size[0]}×{module.kernel_size[1]}"
        return f"C={module.in_channels}", f"C={module.out_channels} k={ks}"
    if isinstance(module, nn.ConvTranspose2d):
        return f"C={module.in_channels}", f"C={module.out_channels}"
    if isinstance(module, nn.Embedding):
        return str(module.num_embeddings), str(module.embedding_dim)
    if isinstance(module, nn.LayerNorm):
        s = tuple(module.normalized_shape)
        return str(s), str(s)
    if isinstance(module, nn.GroupNorm):
        return f"C={module.num_channels}", f"C={module.num_channels}"
    if isinstance(module, nn.MultiheadAttention):
        return str(module.embed_dim), str(module.embed_dim)
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        return str(module.input_size), str(module.hidden_size)
    return "—", "—"


@st.cache_resource(show_spinner="Loading model architecture…")
def _build_table(model_id: str) -> pd.DataFrame:
    """
    Load model on meta device and return a DataFrame of all named modules.

    Meta device allocates only Python objects — parameters have shape/dtype but no
    data — so this uses ~200 MB RAM regardless of model size, and needs no GPU.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    try:
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
    except Exception:
        with torch.device("meta"):
            model = AutoModel.from_config(config)

    rows: list[dict] = []
    for path, module in model.named_modules():
        if not path:
            continue  # skip root
        is_leaf      = len(list(module.children())) == 0
        own_params   = sum(p.numel() for p in module.parameters(recurse=False))
        total_params = sum(p.numel() for p in module.parameters(recurse=True))
        in_s, out_s  = _shape_info(module)
        rows.append({
            "path":         path,
            "type":         type(module).__name__,
            "in_shape":     in_s,
            "out_shape":    out_s,
            "own_params":   own_params,
            "total_params": total_params,
            "depth":        path.count("."),
            "is_leaf":      is_leaf,
        })

    return pd.DataFrame(rows)


# ── Tree layout ────────────────────────────────────────────────────────────────

def _tree_layout(
    df: pd.DataFrame, max_depth: int
) -> tuple[dict[str, tuple[float, float]], list[tuple[str, str]]]:
    """
    Compute a Reingold–Tilford-style top-down layout.

    Returns:
        pos   — {path: (x, y)}  x ∈ [0, 1], y = −depth (root at y=0)
        edges — [(parent_path, child_path), ...]
    """
    subset   = df[df["depth"] <= max_depth]
    all_paths = set(subset["path"].tolist())

    # parent → [children] (sorted for deterministic layout)
    children: dict[str, list[str]] = defaultdict(list)
    for path in sorted(all_paths):
        parent = ".".join(path.split(".")[:-1])
        children[parent].append(path)
    for k in children:
        children[k].sort()

    # Leaf count for proportional x-spacing
    def _leaves(p: str) -> int:
        ch = children.get(p, [])
        return sum(_leaves(c) for c in ch) if ch else 1

    # Recursive layout — root="" is virtual and not drawn
    pos: dict[str, tuple[float, float]] = {}

    def _layout(path: str, x0: float, x1: float, depth: int) -> None:
        if path:  # skip virtual root
            pos[path] = ((x0 + x1) / 2, -depth)
        ch = children.get(path, [])
        if not ch:
            return
        total = sum(_leaves(c) for c in ch)
        cursor = x0
        for child in ch:
            share = _leaves(child) / total * (x1 - x0)
            _layout(child, cursor, cursor + share, depth + 1)
            cursor += share

    _layout("", 0.0, 1.0, 0)

    # Edge list (only between nodes that are both in pos)
    edges = [
        (parent, child)
        for parent, ch_list in children.items()
        for child in ch_list
        if child in pos and (parent in pos or parent == "")
    ]
    return pos, edges


# ── Diagram builder ────────────────────────────────────────────────────────────

def _build_diagram(
    df: pd.DataFrame,
    max_depth: int,
    pre_path: str,
    post_path: str,
    model_name: str,
) -> go.Figure:
    pos, edges = _tree_layout(df, max_depth)
    if not pos:
        return go.Figure().update_layout(title="No modules at this depth.")

    # ── Edge traces (grey lines, not selectable) ──────────────────────────────
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for parent, child in edges:
        px, py = pos.get(parent, (0.5, 0)) if parent else (0.5, 0)
        cx, cy = pos[child]
        edge_x += [px, cx, None]
        edge_y += [py, cy, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#d0d0d0", width=1),
        hoverinfo="none",
        showlegend=False,
    )

    # ── Node traces (one scatter point per module) ────────────────────────────
    # Separate traces: selected-pre, selected-post, normal — so legend is clean
    groups: dict[str, dict] = {
        "pre":    {"x": [], "y": [], "text": [], "cd": [], "size": [], "color": _SLOT_COLORS["pre"]},
        "post":   {"x": [], "y": [], "text": [], "cd": [], "size": [], "color": _SLOT_COLORS["post"]},
        "normal": {"x": [], "y": [], "text": [], "cd": [], "size": [], "color": []},
    }

    for path, (x, y) in pos.items():
        rows = df[df["path"] == path]
        if rows.empty:
            continue
        row  = rows.iloc[0]
        name = path.split(".")[-1]
        tip  = f"{name}<br><span style='color:#888'>{row['type']}</span>"
        if path == pre_path:
            g = groups["pre"]
            g["size"].append(18)
        elif path == post_path:
            g = groups["post"]
            g["size"].append(18)
        else:
            g = groups["normal"]
            # shade by depth: top levels lighter, deep layers darker
            depth = path.count(".")
            shade = max(60, 180 - depth * 22)
            g["size"].append(10)
            g["color"].append(f"rgb({shade}, {shade + 30}, {shade + 60})")
        g["x"].append(x)
        g["y"].append(y)
        g["text"].append(tip)
        g["cd"].append(path)

    def _make_trace(g: dict, name: str, color) -> go.Scatter:
        marker_kw: dict = dict(
            size=g["size"],
            line=dict(color="white", width=1.5),
        )
        if isinstance(color, list):
            marker_kw["color"] = color
        else:
            marker_kw["color"] = color
        return go.Scatter(
            x=g["x"], y=g["y"],
            mode="markers",
            marker=marker_kw,
            text=g["text"],
            hovertemplate="%{text}<br><i>%{customdata}</i><extra></extra>",
            customdata=g["cd"],
            name=name,
            showlegend=name != "normal",
        )

    traces = [
        edge_trace,
        _make_trace(groups["normal"], "modules",       groups["normal"]["color"]),
        _make_trace(groups["pre"],    "pre-projector", _SLOT_COLORS["pre"]),
        _make_trace(groups["post"],   "post-projector", _SLOT_COLORS["post"]),
    ]

    # ── Node labels (text annotations below each marker) ─────────────────────
    annotations = []
    for path, (x, y) in pos.items():
        rows = df[df["path"] == path]
        if rows.empty:
            continue
        name = path.split(".")[-1]
        annotations.append(dict(
            x=x, y=y - 0.06,
            text=f"<span style='font-size:9px'>{name}</span>",
            showarrow=False,
            xanchor="center",
            yanchor="top",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"<b>{model_name}</b>  —  module hierarchy  (depth ≤ {max_depth})",
            font=dict(size=14),
        ),
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.02, 1.02]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=50, b=30),
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        dragmode="select",         # single-click fires on_select
        clickmode="event+select",
    )
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_params(n: int) -> str:
    if n == 0:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _safe_idx(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _slot_card(label: str, path: str, df: pd.DataFrame, color: str) -> None:
    """Render a compact info card for a selected layer slot."""
    if path == "(none)":
        st.markdown(f"<span style='color:#aaa'>None selected</span>", unsafe_allow_html=True)
        return
    rows = df[df["path"] == path]
    if rows.empty:
        st.warning(f"`{path}` — not found in current model")
        return
    r = rows.iloc[0]
    st.markdown(
        f"<div style='border-left:4px solid {color};padding:6px 10px;border-radius:4px;"
        f"background:#f8f9fa;font-size:13px'>"
        f"<code>{path}</code><br>"
        f"<b>{r['type']}</b> · in <code>{r['in_shape']}</code> → out <code>{r['out_shape']}</code>"
        f"<br><span style='color:#888'>own {_fmt_params(int(r['own_params']))} params · "
        f"subtree {_fmt_params(int(r['total_params']))} params</span></div>",
        unsafe_allow_html=True,
    )


# ── Shared selection state ─────────────────────────────────────────────────────
# Both tabs read/write st.session_state["pre_select"] and ["post_select"].
# The selectbox widgets in Tab 1 use these as their keys, so the diagram tab
# can update them programmatically and they stay in sync.

def _current_pre()  -> str: return st.session_state.get("pre_select",  "(none)")
def _current_post() -> str: return st.session_state.get("post_select", "(none)")


# ── Sidebar: model input ───────────────────────────────────────────────────────

def _render_sidebar() -> str | None:
    """Render model ID input in the sidebar. Returns the active model ID or None."""
    with st.sidebar:
        st.title("🔬 Layer Explorer")
        st.caption("Load any HF model on meta device (~200 MB RAM, no GPU).")
        st.divider()
        raw_id = st.text_input(
            "HuggingFace model ID",
            value=st.session_state.get("model_id", ""),
            placeholder="e.g. google/gemma-4-26B-A4B-it",
        )
        load_btn = st.button("Load", type="primary", disabled=not raw_id.strip())

        if load_btn and raw_id.strip():
            new_id = raw_id.strip()
            if st.session_state.get("model_id") != new_id:
                _build_table.clear()
                # Reset selections when switching models
                st.session_state.pop("pre_select", None)
                st.session_state.pop("post_select", None)
            st.session_state.model_id = new_id

        active_id = st.session_state.get("model_id", "")

        if active_id:
            st.divider()
            st.markdown("**Selections**")
            pre  = _current_pre()
            post = _current_post()
            st.markdown(
                f"<span style='color:{_SLOT_COLORS['pre']}'>● Pre</span>&nbsp;&nbsp;"
                f"`{pre.split('.')[-1] if pre != '(none)' else 'none'}`",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='color:{_SLOT_COLORS['post']}'>● Post</span>&nbsp;&nbsp;"
                f"`{post.split('.')[-1] if post != '(none)' else 'none'}`",
                unsafe_allow_html=True,
            )

            st.divider()
            if st.button("💾 Save", type="primary", use_container_width=True):
                _do_save(active_id)

    return active_id or None


def _do_save(model_id: str) -> None:
    """Write current selections to layer_selections.yaml."""
    pre  = _current_pre()
    post = _current_post()
    if pre == "(none)" and post == "(none)":
        st.sidebar.warning("No layers selected.")
        return
    selections = _load_selections()
    entry: dict = {}
    if pre  != "(none)": entry["pre"]  = pre
    if post != "(none)": entry["post"] = post
    selections[model_id] = entry
    _save_selections(selections)
    st.sidebar.success(f"Saved to `{_CONFIG_PATH.name}`")


# ── Tab 1: Table view ──────────────────────────────────────────────────────────

def _render_table_tab(df: pd.DataFrame, all_paths: list[str]) -> None:
    fc1, fc2 = st.columns([4, 1])
    with fc1:
        search = st.text_input(
            "Search", placeholder="filter by path or module type — e.g. vision_tower, Linear"
        )
    with fc2:
        st.write("")
        leaf_only = st.checkbox("Leaf only", value=True)

    view = df.copy()
    if leaf_only:
        view = view[view["is_leaf"]]
    if search.strip():
        q = search.strip()
        view = view[
            view["path"].str.contains(q, case=False, na=False)
            | view["type"].str.contains(q, case=False, na=False)
        ]

    display = view[["path", "type", "in_shape", "out_shape"]].copy()
    display["own_params"]   = view["own_params"].apply(_fmt_params)
    display["total_params"] = view["total_params"].apply(_fmt_params)
    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=400)
    st.caption(f"Showing **{len(view):,}** of **{len(df):,}** modules")

    # ── Slot selectors ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Select embedding layers")
    st.caption(
        "The probe hooks the selected module and mean-pools its output. "
        "**Pre** = vision encoder output; **Post** = after the vision → LM projector."
    )

    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown(f"#### <span style='color:{_SLOT_COLORS['pre']}'>● Pre-projector</span>", unsafe_allow_html=True)
        st.selectbox(
            "pre_layer", all_paths,
            index=_safe_idx(all_paths, _current_pre()),
            label_visibility="collapsed",
            key="pre_select",
        )
        _slot_card("Pre", _current_pre(), df, _SLOT_COLORS["pre"])

    with col_post:
        st.markdown(f"#### <span style='color:{_SLOT_COLORS['post']}'>● Post-projector</span>", unsafe_allow_html=True)
        st.selectbox(
            "post_layer", all_paths,
            index=_safe_idx(all_paths, _current_post()),
            label_visibility="collapsed",
            key="post_select",
        )
        _slot_card("Post", _current_post(), df, _SLOT_COLORS["post"])


# ── Tab 2: Diagram view ────────────────────────────────────────────────────────

def _render_diagram_tab(df: pd.DataFrame, active_id: str) -> None:
    st.caption(
        "Tree diagram built from `named_modules()` — no forward pass, no GPU. "
        "Nodes = modules; edges = parent→child containment (computation order for VLMs). "
        "**Click a row in the table below** to assign it to the active slot — "
        "the diagram updates colours immediately."
    )

    # Controls row
    ctl1, ctl2 = st.columns([3, 2])
    with ctl1:
        slot_mode = st.radio(
            "Row click assigns to:",
            ["Pre-projector", "Post-projector"],
            horizontal=True,
            key="slot_mode",
        )
    with ctl2:
        max_depth = st.slider(
            "Depth", 1, 6, 2, key="tree_depth",
            help="How many nesting levels to show in diagram and table",
        )

    # Diagram (visual only — Plotly click events aren't captured by Streamlit;
    # selection is done via the dataframe below instead)
    fig = _build_diagram(
        df, max_depth,
        _current_pre(), _current_post(),
        active_id.split("/")[-1],
    )
    st.plotly_chart(fig, use_container_width=True, key="tree_chart")

    # ── Clickable module table ─────────────────────────────────────────────────
    st.caption(
        f"**Click a row** to assign it as "
        f"{'**Pre-projector**' if slot_mode == 'Pre-projector' else '**Post-projector**'} "
        f"(showing depth ≤ {max_depth})"
    )

    view = df[df["depth"] <= max_depth].copy()
    disp = view[["path", "type", "in_shape", "out_shape"]].copy()
    disp["own_params"]   = view["own_params"].apply(_fmt_params)
    disp["total_params"] = view["total_params"].apply(_fmt_params)
    disp = disp.reset_index(drop=True)

    selected = st.dataframe(
        disp,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=240,
        key="diagram_table",
    )

    # Assign selected row to the active slot.
    # We can't write to the selectbox keys (pre_select / post_select) here because
    # Tab 1 already instantiated those widgets earlier in this run.
    # Instead, stage the update in a "pending" key and apply it at the top of
    # main() on the next rerun — before any widget is created.
    if selected.selection.rows:
        row_idx      = selected.selection.rows[0]
        clicked_path = view.iloc[row_idx]["path"]
        slot_key     = "pre_select" if slot_mode == "Pre-projector" else "post_select"
        if st.session_state.get(slot_key) != clicked_path:
            st.session_state["_pending_slot_key"]   = slot_key
            st.session_state["_pending_slot_value"] = clicked_path
            st.rerun()

    # Current selection cards
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"**<span style='color:{_SLOT_COLORS['pre']}'>Pre-projector</span>**",
            unsafe_allow_html=True,
        )
        _slot_card("Pre", _current_pre(), df, _SLOT_COLORS["pre"])
    with c2:
        st.markdown(
            f"**<span style='color:{_SLOT_COLORS['post']}'>Post-projector</span>**",
            unsafe_allow_html=True,
        )
        _slot_card("Post", _current_post(), df, _SLOT_COLORS["post"])

    with st.expander("How to read this diagram"):
        st.markdown(
            "- **Nodes** are PyTorch modules (`nn.Module` subclasses).  \n"
            "- **Edges** connect parent modules to their direct children — for VLMs "
            "this maps to data flow: image tokens through the **vision tower** → "
            "**projector** → **language model**.  \n"
            "- Increase **Depth** to reveal inner layers (attention, MLP, etc.).  \n"
            "- This is a containment tree, not a traced computation graph — skip "
            "connections and attention patterns are not shown. For picking embedding "
            "tap points, the containment tree is sufficient."
        )


# ── Existing config viewer ─────────────────────────────────────────────────────

def _render_config_expander() -> None:
    selections = _load_selections()
    if not selections:
        return
    with st.expander("📄 Current layer_selections.yaml", expanded=False):
        st.code(yaml.dump(selections, default_flow_style=False), language="yaml")
        if st.button("Clear all saved selections", key="clear_btn"):
            _save_selections({})
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Layer Explorer", layout="wide", page_icon="🔬")

    # Apply any pending slot update from the diagram tab BEFORE widgets are instantiated.
    # (Streamlit forbids writing to a widget's session_state key after it has rendered.)
    if "_pending_slot_key" in st.session_state:
        key = st.session_state.pop("_pending_slot_key")
        val = st.session_state.pop("_pending_slot_value")
        st.session_state[key] = val

    active_id = _render_sidebar()
    if not active_id:
        st.info("Enter a model ID in the sidebar and click **Load** to inspect the architecture.")
        return

    try:
        df = _build_table(active_id)
    except Exception as exc:
        st.error(f"Failed to load `{active_id}`:\n\n```\n{exc}\n```")
        return

    leaf_count = int(df["is_leaf"].sum())
    st.success(f"`{active_id}` — **{len(df):,} modules** ({leaf_count:,} leaf)")

    all_paths = ["(none)"] + df["path"].tolist()

    tab1, tab2 = st.tabs(["📋 Table", "🌳 Diagram"])

    with tab1:
        _render_table_tab(df, all_paths)

    with tab2:
        _render_diagram_tab(df, active_id)

    _render_config_expander()


if __name__ == "__main__":
    main()
