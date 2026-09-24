# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding.

## Prerequisites

```bash
uv sync
```

## Paper 1

**Paper:** [Vision-language models know more about agriculture than they show and rubric-grounded verifications close the gap](https://arxiv.org/abs/2609.09417)

### Experiments Pipeline

| Stage | Doc | What it does |
|---|---|---|
| 0a | [docs/stage0a_datasets.md](docs/stage0a_datasets.md) | Download & sample datasets from Project-AgML (HuggingFace) |
| 0b | [docs/stage0_data_generation.md](docs/stage0_data_generation.md) | (OPTIONAL) Fetch grounding sources (Wikipedia + web search) then synthesize classification JSONL from them |
| 1 | [docs/stage1_linear_probe.md](docs/stage1_linear_probe.md) | H1: test vision encoder feature separability across backbones |
| 2 | [docs/stage2_lower_upper_bounds.md](docs/stage2_lower_upper_bounds.md) | Lower (vision-only) vs. upper (description-augmented) classification bounds |
| 3 | [docs/stage3_cot.md](docs/stage3_cot.md) | Chain-of-thought: self-generated description vs. structured multi-step reasoning |
| 4 | [docs/stage4_verifier.md](docs/stage4_verifier.md) | Single-round verifier: K structured-criteria candidates ranked by a PPT tournament, argmax reported, vs. a full PPT-tournament baseline |

### Viewers

```bash
streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
```

Browse Stage 0b's output: dataset/crop/class per record, the grounding sources fetched for it, and the synthesized description. Works right after Step 1 alone (before descriptions exist) too — see [app/README.md](app/README.md).

## Paper 2

**Paper:** TBA

### Experiments Pipeline

| Stage | Doc | What it does |
|---|---|---|
| 1 | [docs/stage1_judge_improvements.md](docs/stage1_judge_improvements.md) | LLM judge leniency (Phase 1), self-consistency (Phase 2), and template sensitivity (Phase 3, `hybrid` template) on the thinking-enabled classification checkpoint — judge thinking on by default; includes a human-QA review export for hand-checking judge verdicts |
| 2 | [docs/stage2_baseline_rubric.md](docs/stage2_baseline_rubric.md) | Baseline with rubric: same thinking-enabled checkpoint, classification question wrapped through a fill-in-the-blank criteria chain before the final answer (Paper 1 Appendix A's "Manual CoT," rerun on the current pipeline) — includes multi-GPU generation-with-inline-judging via `--num_shards`; Step 3 briefly revisits `verifier-cot` (now also `--num_shards`-capable) with its sum-not-max scoring fix + rewritten pairwise prompt applied |
| 3 | [docs/stage3_react_baseline.md](docs/stage3_react_baseline.md) | ReAct baseline: a tool-using agent (`read`/`glob`/`grep`/`write` + agml-agent's `retrieve`/`search_agml`) over the model's own native tool calling via vLLM — one checkpoint row per episode (resumable), a shared per-model trajectory log, and a Streamlit viewer (`app/react_baseline_viewer.py`); starts with a 3-task text-only smoke test, real heldout-image harness not built yet |

### Viewers

```bash
streamlit run app/react_baseline_viewer.py -- --output_dir outputs_final/react_baseline/
```

Browse Stage 3's ReAct baseline output — per-episode task/result/trajectory, filterable by `stop_reason`, with per-episode timing. See [docs/stage3_react_baseline.md](docs/stage3_react_baseline.md).
