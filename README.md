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
| 0 | [docs/stage0_judge_improvements.md](docs/stage0_judge_improvements.md) | Judge an existing baseline checkpoint with the pipeline-wide default `hybrid` template — one pass, never mutates the checkpoint in place, point `--output_dir` at a fresh folder per template wording you're trying |
| 1 | [docs/stage1_baseline_classification.md](docs/stage1_baseline_classification.md) | Baseline classification runs: the plain, thinking-enabled single-shot classification checkpoint (no rubric, no verifier) every later stage builds on |
| 2 | [docs/stage2_object_detection.md](docs/stage2_object_detection.md) | Baseline object detection runs (a Paper 1 leftover): builds a class-balanced 10-images/class manifest (`dataset/selected_images_detection_10.json`) from Project-AgML's detection-type datasets, then runs Detection-Baseline (direct box prediction, no reasoning) with IoU-based precision/recall/F1 + QC plots — the spatial-task analog of Stage 1. Rubric/verifier conditions deferred to Stage 6 |
| 3 | [docs/stage3_context_memory.md](docs/stage3_context_memory.md) | Context/skill memory construction: held out (never used for eval) Digital Green crop-disease images + expert-written `details` as a lesson-building split — zero-shot classify, reveal ground truth on EVERY row (both right and wrong), ask the model to explain the gap/confirm what worked, summarize the gathered traces (map-reduce, grouped by heuristic category) into a single skill_memory.md "what worked / what didn't / common diseases to look for" file. **Scope: disease/pest/nutrient-deficiency/healthy only** — the real file has zero content for quality grading or species/weed identification (DigiGreen's dataset doesn't cover those), so injection into `vision-baseline`/`verifier-cot` is gated to `task_type in {disease, pest_damage}`, never applied to other task_types. **Ablation status**: the original 4-iteration result (+0.0603 judge_accuracy, 0.3618→0.4221) no longer stands alone — a `--wrapper_only` condition (the injection wrapper's instructions with NO skill content at all) scored AS HIGH OR HIGHER in a follow-up run, and none of the pairwise differences among conditions clear statistical significance at n=199 (two-proportion z-tests, |z|<1.5 throughout) — read as "the wrapper rules account for most of the measured gain," not as a settled win for the skill content itself. An automated **Step 3b self-optimization loop** (`heldout-skill-optimize`) now exists to try to close that gap: one command runs N rounds of score→edit→rescore against the skill file directly (TRAIN drives the edit, VAL only selects the best round — never train-on-test), logging a mechanical diff + the editing model's own stated rationale + automated QC (catches a real channel-thought-leak bug + bare-category-word/common-section regressions) per round, selecting the best-scoring round at the end. New opt-in `--feedback_prior_round` flag (default off, so it's a one-flag A/B test against the original behavior): when on, each round's edit call is also shown the IMMEDIATELY-prior round's own changelog paired with the TRAIN scores that edit actually produced, so the editor can tell whether its last change helped before making another one — previously the changelog was write-only (logged for human reading, never fed back): `uv run python main.py heldout-skill-optimize --model RedHatAI/gemma-4-31B-it-FP8-dynamic --init_skill_path src/config/skill_memory_disease-pest.md --max_rounds 10 --feedback_prior_round --output_dir outputs_final/heldout_skill_optimize_feedback/`. `vision-baseline` injection (`--skill_memory_path`) is built; not yet run for real, `verifier-cot` injection still TODO. Run this stage to completion before starting real Stage 6 (detection) work, not in parallel |
| 4 | [docs/stage4_detection_context_memory.md](docs/stage4_detection_context_memory.md) | Detection context/skill memory construction — the detection analog of Stage 3's skill-memory pipeline, applied to `detection-multiclass` (not plain `detection-baseline` — real local metrics showed multiclass wins on F1/recall). The one new problem detection has that classification didn't: ground truth is `[x1,y1,x2,y2]` coordinates, not a text label, so "how do you show a model its own box mistake" needs its own answer, not an assumed one. **Step −1 resolved**: a real 30-sample qualitative comparison (`detection-skill-representation-ablation`) found the model handles raw coordinates + the plain image fine — no overlay rendering needed, also the cheapest option. **Step 0 built**: a real 3-way split by DATASET MEMBERSHIP (not a within-dataset random split) — 10 TRAIN + 5 VAL datasets (15 total, including 3 synthetic ones: `synthetic_cowpea_flower_detection`, `synthetic_cowpea_pod_detection`, `grape_detection_syntheticday`) marked `skip: true` and held out completely from Stage 2/6's own active eval pool (54 → 39); the remaining ~39 are TEST, read for a final confirmatory score only, never touched during lesson-generation or optimization. `download-heldout-detection` caches TRAIN+VAL and writes the lessons/eval split manifest. **Step 1 built** (`heldout-detection-zeroshot`): zero-shot `detection-multiclass` prediction on the lessons/eval pool, reusing `detection_baseline.py`'s own generation+scoring directly. **Step 2 built** (`heldout-detection-explain-traces`): box-level explain traces on the lessons split, ALL items not just misses (mirrors Stage 3's own "explain what worked too" choice) — one pass of `bbox_metrics.match_boxes` per row yields 4 `error_type` categories (`false_positive`/`missed`/`loose_match`, named after `detection_box_refine.py`'s own 3-criterion rubric, plus `worked`), one explanation call per box-level item, condition (a) only (Step −1's settled winner). No separate `wrong_class_label` category — `category_name` is deliberately never compared (matches `detection_baseline.py`'s own class-agnostic `score_row`); a first version did compare it and misflagged a correct "grape: ripe cluster" prediction against an uninformative upstream ground-truth label literally named `"1"` (`grape_detection_syntheticday`'s own `ClassLabel(names=['grape', '1'])`). **QC check** (`heldout-detection-explain-traces-qc`): no model calls, pure report generator over Step 2's checkpoint — renders a handful of samples (image + annotated pred/GT boxes + the model's own written explanation), round-robin sampled across every `error_type` present so no single category dominates the sample: `uv run python main.py heldout-detection-explain-traces-qc --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv --n_samples 20`. **Step 3 built** (`heldout-detection-summarize`): text-only map-reduce, same mechanic `heldout-summarize` uses — MAP groups Step 2's explanations by `error_type` (false_positive/missed/loose_match/worked) into word-bounded chunks (one summarization call per chunk), REDUCE merges every chunk summary into one `skill_memory_detection.md`; new rule vs. classification's own Step 3 — every bullet must describe a VISUAL/SPATIAL pattern, never a coordinate number: `uv run python main.py heldout-detection-summarize --model RedHatAI/gemma-4-31B-it-FP8-dynamic --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv`. **Step 3b built** (`heldout-detection-skill-optimize`): a SEPARATE script (not a parameterized version of `heldout-skill-optimize`, to avoid any risk to that actively-used classification script) that imports its genuinely domain-agnostic pieces (`parse_edit_output`, `diff_skill_text`) directly rather than duplicating them — same TRAIN/VAL loop/resume/selection discipline, swapped for detection's own F1 scoring and Step 2's `error_type` taxonomy (computed fresh each round via `enumerate_items`) instead of `judge_accuracy`/disease-pest-deficiency-healthy; miss/worked examples shown to the edit call are box-level coordinates/category/IoU, not a written explanation; `check_detection_rule_violations` is its own detection-specific QC (coordinate-citing bullets, "Common Objects" list), not a reuse of classification's bare-category-word check. Also supports `--feedback_prior_round`: `uv run python main.py heldout-detection-skill-optimize --model RedHatAI/gemma-4-31B-it-FP8-dynamic --init_skill_path outputs_final/heldout_detection_summarize/skill_memory_detection.md --max_rounds 10 --feedback_prior_round --output_dir outputs_final/heldout_detection_skill_optimize_feedback/`. **Step 4 built** (baseline vs. not-optimized skill; optimized-skill condition TBD once Step 3b is wired up for detection): `heldout-detection-zeroshot` gained `--skill_memory_path` (new `build_detection_skill_system_prompt` in `src/common/heldout_skill.py` — plain reference-material framing, deliberately WITHOUT classification's own empirically-tuned "firm rules" wrapper, since nothing equivalent has been validated for detection yet), and `heldout-detection-ablation-summary` is the detection analog of `heldout-ablation-summary` — precision/recall/F1 instead of `judge_accuracy`, macro-averaged same as `detection_final_metrics.py`'s own `_aggregate()`: `uv run python main.py heldout-detection-zeroshot --model RedHatAI/gemma-4-31B-it-FP8-dynamic --split eval --skill_memory_path outputs_final/heldout_detection_summarize/skill_memory_detection.md --output_dir outputs_final/heldout_detection_zeroshot_eval_with_skill/`. Step 5 still TODO |
| 5 | [docs/stage5_classification_verifier.md](docs/stage5_classification_verifier.md) | Classification verifier runs: `verifier-cot`'s new default (`--verifier_type simple`) — ONE trajectory per image (no K-candidate pool, no PPT tournament), graded against each rubric criterion independently on a 1-3 scale and summed/normalized to a `verifier_score` in [0, 1]; reports its point-biserial correlation with judged correctness, to test whether this cheaper signal is usable as a stopping/feedback gate (e.g. for Stage 7's ReAct agent) before adopting it there. `--from_ckpt` grades an existing Stage 1 checkpoint's trajectories directly instead of regenerating them |
| 6 | [docs/stage6_detection_verifier.md](docs/stage6_detection_verifier.md) | Detection verifier — the spatial-task analog of Stage 5, applied to object detection. Design not started yet (placeholder doc) |
| 7 | [docs/stage7_react_baseline.md](docs/stage7_react_baseline.md) | ReAct baseline: a tool-using agent (`read`/`glob`/`grep`/`write` + agml-agent's `retrieve`/`search_agml`) over the model's own native tool calling via vLLM — one checkpoint row per episode (resumable), a shared per-model trajectory log, and a Streamlit viewer (`app/react_baseline_viewer.py`); starts with a 3-task text-only smoke test, real heldout-image harness not built yet |

### Model support (Stage 3 context-memory pipeline)

`heldout-zeroshot`/`heldout-explain-traces`/`heldout-summarize`/`heldout-skill-optimize` (docs/stage3_context_memory.md) default to `RedHatAI/gemma-4-31B-it-FP8-dynamic`, but work with any model vLLM `0.20.1` supports — including Qwen (e.g. `Qwen/Qwen3.8-27B-FP8`, confirmed registered in this repo's pinned vLLM version, no upgrade needed).

- **`--chat_template_path`** defaults to the vendored `src/config/tool_chat_template_gemma4.jinja` (Gemma-4's own bundled template has no thinking support at all). Qwen's own bundled template already supports `enable_thinking` natively — pass `--chat_template_path ""` to use it instead of Gemma's template.
- **Thinking-trace extraction is model-agnostic** (`extract_thinking`, `src/common/metrics.py`) — tries Gemma-4's `<|channel>thought...<channel|>` format first, falls back to Qwen's `<think>...</think>` tags, so neither model leaks a raw thinking block into a checkpointed prediction/explanation/the skill-memory `.md` itself.
- **`--judge_model`** stays independent of `--model` — keep it on Gemma-4 (the default) so a Qwen-vs-Gemma generation comparison isn't also confounded by a different judge, unless you're deliberately testing judge choice too.

```bash
uv run python main.py heldout-zeroshot \
    --model Qwen/Qwen3.8-27B-FP8 \
    --chat_template_path "" \
    --split lessons
```

### Viewers

```bash
streamlit run app/react_baseline_viewer.py -- --output_dir outputs_final/react_baseline/
```

Browse Stage 7's ReAct baseline output — per-episode task/result/trajectory, filterable by `stop_reason`, with per-episode timing. See [docs/stage7_react_baseline.md](docs/stage7_react_baseline.md).
