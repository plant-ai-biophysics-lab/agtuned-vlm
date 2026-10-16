# Paper 2, Stage 4 — Detection Context/Skill Memory Construction

[← Back to README](../README.md)

## Step −1 — box-representation ablation (raw-coords / overlay / cropped)

```bash
uv run python main.py detection-multiclass \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/detection_multiclass/

uv run python main.py detection-skill-representation-ablation \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/detection_multiclass/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --n_samples 30
```

Decided: (a) raw coordinates as text, plain image — no overlay rendering. Args to note: `--ckpt` (required), `--n_samples`, `--crop_pad_frac`, `--out_dir`.

## Step 0 — held-out detection datasets

```bash
uv run python main.py download-heldout-detection
```

10 TRAIN + 5 VAL datasets marked `skip: true` in `detection_datasets.yaml`. The remaining ~39 are TEST — untouched, stay in the normal AgML detection eval pool. Args to note: none (dataset membership is fixed in-script, not a flag).

## Step 1 — zero-shot detection prediction

```bash
uv run python main.py heldout-detection-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split lessons \
    --output_dir outputs_final/heldout_detection_zeroshot_lessons/
```

Args to note: `--split` (`lessons`=TRAIN default, `eval`=VAL), `--skill_memory_path`, `--rerun`.

## Step 2 — explain traces

```bash
uv run python main.py heldout-detection-explain-traces \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_detection_zeroshot_lessons/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_dir outputs_final/heldout_detection_explain_traces/
```

QC (no model calls):

```bash
uv run python main.py heldout-detection-explain-traces-qc \
    --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --n_samples 20
```

Args to note: `--ckpt` (required), `--iou_threshold`, `--loose_iou_threshold`.

## Step 3 — summarize into skill_memory_detection.md

```bash
uv run python main.py heldout-detection-summarize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_detection_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv \
    --output_md outputs_final/heldout_detection_summarize/skill_memory_detection.md
```

Args to note: `--max_words_per_chunk`, `--max_words_per_reduce_group`, `--output_md`, `--rerun`.

## Step 3b — self-optimization loop

```bash
uv run python main.py heldout-detection-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --init_skill_path outputs_final/heldout_detection_summarize/skill_memory_detection.md \
    --max_rounds 10 \
    --feedback_prior_round \
    --output_dir outputs_final/heldout_detection_skill_optimize_feedback/
```

Args to note: `--init_skill_path` (required), `--max_rounds` (required), `--n_examples_per_bucket`, `--feedback_prior_round`, `--output_dir`. Winning round is written to `<output_dir>/skill_memory_best.md`.

## Step 4 — ablation

```bash
# Baseline, no skill file:
uv run python main.py heldout-detection-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --output_dir outputs_final/heldout_detection_zeroshot_eval_baseline/

# With skill file:
uv run python main.py heldout-detection-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --skill_memory_path outputs_final/heldout_detection_summarize/skill_memory_detection.md \
    --output_dir outputs_final/heldout_detection_zeroshot_eval_with_skill/
```

Args to note: `--split eval` (VAL — the 5 held-out VAL datasets, not the main AgML pool), distinct `--output_dir` per run.

## Step 5 — inject into detection-multiclass

```bash
uv run python main.py detection-multiclass \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path outputs_final/heldout_detection_skill_optimize_feedback/skill_memory_best.md \
    --output_dir outputs_final/detection_multiclass_with_skill/
```

Args to note: `--skill_memory_path` (applied to every active dataset this run touches — no per-row gate, unlike classification's disease/pest-only gate), `--output_dir` (must not collide with `outputs_final/detection_multiclass/`).

## Default args

### `detection-skill-representation-ablation`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--ckpt` | required |
| `--images_root` | `dataset/images_detection` |
| `--n_samples` | `30` |
| `--iou_threshold` | `0.5` |
| `--crop_pad_frac` | `0.25` |
| `--out_dir` | `outputs_final/detection_skill_representation_ablation/` |
| `--enable_thinking` | `False` |
| `--gen_max_tokens` | `512` |

### `download-heldout-detection`

No flags — TRAIN/VAL dataset membership is fixed in-script.

### `heldout-detection-zeroshot`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--manifest` | `dataset/detection_heldout_selected_images.json` |
| `--images_root` | `dataset/images_detection_heldout` |
| `--split` | `lessons` |
| `--skill_memory_path` | `None` |
| `--output_dir` | `outputs_final/heldout_detection_zeroshot/` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--gen_max_tokens` | `4096` |
| `--iou_threshold` | `0.5` |
| `--rerun` | `False` |

### `heldout-detection-explain-traces`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--ckpt` | required |
| `--images_root` | `dataset/images_detection_heldout` |
| `--output_dir` | `outputs_final/heldout_detection_explain_traces/` |
| `--iou_threshold` | `0.5` |
| `--loose_iou_threshold` | `0.7` |
| `--enable_thinking` | `False` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--gen_max_tokens` | `512` |
| `--chunk_size` | `64` |

### `heldout-detection-explain-traces-qc`

| Arg | Default |
|---|---|
| `--ckpt` | required |
| `--images_root` | `dataset/images_detection_heldout` |
| `--n_samples` | `20` |
| `--seed` | `42` |
| `--out_dir` | `outputs_final/heldout_detection_explain_traces_qc/` |

### `heldout-detection-summarize`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--ckpt` | required |
| `--output_dir` | `outputs_final/heldout_detection_summarize/` |
| `--output_md` | `outputs_final/heldout_detection_summarize/skill_memory_detection.md` |
| `--max_words_per_chunk` | `8000` |
| `--max_words_per_reduce_group` | `1500` |
| `--map_max_tokens` | `2048` |
| `--reduce_max_tokens` | `4096` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--rerun` | `False` |

### `heldout-detection-skill-optimize`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--init_skill_path` | required |
| `--max_rounds` | required |
| `--output_dir` | `outputs_final/heldout_detection_skill_optimize/` |
| `--round_log` | `None` (→ `<output_dir>/round_log.json`) |
| `--n_examples_per_bucket` | `10` |
| `--feedback_prior_round` | `False` |
| `--iou_threshold` | `0.5` |
| `--loose_iou_threshold` | `0.7` |
| `--enable_thinking` | `True` |
| `--edit_enable_thinking` | `False` |
| `--gen_max_tokens` | `4096` |
| `--edit_max_tokens` | `4096` |
| `--seed` | `42` |

### `heldout-detection-ablation-summary`

| Arg | Default |
|---|---|
| `--baseline_ckpt` | required |
| `--skill_ckpt` | required |
| `--out_dir` | `None` (→ `<skill_ckpt's own dir>/ablation/`) |
| `--title` | `""` |

### `detection-multiclass` (flags relevant to this stage)

| Arg | Default |
|---|---|
| `--model` | `None` |
| `--config` | `src/config/detection_datasets.yaml` |
| `--manifest` | `dataset/selected_images_detection_10.json` |
| `--images_root` | `dataset/images_detection` |
| `--output_dir` | `outputs_final/detection_multiclass/` |
| `--iou_threshold` | `0.5` |
| `--enable_thinking` | `True` |
| `--gen_max_tokens` | `4096` |
| `--skill_memory_path` | `None` |
| `--smoketest` | `False` |

## Open questions

1. Step 3b's self-optimization loop: rounds 0-4 real (VAL F1 0.26-0.28, non-monotonic, best round 4 at 0.2799) — not yet converged/concluded.
2. Step 4's optimized-skill ablation condition (3-way: baseline / not-optimized / optimized) not yet run for real, now that Step 3b is wired up.
3. Whether a `--quality_skill_memory_path`-style per-task gate is ever needed for detection — not currently, since detection has one task_type across all active datasets.
