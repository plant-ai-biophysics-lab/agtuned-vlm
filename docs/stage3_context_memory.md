# Paper 2, Stage 3 — Context/Skill Memory Construction

[← Back to README](../README.md)

## Model support

`heldout-zeroshot`/`heldout-explain-traces`/`heldout-summarize`/`heldout-skill-optimize` default to `RedHatAI/gemma-4-31B-it-FP8-dynamic`, but work with any model vLLM `0.20.1` supports — including Qwen (e.g. `Qwen/Qwen3.8-27B-FP8`, confirmed registered in this repo's pinned vLLM version, no upgrade needed).

- **`--chat_template_path`** defaults to the vendored `src/config/tool_chat_template_gemma4.jinja` (Gemma-4's own bundled template has no thinking support at all). Qwen's own bundled template already supports `enable_thinking` natively — pass `--chat_template_path ""` to use it instead of Gemma's template.
- **Thinking-trace extraction is model-agnostic** (`extract_thinking`, `src/common/metrics.py`) — tries Gemma-4's `<|channel>thought...<channel|>` format first, falls back to Qwen's `<think>...</think>` tags, so neither model leaks a raw thinking block into a checkpointed prediction/explanation/the skill-memory `.md` itself.
- **`--judge_model`** defaults to whatever `--model` is (the model judges its own output) — pass `--judge_model` explicitly to judge with a different model, e.g. keeping Gemma-4 as judge while generating with Qwen so a Qwen-vs-Gemma comparison isn't also confounded by a different judge.

```bash
uv run python main.py heldout-zeroshot \
    --model Qwen/Qwen3.8-27B-FP8 \
    --chat_template_path "" \
    --split lessons
```

## Step 0 — download the held-out dataset (disease/pest)

```bash
uv run python main.py download-heldout-digitalgreen
```

Args to note: `--eval_frac` (lessons/eval split), `--seed`.

## Step 0b — download the held-out dataset (quality)

```bash
uv run python main.py download-heldout-quality
```

5 of AgML's 24 `quality_identification` datasets (`skip: true` in `src/config/datasets.yaml`) are held out as the quality-domain lesson-building source, never sampled into the main AgML eval: `VegNet_quality_classification`, `banana_bunch_maturity_classification`, `coffee_bean_quality_classification`, `pomegranate_quality_classification`, `riseholme_strawberry_classification_2021`.

## Step 0c — download the held-out dataset (species)

```bash
uv run python main.py download-heldout-species
```

iNatAg-mini (`Project-AgML/iNatAg-mini` on HuggingFace) — external to AgML's own 25 `species_identification` datasets (never wired into `datasets.yaml`, same separation as Digital Green/quality). 40 species (20 crop-labeled + 20 weed-labeled, seeded) downloaded whole (one `.zip` per species, ~3.48 GB), subsampled to `--images_per_species` (default 50) each for a 2,000-image local pool (1,600 lessons / 400 eval).

Args to note: `--n_crop_species`, `--n_weed_species`, `--images_per_species` (download size is fixed by species count — this only controls the local pool size), `--selection_log` (written once, then reused on every later run — the committed, definitive record of which species were selected, same role `skip: true` + `notes` plays for quality's datasets.yaml entries), `--eval_frac`, `--seed`.

## Step 1 — zero-shot prediction

```bash
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

Args to note: `--domain` (`disease_pest` default, `quality`, `species`), `--split` (`lessons` default, `eval` for Step 4), `--skill_memory_path`, `--manifest`/`--images_root` (must match `--domain`), `--rerun`.

- quality: add `--domain quality --manifest dataset/heldout_quality_selected_images.json --images_root dataset/images_heldout_quality --output_dir outputs_final/heldout_zeroshot_quality/`
- species: add `--domain species --manifest dataset/heldout_species_selected_images.json --images_root dataset/images_heldout_species --output_dir outputs_final/heldout_zeroshot_species/`

## Step 2 — reveal ground truth, every row

```bash
uv run python main.py heldout-explain-traces \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_zeroshot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Args to note: `--domain`, `--ckpt` (required), `--images_root` (must match `--domain`), `--gen_max_tokens` (bump if thinking traces truncate).

- quality: add `--domain quality --images_root dataset/images_heldout_quality --ckpt outputs_final/heldout_zeroshot_quality/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv --output_dir outputs_final/heldout_explain_traces_quality/`
- species: add `--domain species --images_root dataset/images_heldout_species --ckpt outputs_final/heldout_zeroshot_species/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv --output_dir outputs_final/heldout_explain_traces_species/`

## Step 3 — summarize into the skill `.md`

```bash
uv run python main.py heldout-summarize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Args to note: `--domain`, `--max_words_per_chunk`, `--output_md`, `--rerun`.

- quality: add `--domain quality --ckpt outputs_final/heldout_explain_traces_quality/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv --output_dir outputs_final/heldout_summarize_quality/ --output_md outputs_final/heldout_summarize_quality/skill_memory_quality.md`
- species: add `--domain species --ckpt outputs_final/heldout_explain_traces_species/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv --output_dir outputs_final/heldout_summarize_species/ --output_md outputs_final/heldout_summarize_species/skill_memory_species.md`

## Step 3b — self-optimization loop

```bash
uv run python main.py heldout-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --init_skill_path src/config/skill_memory_disease-pest.md \
    --max_rounds 10 \
    --output_dir outputs_final/heldout_skill_optimize/
```

Args to note: `--domain` (also forwarded into every scoring call this loop makes — `--manifest`/`--images_root` must match it), `--init_skill_path` (required), `--max_rounds` (required), `--n_examples_per_bucket`, `--feedback_prior_round`, `--round_log`, `--output_dir`. Winning round is written to `<output_dir>/skill_memory_best.md`.

- prior-round feedback: add `--feedback_prior_round --output_dir outputs_final/heldout_skill_optimize_feedback/`
- quality: add `--domain quality --manifest dataset/heldout_quality_selected_images.json --images_root dataset/images_heldout_quality --init_skill_path outputs_final/heldout_summarize_quality/skill_memory_quality.md --output_dir outputs_final/heldout_skill_optimize_quality/`
- species: add `--domain species --manifest dataset/heldout_species_selected_images.json --images_root dataset/images_heldout_species --init_skill_path outputs_final/heldout_summarize_species/skill_memory_species.md --output_dir outputs_final/heldout_skill_optimize_species/`

## Step 4 — ablation

```bash
# baseline, no skill file:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --output_dir outputs_final/heldout_zeroshot_eval_baseline/

# with skill file:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/heldout_zeroshot_eval_not_optimized_skill/
```

Args to note: `--domain`, `--split eval`, distinct `--output_dir` per run.

- quality: add `--domain quality --manifest dataset/heldout_quality_selected_images.json --images_root dataset/images_heldout_quality` to both, `--skill_memory_path outputs_final/heldout_skill_optimize_quality/skill_memory_best.md` on the skill run, `--output_dir outputs_final/heldout_zeroshot_quality_eval_{baseline,with_skill}/`
- species: add `--domain species --manifest dataset/heldout_species_selected_images.json --images_root dataset/images_heldout_species` to both, `--skill_memory_path outputs_final/heldout_skill_optimize_species/skill_memory_best.md` on the skill run, `--output_dir outputs_final/heldout_zeroshot_species_eval_{baseline,with_skill}/`

## Step 5 — inject into vision-baseline

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/
```

Args to note: `--skill_memory_path` (gated per-row to `task_type in {disease, pest_damage}`), `--quality_skill_memory_path` (gated per-row to `task_type == quality_identification`, independent flag/file), `--species_skill_memory_path` (gated per-row to `task_type == species_identification`, independent flag/file, added 2026-10 — same pattern as quality's flag) — all three can be set in the same run, each only ever applies to its own task_type. Also `--task_types` (restrict the run to specific task_type(s) instead of every active one — e.g. `--task_types disease pest_damage` to test only the disease/pest skill), `--output_dir` (must not collide with `outputs_final/vision_baseline_thinking/`).

- quality skill only: swap `--skill_memory_path ...` for `--quality_skill_memory_path outputs_final/heldout_skill_optimize_quality/skill_memory_best.md`
- species skill only: swap `--skill_memory_path ...` for `--species_skill_memory_path outputs_final/heldout_skill_optimize_species/skill_memory_best.md`
- any combination at once: pass multiple flags together — each only ever applies to its own task_type's rows

Sharded across GPUs:

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID
```

Merge shards (no `--skill_memory_path`/`--num_shards`/`--shard_index` needed — just merges whatever shard files are already in `--output_dir`):

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --merge_shards
```

## Step 6 (not built)

Repeat Steps 0-5 for object detection, after classification gains hold up. See [Stage 6](stage6_detection_verifier.md).

## Default args

### `heldout-zeroshot`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--domain` | `disease_pest` |
| `--manifest` | `dataset/heldout_selected_images.json` |
| `--images_root` | `dataset/images_heldout` |
| `--split` | `lessons` |
| `--skill_memory_path` | `None` |
| `--prompts` | `src/config/prompts.yaml` |
| `--output_dir` | `outputs_final/heldout_zeroshot/` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--gen_max_tokens` | `4096` |
| `--chunk_size` | `64` |
| `--quantize` | `None` |
| `--gpu_memory_utilization` | `0.9` |
| `--max_model_len` | `24576` |
| `--max_num_seqs` | `64` |
| `--kv_cache_dtype` | `None` |
| `--judge_model` | `None` (→ `--model`) |
| `--judge_confidence_threshold` | `1` |
| `--judge_chunk_size` | `200` |
| `--judge_max_tokens` | `1024` |
| `--judge_chat_template_path` | `None` |

### `heldout-explain-traces`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--domain` | `disease_pest` |
| `--ckpt` | required |
| `--images_root` | `dataset/images_heldout` |
| `--heldout_prompts` | `src/config/heldout_prompts.yaml` |
| `--output_dir` | `outputs_final/heldout_explain_traces/` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--gen_max_tokens` | `4096` |
| `--chunk_size` | `64` |
| `--seed` | `42` |
| `--gpu_memory_utilization` | `0.9` |
| `--max_model_len` | `16384` |
| `--max_num_seqs` | `64` |

### `heldout-summarize`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--domain` | `disease_pest` |
| `--ckpt` | required |
| `--heldout_prompts` | `src/config/heldout_prompts.yaml` |
| `--output_dir` | `outputs_final/heldout_summarize/` |
| `--output_md` | `outputs_final/heldout_summarize/skill_memory.md` |
| `--max_words_per_chunk` | `8000` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--map_max_tokens` | `2048` |
| `--reduce_max_tokens` | `4096` |
| `--seed` | `42` |
| `--gpu_memory_utilization` | `0.9` |
| `--max_model_len` | `24576` |
| `--max_num_seqs` | `64` |

### `heldout-skill-optimize`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--domain` | `disease_pest` |
| `--init_skill_path` | required |
| `--max_rounds` | required |
| `--output_dir` | `outputs_final/heldout_skill_optimize/` |
| `--round_log` | `None` (→ `<output_dir>/round_log.json`) |
| `--n_examples_per_bucket` | `10` |
| `--feedback_prior_round` | `False` |
| `--heldout_prompts` | `src/config/heldout_prompts.yaml` |
| `--manifest` | `dataset/heldout_selected_images.json` |
| `--images_root` | `dataset/images_heldout` |
| `--prompts` | `src/config/prompts.yaml` |
| `--enable_thinking` | `True` |
| `--chat_template_path` | `src/config/tool_chat_template_gemma4.jinja` |
| `--gen_max_tokens` | `4096` |
| `--chunk_size` | `64` |
| `--edit_enable_thinking` | `False` |
| `--edit_max_tokens` | `4096` |
| `--seed` | `42` |
| `--gpu_memory_utilization` | `0.9` |
| `--max_model_len` | `24576` |
| `--max_num_seqs` | `64` |
| `--judge_model` | `None` (→ `--model`) |
| `--judge_confidence_threshold` | `1` |
| `--judge_chunk_size` | `200` |
| `--judge_max_tokens` | `1024` |

### `heldout-ablation-summary`

| Arg | Default |
|---|---|
| `--baseline_ckpt` | required |
| `--skill_ckpt` | required |
| `--out_dir` | `None` (→ `<skill_ckpt's own dir>/ablation/`) |
| `--title` | `""` |

### `vision-baseline` (flags relevant to this stage)

| Arg | Default |
|---|---|
| `--model` | `None` (from config) |
| `--config` | `src/config/datasets.yaml` |
| `--output_dir` | `outputs_final/vision_baseline/` |
| `--n_samples` | `10` |
| `--enable_thinking` | `False` |
| `--gen_max_tokens` | `256` |
| `--skill_memory_path` | `None` |
| `--quality_skill_memory_path` | `None` |
| `--species_skill_memory_path` | `None` |
| `--chat_template_path` | `None` |
| `--judge_model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--judge_confidence_threshold` | `1` |
| `--judge_chunk_size` | `200` |
| `--judge_max_tokens` | `None` |
| `--num_shards` | `1` |
| `--shard_index` | `0` |

### `download-heldout-digitalgreen`

| Arg | Default |
|---|---|
| `--images_root` | `dataset/images_heldout` |
| `--output` | `dataset/heldout_selected_images.json` |
| `--eval_frac` | `0.2` |
| `--seed` | `42` |

### `download-heldout-quality`

| Arg | Default |
|---|---|
| `--images_root` | `dataset/images_heldout_quality` |
| `--output` | `dataset/heldout_quality_selected_images.json` |
| `--eval_frac` | `0.2` |
| `--max_per_dataset` | `400` (→ ~2,000 total across the 5 datasets, stratified by class; `0` = no cap, every row) |
| `--seed` | `42` |

### `download-heldout-species`

| Arg | Default |
|---|---|
| `--images_root` | `dataset/images_heldout_species` |
| `--output` | `dataset/heldout_species_selected_images.json` |
| `--selection_log` | `dataset/heldout_species_selection.json` |
| `--n_crop_species` | `20` |
| `--n_weed_species` | `20` |
| `--images_per_species` | `50` |
| `--eval_frac` | `0.2` |
| `--seed` | `42` |
