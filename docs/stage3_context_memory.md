# Paper 2, Stage 3 — Context/Skill Memory Construction

[← Back to README](../README.md)

## Step 0 — download the held-out dataset

```bash
uv run python main.py download-heldout-digitalgreen
```

Args to note: `--eval_frac` (lessons/eval split), `--seed`.

## Step 1 — zero-shot prediction

```bash
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic
```

Args to note: `--split` (`lessons` default, `eval` for Step 4), `--skill_memory_path`, `--ckpt`-free resume via `--output_dir`, `--rerun`.

## Step 2 — reveal ground truth, every row

```bash
uv run python main.py heldout-explain-traces \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_zeroshot/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Args to note: `--ckpt` (required), `--gen_max_tokens` (bump if thinking traces truncate).

## Step 3 — summarize into the skill `.md`

```bash
uv run python main.py heldout-summarize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --ckpt outputs_final/heldout_explain_traces/RedHatAI_gemma-4-31B-it-FP8-dynamic/ckpt_RedHatAI_gemma-4-31B-it-FP8-dynamic.csv
```

Args to note: `--max_words_per_chunk`, `--output_md`, `--rerun`.

## Step 3b — self-optimization loop

```bash
uv run python main.py heldout-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --init_skill_path src/config/skill_memory_disease-pest.md \
    --max_rounds 10 \
    --output_dir outputs_final/heldout_skill_optimize/
```

With prior-round feedback:

```bash
uv run python main.py heldout-skill-optimize \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --init_skill_path src/config/skill_memory_disease-pest.md \
    --max_rounds 10 \
    --feedback_prior_round \
    --output_dir outputs_final/heldout_skill_optimize_feedback/
```

Args to note: `--init_skill_path` (required), `--max_rounds` (required), `--n_examples_per_bucket`, `--feedback_prior_round`, `--round_log`, `--output_dir`. Winning round is written to `<output_dir>/skill_memory_best.md`.

## Step 4 — ablation

```bash
# Baseline, no skill file:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --output_dir outputs_final/heldout_zeroshot_eval_baseline/

# With skill file:
uv run python main.py heldout-zeroshot \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --split eval \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/heldout_zeroshot_eval_not_optimized_skill/
```

Args to note: `--split eval`, distinct `--output_dir` per run.

## Step 5 — inject into vision-baseline

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/
```

Sharded across GPUs:

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skill_memory_path src/config/skill_memory_disease-pest.md \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --num_shards 4 --shard_index $SLURM_ARRAY_TASK_ID
```

Merge shards:

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --output_dir outputs_final/vision_baseline_thinking_with_skill/ \
    --merge_shards
```

Args to note: `--skill_memory_path` (gated per-row to `task_type in {disease, pest_damage}`), `--output_dir` (must not collide with `outputs_final/vision_baseline_thinking/`), `--num_shards`/`--shard_index`, `--merge_shards`.

## Step 6 (not built)

Repeat Steps 0-5 for object detection, after classification gains hold up. See [Stage 6](stage6_detection_verifier.md).

## Default args

### `heldout-zeroshot`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
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
| `--judge_model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
| `--judge_confidence_threshold` | `1` |
| `--judge_chunk_size` | `200` |
| `--judge_max_tokens` | `1024` |
| `--judge_chat_template_path` | `None` |

### `heldout-explain-traces`

| Arg | Default |
|---|---|
| `--model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
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
| `--judge_model` | `RedHatAI/gemma-4-31B-it-FP8-dynamic` |
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
