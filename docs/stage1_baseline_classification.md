# Paper 2, Stage 1 — Baseline Classification Runs

[← Back to README](../README.md)

The plain, thinking-enabled single-shot classification checkpoint — no rubric, no verifier, no tournament, just the model answering the classification question directly. Every later Paper 2 stage builds on this: Stage 0 tests the judge against it, Stage 3's `verifier-cot --from_ckpt` can grade its trajectories directly, and it's the reference point for Stage 3's verifier score and Stage 4's ReAct agent.

## Generate the checkpoint (thinking enabled)

```bash
uv run python main.py vision-baseline \
    --model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --skip_judge \
    --enable_thinking --gen_max_tokens 4096 \
    --chat_template_path src/config/tool_chat_template_gemma4.jinja \
    --output_dir outputs_final/vision_baseline_thinking/
```

Gemma-4's own bundled chat template has no `enable_thinking` support — `--chat_template_path` is required alongside `--enable_thinking` for it to actually produce a `<|channel>thought...<channel|>` trace (Qwen models don't need this). `--gen_max_tokens 4096` matters: at the default, a real thinking trace routinely truncates before reaching `<answer>`.

The checkpoint's `raw_response` column holds the fully untouched generation; `reasoning` holds the extracted thinking text; `prediction` — the ONLY column the judge ever reads — is required to include a brief evidence statement before the answer (`assessment_classification_tagged`, `src/config/prompts.yaml`), not just a bare class name.

**Splitting across multiple GPUs**: `--num_shards N --shard_index i` partitions the work into N independent single-GPU processes; `--merge_shards` combines them afterward. Don't leave shard files sitting alongside the merged canonical file when pointing Stage 0's tooling at that directory — its `--ckpt` glob matches both and double-judges every row.

## Without thinking (optional, for a thinking-vs-not comparison)

Drop `--enable_thinking`/`--chat_template_path`/`--gen_max_tokens` — same command otherwise, a different `--output_dir` (e.g. `outputs_final/vision_baseline/`) so the two checkpoints don't collide.

## What's next

Stage 0 (judge tests), Stage 2 (same idea for detection), Stage 3 (`verifier-cot --from_ckpt` grades these trajectories directly), Stage 4 (ReAct agent answers the same question for comparison).

## Parameters

| Flag | Default | What it does |
|---|---|---|
| `--model` | *(config: `google/gemma-4-26B-A4B-it`)* | Model to run |
| `--enable_thinking` | off | Turn on extended thinking |
| `--gen_max_tokens` | `256` | Generation budget (raise with thinking on) |
| `--chat_template_path` | none | Custom chat template (required for Gemma-4 + thinking) |
| `--output_dir` | `outputs_final/vision_baseline/` | Checkpoint output dir |
| `--skip_judge` | off | Skip the judge phase (F1/ROUGE only) |
| `--num_shards` / `--shard_index` | `1` / `0` | Data-parallel sharding across GPUs |
| `--n_samples` | `10` | Images/class |
| `--rubric` | off | Wrap the question through the rubric fill-in-the-blank template |
