# Stage 1 — Linear Probe (H1)

[← Back to README](../README.md)

Tests whether a backbone's frozen vision features are linearly separable across classes — trains a single linear layer (no hidden units) on frozen embeddings, per the paper's "Linear probing" section.

Produces `outputs_final/linear_probe/linear_probe_results.json` plus summary charts (`summary_by_task.png`, `summary_overall.png`, t-SNE, confusion matrices).

## Step 0 (optional) — pre-check backbones

```bash
uv run python main.py check-vision-towers
```

Loads each model in `probe_config.yaml`'s `models` list, confirms its vision tower is resolvable, and reports which layer the heuristic fallback would hook. Cheap way to catch an architecture-incompatible model before running the full pipeline.

## Step 1 — run the probe

```bash
uv run python main.py probe
```

Defaults come from `src/linear_probe/probe_config.yaml` — edit that file directly (models, hyperparameters, output dir) instead of passing flags. Any flag passed on the command line overrides the config.

Embeddings are cached to `.npz` automatically (`cache_embeddings: true`) — reruns after a hyperparameter tweak skip re-extraction entirely, no extra flag needed.

To recover cheaply after a run's plotting phase failed (e.g. OOM) without re-extracting or retraining:

```bash
uv run python main.py probe --plots_only
```

## Step 2 (optional) — collate across seeds with 95% CI

`--seed` (`probe_config.yaml`, default 42) controls which `n_samples` images of each class land in training. To check results aren't an artifact of one particular draw, rerun the probe with different seeds into separate output dirs, sharing one `cache_dir` so extraction only happens once:

```bash
uv run python main.py probe --seed 1 --output_dir outputs_final/linear_probe_seed1/ --cache_dir outputs_final/linear_probe/embedding_cache/
uv run python main.py probe --seed 2 --output_dir outputs_final/linear_probe_seed2/ --cache_dir outputs_final/linear_probe/embedding_cache/
uv run python main.py probe --seed 3 --output_dir outputs_final/linear_probe_seed3/ --cache_dir outputs_final/linear_probe/embedding_cache/
```

Then combine them into one chart with 95% CI error bars (Student's t across seeds):

```bash
uv run python main.py probe-collate-seeds \
  --seed_dirs outputs_final/linear_probe_seed1/ outputs_final/linear_probe_seed2/ outputs_final/linear_probe_seed3/ \
  --output_dir outputs_final/linear_probe_seeds_collated/
```

Produces `summary_by_task_seeds.png` (same layout as `summary_by_task.png`) plus `seed_collate_summary.csv`/`.json`.

## Picking which layer to hook

Most backbones use a heuristic fallback (the vision tower's own final output). For architectures where that's wrong (e.g. Qwen-style towers with a merger), the pipeline auto-detects and persists the correct layer to `src/linear_probe/layer_selections.yaml`. This file is shared — commit it so others get the same layer choices.

To pick a layer manually, add an entry:

```yaml
some/model-id:
  pre: model.vision_tower.blocks.26
```
