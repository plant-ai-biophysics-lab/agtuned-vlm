# agtuned-vlm

Evaluation framework for vision-language models on agricultural image understanding.

## Prerequisites

```bash
uv sync
```

## Experiments Pipeline

| Stage | Doc | What it does |
|---|---|---|
| 0 | [docs/stage0_judge_improvements.md](docs/stage0_judge_improvements.md) | Judge improvements |
| 1 | [docs/stage1_baseline_classification.md](docs/stage1_baseline_classification.md), [docs/stage1_linear_probe.md](docs/stage1_linear_probe.md), [docs/stage4_verifier.md](docs/stage4_verifier.md) | Baseline methods: linear probe, plain zero-shot, rubric-guided CoT, PPT-tournament verifier |
| 2 | [docs/stage2_object_detection.md](docs/stage2_object_detection.md) | Baseline object detection |
| 3 | [docs/stage3_context_memory.md](docs/stage3_context_memory.md) | Classification skill-memory construction + injection |
| 4 | [docs/stage4_detection_context_memory.md](docs/stage4_detection_context_memory.md) | Detection skill-memory construction + injection |
| 5 | [docs/stage5_classification_verifier.md](docs/stage5_classification_verifier.md) | Lightweight per-trajectory classification verifier |
| 6 | [docs/stage6_detection_verifier.md](docs/stage6_detection_verifier.md) | Detection verifier (not started) |
| 7 | [docs/stage7_react_baseline.md](docs/stage7_react_baseline.md) | Tool-using ReAct agent baseline |

## Viewers

```bash
streamlit run app/classification_viewer.py -- --output_dir outputs/data_generation/
streamlit run app/react_baseline_viewer.py -- --output_dir outputs_final/react_baseline/
```

See [app/README.md](app/README.md) and [docs/stage7_react_baseline.md](docs/stage7_react_baseline.md).
