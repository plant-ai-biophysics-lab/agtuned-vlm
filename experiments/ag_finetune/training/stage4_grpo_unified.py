"""
Stage 4 (new) — unified GRPO fine-tuning across all 4 ag tasks (disease_pest,
quality, species classification + object detection), as a comparison
condition against the in-context skill-memory pipeline. See the approved
plan at ~/.claude/plans/linear-giggling-honey.md for the full design and
why GRPO (not SFT/DPO) was chosen, and why one unified model covers
detection too.

Reward is a per-row dispatch by the "task" column:
  - disease_pest / quality / species: the EXACT judge mechanism the rest of
    this pipeline already uses to pick skill-memory's own "best round"
    (apply_judge, src/common/metrics.py) — judged_correct (0/1) per rollout.
  - detection: IoU-based F1 against ground-truth boxes
    (src/common/bbox_metrics.py's match_boxes/precision_recall_f1), no
    judge/model call needed at all — pure box-matching math.

Confirmed against the REAL installed trl==0.24.0 source (not assumed):
  - GRPOTrainer has native multimodal support — dataset needs a "prompt"
    column (conversational message list) + an "image" column (PIL images,
    via datasets.Image() for lazy decoding) for the policy's own
    generation; see _generate_single_turn in trl/trainer/grpo_trainer.py.
  - reward_funcs receive dataset columns by their EXACT, unpluralized
    names via **kwargs (reward_kwargs = {key: [example[key] for example in
    inputs] for key in keys}) — NOT pluralized as
    experiments/ag_finetune/training/stage3_grpo_verifier.py's existing
    reward_fn(completions, ground_truths, **kwargs) assumes (that script's
    own dataset column is "ground_truth", singular — its reward_fn as
    written would raise TypeError if actually run; this script does not
    repeat that mistake).

Data: run experiments/ag_finetune/data_prep/build_grpo_unified_data.py
first to produce the input JSONL this script reads.

GPU layout (role-based, NOT model-sharding — open-source unsloth is
single-GPU-only for actually splitting one model's training across
cards; see the conversation that led to this design). Both the judge
and the policy's own rollout generation are reached over HTTP from
standalone vLLM servers, NOT loaded in-process — src.common.vlm_load's
load_llm() has no device-targeting parameter at all, so there is no
reliable way to keep two heavy in-process vLLM/unsloth models apart on
different GPUs within one Python process; confirmed by reading that
module directly, not assumed:

  GPU 0 — this script itself (the policy, 4-bit QLoRA by default).
  GPU 1 — a standalone vLLM server for the JUDGE (classification reward
          only — detection's reward is pure box-matching math, no judge
          call at all). Launch FIRST:

    CUDA_VISIBLE_DEVICES=1 vllm serve RedHatAI/gemma-4-31B-it-FP8-dynamic \
        --port 8001

  GPU 2 (optional but recommended) — a standalone vLLM server for the
          policy's OWN rollout generation (--use_vllm's "server" mode,
          default on; uses TRL's own purpose-built CLI, which also
          handles weight syncing from the training process — a plain
          `vllm serve` would NOT do that). Launch FIRST:

    CUDA_VISIBLE_DEVICES=2 trl vllm-serve \
        --model unsloth/gemma-4-31B-it \
        --port 8000

  Without GPU 2, pass --vllm_mode colocate (shares GPU 0 with training —
  simpler, but competes for that card's memory) or --no-use_vllm (falls
  back to TRL's default, much slower, naive model.generate()). GPU 1
  (the judge) has no such fallback — it's always a separate server; see
  --judge_server_url.

Usage (launch BOTH server commands above first, then on GPU 0):
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/ag_finetune/training/stage4_grpo_unified.py \
    --model unsloth/gemma-4-31B-it \
    --data outputs/grpo_unified_lessons.jsonl \
    --judge_model RedHatAI/gemma-4-31B-it-FP8-dynamic \
    --judge_server_url http://localhost:8001/v1 \
    --output_dir outputs/checkpoints/stage4_grpo_unified
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

_CLASSIFICATION_TASKS = {"disease_pest", "quality", "species"}


def build_prompt(question: str, system_prompt: str) -> list[dict]:
    """Conversational message list — GRPOTrainer auto-rewrites the plain
    text "content" into the [{"type":"image"},{"type":"text",...}] shape
    once it sees this row also has an "image" column (prepare_multimodal_messages,
    confirmed in the real trl source). Same system+user shape every other
    classification stage in this repo uses (assessment_system from
    prompts.yaml) — detection rows get their own system prompt via the
    SAME param, just a different string, kept as one function rather than
    two near-identical ones."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    return msgs


def judge_via_http(rows: list[dict], client, judge_model: str, system_prompt: str,
                    judge_template: str, gt_key: str, pred_key: str,
                    extra_keys: dict[str, str] | None = None, confidence_threshold: int = 1,
                    max_workers: int = 8) -> list[dict]:
    """Same prompt-building/response-parsing contract as
    src.common.metrics.apply_judge (reuses its own _parse_judge_response —
    pure text parsing, no model dependency, safe to import directly), but
    calls a STANDALONE vLLM OpenAI-compatible server over HTTP instead of
    an in-process vLLM `llm.generate()`. Needed because apply_judge assumes
    an in-process, already-loaded `llm`/`tokenizer` pair; a standalone
    server lets the judge sit on its own GPU with no in-process device-
    targeting problem (see this module's own docstring for why that
    matters here). Not a general-purpose replacement for apply_judge —
    every other caller in this pipeline keeps using the in-process version,
    which is fine for their own offline-batch-then-destroy usage pattern.

    The server applies the judge model's own chat template server-side
    (standard OpenAI-compatible /v1/chat/completions behavior) — so unlike
    apply_judge's own _fmt(), this does NOT need to call
    tokenizer.apply_chat_template itself at all; `messages` goes straight
    to the client.

    max_workers: rows are judged concurrently via a thread pool (each call
    is a blocking HTTP request) — one GRPO batch's classification subset is
    typically small (a handful to a few dozen rows), so a modest thread
    pool is enough; not meant to scale to a large offline judging pass the
    way run_judge_phase's own chunking does."""
    from concurrent.futures import ThreadPoolExecutor
    from src.common.metrics import _parse_judge_response

    def _extra_fmt_kwargs(r: dict) -> dict:
        if not extra_keys:
            return {}
        return {placeholder: str(r.get(row_field, "")) for placeholder, row_field in extra_keys.items()}

    raw_prompts = [
        judge_template.format(
            gt=str(r.get(gt_key, ""))[:400], pred=str(r.get(pred_key, ""))[:400],
            **_extra_fmt_kwargs(r),
        )
        for r in rows
    ]

    def _call_one(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        raw_responses = list(pool.map(_call_one, raw_prompts))

    updated = []
    for row, raw in zip(rows, raw_responses):
        judged_correct = _parse_judge_response(raw, threshold=confidence_threshold)
        updated.append({**row, "judged_correct": judged_correct, "judge_raw_response": raw})
    return updated


def make_reward_fn(judge_client, judge_model: str, judge_system_prompt: str, judge_template: str,
                    iou_threshold: float, policy_model_id: str):
    """Returns the actual reward_fn TRL calls — a closure over an OpenAI
    client pointed at a STANDALONE vLLM server for the judge (see this
    module's own docstring for why in-process apply_judge isn't used here:
    no way to pin an in-process vLLM/unsloth pair to different GPUs)."""
    from src.common.bbox_metrics import match_boxes, precision_recall_f1
    from src.common.bbox_parse import (
        clamp_box, denormalize_box_1000, extract_boxes, extract_boxes_json,
        needs_1000_scale_boxes, uses_json_box_format,
    )

    # Same model-keyed decision every other detection script in this repo
    # makes (detection_multiclass.py/heldout_detection_zeroshot.py) — for
    # Gemma-4, both are True: it natively emits box_2d JSON on a 0-1000
    # normalized scale regardless of what the prompt asks for (measured
    # directly, see bbox_parse.py's own docstrings), so parsing/rescaling
    # must match, not just "parse whatever format the prompt requested."
    use_json = uses_json_box_format(policy_model_id)
    rescale_1000 = needs_1000_scale_boxes(policy_model_id)
    print(f"  [reward] detection box format: use_json={use_json}, rescale_1000={rescale_1000} "
          f"(decided from --model={policy_model_id!r})")

    def reward_fn(completions: list[str], task: list[str], ground_truth: list[str],
                  class_list_text: list[str], image: list, **kwargs) -> list[float]:
        n = len(completions)
        rewards: list[float | None] = [None] * n

        # --- classification subset: one batched apply_judge call -------
        cls_idx = [i for i in range(n) if task[i] in _CLASSIFICATION_TASKS]
        if cls_idx:
            judge_rows = [
                {"prediction": completions[i], "ground_truth": ground_truth[i],
                 "class_list_text": class_list_text[i]}
                for i in cls_idx
            ]
            judged = judge_via_http(
                rows=judge_rows, client=judge_client, judge_model=judge_model,
                system_prompt=judge_system_prompt, judge_template=judge_template,
                gt_key="ground_truth", pred_key="prediction",
                extra_keys={"class_list": "class_list_text"},
            )
            for i, row in zip(cls_idx, judged):
                rewards[i] = 1.0 if row.get("judged_correct") in (1, "1", True) else 0.0

        # --- detection subset: pure box-matching math, no model call ---
        det_idx = [i for i in range(n) if task[i] == "detection"]
        for i in det_idx:
            try:
                gt_boxes = json.loads(ground_truth[i])
                raw_boxes = extract_boxes_json(completions[i]) if use_json else extract_boxes(completions[i])
                img = image[i]  # PIL Image via the dataset's own Image() feature cast
                img_w, img_h = img.width, img.height
                pred_boxes = []
                for b in raw_boxes:
                    bbox = denormalize_box_1000(b["bbox"], img_w, img_h) if rescale_1000 else b["bbox"]
                    pred_boxes.append({"bbox": clamp_box(bbox, img_w, img_h), "category_name": b.get("category_name", "")})
                matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
                rewards[i] = precision_recall_f1(matches, unmatched_pred, unmatched_gt)["f1"]
            except Exception as e:
                # Unparseable/malformed box output — reward 0, same
                # convention every other detection checkpoint in this
                # repo already uses for a failed prediction. Printed (not
                # silent) since a high rate of these would mean the policy
                # isn't learning the expected output format at all, worth
                # noticing early rather than reading it off a flat-zero
                # reward curve with no diagnosis.
                print(f"  [reward] detection row {i} parse/score error: {e}")
                rewards[i] = 0.0

        assert all(r is not None for r in rewards), "reward dispatch missed a row — unknown task tag?"
        return rewards

    return reward_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/gemma-4-31B-it",
                     help="Base VLM checkpoint FastVisionModel can load for training — NOT the FP8 "
                          "vLLM-serving export (inference-only), same caveat docs/stage1_linear_probe.md "
                          "already documents for the linear-probe stage.")
    ap.add_argument("--data", default="outputs/grpo_unified_lessons.jsonl",
                     help="Output of build_grpo_unified_data.py.")
    ap.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic",
                     help="Loaded via vLLM, resident for the whole run on its own GPU (see module "
                          "docstring) — same default every other judge call in this pipeline uses.")
    ap.add_argument("--prompts", default="src/config/prompts.yaml")
    ap.add_argument("--output_dir", default="outputs/checkpoints/stage4_grpo_unified")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--n_completions", type=int, default=8,
                     help="K rollouts per prompt (GRPO group size) — 8 matches the K used elsewhere "
                          "in this pipeline (e.g. verifier-cot's own K-candidate default).")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_coeff", type=float, default=0.01)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--iou_threshold", type=float, default=0.5,
                     help="Matches detection-final-metrics' own default.")
    ap.add_argument("--judge_server_url", default="http://localhost:8001/v1",
                     help="Base URL of a STANDALONE vLLM OpenAI-compatible server hosting "
                          "--judge_model (launch separately, on its own GPU — see this script's "
                          "module docstring for the exact `vllm serve` command). The judge is "
                          "reached over HTTP, never loaded in-process, so it can live on a "
                          "different GPU than the policy with no in-process device conflict.")
    ap.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                     help="QLoRA: load the policy base in 4-bit (bitsandbytes NF4 — a DIFFERENT "
                          "quantization scheme from the RedHatAI FP8-dynamic vLLM-serving export; "
                          "not interchangeable with it, see the conversation that led to this flag) "
                          "instead of full bf16. Default True — bf16 needs ~62GB for weights alone "
                          "on a 31B model, out of reach on e.g. a 48GB RTX 6000 Ada; 4-bit brings "
                          "weights down to ~16GB, leaving real headroom for LoRA/optimizer/"
                          "activations/generation on a single such card. Pass --no-load_in_4bit "
                          "only if you have a GPU large enough for full-precision training (~80GB+) "
                          "and want the (likely marginal, unvalidated) quality difference.")
    ap.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True,
                     help="Route GRPO's own rollout generation (K completions/prompt) through vLLM "
                          "instead of TRL's default naive model.generate() — same completions, far "
                          "more memory-efficient KV cache (paged attention + continuous batching), "
                          "and faster. Default True; pass --no-use_vllm only for a quick local smoke "
                          "test where you don't want to stand up a separate server first.")
    ap.add_argument("--vllm_mode", default="server", choices=["server", "colocate"],
                     help="'server' (default): send generation requests to a separately-launched "
                          "vLLM server (start it with `trl vllm-serve`, see this script's module "
                          "docstring) — can sit on its own GPU, zero memory impact on the training "
                          "GPU. 'colocate': vLLM shares the training GPU in the same process — "
                          "simpler (no second process to launch) but competes with training for the "
                          "same card's memory. Only relevant when --use_vllm is set.")
    ap.add_argument("--vllm_server_host", default="0.0.0.0")
    ap.add_argument("--vllm_server_port", type=int, default=8000)
    args = ap.parse_args()

    from datasets import Dataset, Image as HFImage
    from unsloth import FastVisionModel
    from trl import GRPOTrainer, GRPOConfig

    from openai import OpenAI

    from src.common.io import load_prompts
    from src.common.metrics import DEFAULT_JUDGE_TEMPLATE_KEY

    prompts = load_prompts(Path(args.prompts))

    # --- load the combined dataset -------------------------------------
    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows):,} GRPO training row(s) across tasks: "
          f"{sorted({r['task'] for r in rows})}")

    system_prompt = prompts.get("assessment_system", "")
    cols: dict[str, list] = {
        "prompt": [build_prompt(r["question"], system_prompt) for r in rows],
        "image": [r["image_path"] for r in rows],  # cast to HFImage below for lazy decoding
        "task": [r["task"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
        "class_list_text": [r["class_list_text"] for r in rows],
        "agml_dataset": [r["agml_dataset"] for r in rows],
    }
    dataset = Dataset.from_dict(cols).cast_column("image", HFImage())

    # --- policy model (LoRA, vision layers only — same config as the
    # existing SFT scaffold, experiments/lora_setup/lora_gemma4.py, except
    # that script defaults load_in_4bit=False/True per-model-size; this one
    # defaults True unconditionally since the 31B model is the only size
    # this pipeline uses here, see --load_in_4bit's own help text) --------
    print(f"  Loading policy {args.model!r} with load_in_4bit={args.load_in_4bit}")
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=False,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none",
        random_state=3407, use_rslora=False, loftq_config=None,
    )
    FastVisionModel.for_training(model)

    # --- judge (standalone vLLM server, reached over HTTP — see module
    # docstring for why this isn't loaded in-process) --------------------
    judge_client = OpenAI(base_url=args.judge_server_url, api_key="not-needed")
    try:
        judge_client.models.list()
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not reach judge server at {args.judge_server_url!r} ({e}). "
            f"Start it first — see this script's module docstring for the exact "
            f"`vllm serve {args.judge_model} --port ...` command."
        )
    reward_fn = make_reward_fn(
        judge_client, args.judge_model,
        judge_system_prompt=prompts.get("judge_system_vision", ""),
        judge_template=prompts[DEFAULT_JUDGE_TEMPLATE_KEY],
        iou_threshold=args.iou_threshold,
        policy_model_id=args.model,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[reward_fn],
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            num_generations=args.n_completions,
            learning_rate=args.lr,
            kl_coeff=args.kl_coeff,
            bf16=True,
            logging_steps=20,
            save_steps=200,
            # 512 was the stage3_grpo_verifier.py scaffold's own default —
            # too low here: detection completions are a JSON array with one
            # object per instance, and some held-out datasets have ~50 GT
            # instances/image (coffee_detection). A truncated completion
            # can't parse as valid JSON at all, silently forcing reward=0
            # regardless of localization quality — the exact same class of
            # bug diagnosed earlier this session for Qwen's classification
            # generations (--gen_max_tokens too low, near-zero judge_accuracy
            # until raised to 8192). Watch completion-length histograms
            # early in a real run and raise further if truncation shows up.
            max_completion_length=2048,
            remove_unused_columns=False,  # GRPOTrainer reads non-"prompt" columns itself; see
                                            # _set_signature_columns_if_needed in the real trl source
            use_vllm=args.use_vllm,
            vllm_mode=args.vllm_mode,
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
        ),
    )
    trainer.train()

    model.save_pretrained(f"{args.output_dir}/lora_adapters")
    processor.save_pretrained(f"{args.output_dir}/lora_adapters")
    model.save_pretrained_merged(f"{args.output_dir}/vllm_model", processor, save_method="merged_16bit")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
