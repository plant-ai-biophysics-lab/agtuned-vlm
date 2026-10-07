"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 3b — automated
self-optimization loop: starting from one skill-memory.md, repeatedly
(1) score it on the TRAIN (lessons) split, (2) EDIT it directly against
that round's own TRAIN per-task scores + a sample of its own TRAIN
mistakes (NOT a re-summarize from Step 2's original traces), (3) ALSO
score the revised file on the VAL (eval) split — used ONLY to pick
which round's file to keep, never to steer an edit, (4) repeat for
--max_rounds — then report the single best-VAL-scoring round's file.

**This is the train/val split a self-optimization loop needs to not be
pure train-on-test** — an earlier version of this loop scored AND
edited against the same fixed eval split every round, which is
optimizing directly against your only measurement of generalization.
Now: TRAIN scores/mistakes drive what gets edited; VAL score ONLY
decides which round's resulting file is kept. VAL is never shown to the
edit call, directly or indirectly.

One invocation runs the WHOLE loop (no more manually alternating
`heldout-zeroshot`/`heldout-skill-optimize` calls round by round):

  round 0: score --init_skill_path as-is on BOTH splits (no edit yet)
    — the starting point every later round is judged against.
  round i (1..max_rounds): edit round (i-1)'s skill file against round
    (i-1)'s own TRAIN scores/mistakes -> round i's skill file; score
    round i on BOTH splits.

Each round writes its own checkpoints + skill.md under
--output_dir/round<i>/{train,val}/, and appends ONE entry to
--round_log (a single JSON file, list of round records) with:
  - TRAIN and VAL overall + per-task-bucket judge_accuracy, kept
    separate (never averaged/combined) so it's always visible whether
    a round's improvement shows up on both or just one
  - a MECHANICAL diff (added/removed lines, via difflib — not trusted
    to the editing model's own self-report) between round (i-1)'s and
    round i's skill text
  - the editing model's own stated CHANGELOG (what it changed and why,
    parsed from the ---CHANGELOG--- section heldout_skill_edit_prompt
    asks for) — the diff is the objective "what", the changelog is the
    model's own "why"
Resumable: a round already present in --round_log (and whose output
files still exist) is skipped on a re-run, so a --max_rounds run split
across multiple SLURM jobs just continues from the last completed
round.

After the last round, the round with the HIGHEST VAL overall
judge_accuracy across ALL rounds (round 0 included — a loop that only
makes things worse should report round 0, not a forced "optimized"
file) is copied to --output_dir/skill_memory_best.md. TRAIN score is
NEVER used for this selection — a round could overfit TRAIN and still
correctly lose to an earlier round on VAL.

**Real cost tradeoff, worth knowing going in**: TRAIN (lessons, n=827)
is ~4x VAL (eval, n=199) — scoring BOTH every round roughly doubles
this loop's GPU time per round versus the old eval-only design (and
TRAIN's own scoring pass dominates that increase). This is the
necessary cost of the fix, not a bug.

Usage:
    uv run python main.py heldout-skill-optimize \\
        --model RedHatAI/gemma-4-31B-it-FP8-dynamic \\
        --init_skill_path src/config/skill_memory_disease-pest.md \\
        --max_rounds 10 \\
        --output_dir outputs_final/heldout_skill_optimize/
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
csv.field_size_limit(2**31 - 1)  # same latent-bug fix every other checkpoint-reading script in this
# repo needed — explanations/reasoning fields can exceed the 131072-byte default on a full run.
import difflib
import json
import random
import re
import shutil
from collections import defaultdict

from src.common.chat_prompt import render_chat_template
from src.common.metrics import extract_channel_thought
from src.common.vision_eval import safe_model_name
from src.common.vlm_load import DEFAULT_VISION_TOKEN_BUDGET, destroy_llm, load_llm
from src.heldout_zeroshot import bucket_category, build_arg_parser as build_zeroshot_arg_parser, run_model as run_zeroshot

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None  # type: ignore[assignment,misc]

_BUCKETS = ["disease", "pest", "deficiency", "healthy"]
_CHANGELOG_DELIM = "---CHANGELOG---"


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def compute_task_scores(rows: list[dict]) -> dict[str, dict]:
    """{bucket: {"accuracy", "n"}} — same bucket_category heuristic Step
    1/3 use, applied to each row's own class_label (ground truth)."""
    per_bucket: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        jc = r.get("judged_correct")
        if jc in (None, ""):
            continue
        per_bucket[bucket_category(r["class_label"])].append(int(jc))
    out = {}
    for b in _BUCKETS:
        vals = per_bucket.get(b, [])
        out[b] = {"accuracy": round(sum(vals) / len(vals), 4) if vals else None, "n": len(vals)}
    return out


def compute_overall_accuracy(rows: list[dict]) -> float | None:
    vals = [int(r["judged_correct"]) for r in rows if r.get("judged_correct") not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else None


def build_task_scores_block(scores: dict[str, dict]) -> str:
    lines = []
    for b in _BUCKETS:
        s = scores[b]
        if s["n"] == 0:
            continue
        lines.append(f"- {b}: {s['accuracy']:.1%} accuracy (n={s['n']})")
    return "\n".join(lines)


def build_prior_round_feedback_block(changelog: str | None, prior_task_scores: dict | None, enabled: bool) -> str:
    """Empty string when --feedback_prior_round is off (the original
    behavior, kept default so A/B testing this against the baseline is
    one flag flip) or there's no prior round yet. When on: the prior
    round's own stated changelog (what it changed and why) PAIRED with
    the TRAIN scores that edit actually produced — letting the editor
    see whether its last change helped, before making another one.
    Deliberately NOT the full multi-round history (would both bloat the
    prompt and let the model chase noise across many rounds' worth of
    small-sample TRAIN scores) — just the immediately-prior round."""
    if not enabled or changelog is None or prior_task_scores is None:
        return ""
    scores_block = build_task_scores_block(prior_task_scores)
    return (
        "\nFor reference, here is what you changed in the PREVIOUS round of "
        "editing this document, and the TRAIN scores that edit actually "
        "produced (use this to judge whether your last change helped, hurt, "
        "or had no real effect before making another one):\n\n"
        f"Previous round's changelog:\n{changelog}\n\n"
        f"Resulting TRAIN scores:\n{scores_block}\n"
    )


def _sample_rows_by_judgement(rows: list[dict], n_per_bucket: int, seed: int, want_correct: bool) -> dict[str, list[dict]]:
    """{bucket: [row, ...]} — up to n_per_bucket rows per bucket matching
    want_correct, sampled (not just the first N) so repeated rounds
    against the same TRAIN split don't always see the identical subset.
    Callers pass TRAIN rows only — this function has no opinion on which
    split it's given, but the loop's own design never calls it on VAL
    rows. Shared by sample_miss_examples/sample_worked_examples so the
    two can never silently drift in sampling behavior."""
    rng = random.Random(seed)
    matched: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("judged_correct") in (None, ""):
            continue
        if (int(r["judged_correct"]) == 1) != want_correct:
            continue
        matched[bucket_category(r["class_label"])].append(r)
    out = {}
    for b in _BUCKETS:
        pool = matched.get(b, [])
        out[b] = rng.sample(pool, min(n_per_bucket, len(pool)))
    return out


def sample_miss_examples(rows: list[dict], n_per_bucket: int, seed: int) -> dict[str, list[dict]]:
    """Up to n_per_bucket WRONG rows per bucket — what to fix."""
    return _sample_rows_by_judgement(rows, n_per_bucket, seed, want_correct=False)


def sample_worked_examples(rows: list[dict], n_per_bucket: int, seed: int) -> dict[str, list[dict]]:
    """Up to n_per_bucket RIGHT rows per bucket — what to protect. Same
    seed as the paired sample_miss_examples call (caller's choice) is
    fine — they draw from disjoint pools (judged_correct 0 vs 1), so
    using the same seed doesn't correlate which specific rows get picked
    from each."""
    return _sample_rows_by_judgement(rows, n_per_bucket, seed, want_correct=True)


def _build_examples_block(examples_by_bucket: dict[str, list[dict]], outcome_label: str) -> str:
    blocks = []
    for b in _BUCKETS:
        for r in examples_by_bucket.get(b, []):
            blocks.append(
                f"[{b}] {r.get('crop', '?')} plant — model {outcome_label} "
                f"\"{(r.get('extracted_answer') or r.get('prediction') or '').strip()}\", actual answer: "
                f"\"{r['class_label']}\". Model's own reasoning: \"{(r.get('reasoning') or '').strip()[:500]}\". "
                f"Expert's observations: \"{(r.get('details') or '').strip()[:500]}\"."
            )
    return "\n\n".join(blocks)


def build_miss_examples_block(examples_by_bucket: dict[str, list[dict]]) -> str:
    return _build_examples_block(examples_by_bucket, "answered")


def build_worked_examples_block(examples_by_bucket: dict[str, list[dict]]) -> str:
    return _build_examples_block(examples_by_bucket, "correctly answered")


def diff_skill_text(old_text: str, new_text: str) -> dict:
    """MECHANICAL diff (difflib, not the editing model's own self-report)
    between two skill.md versions — the objective "what changed" that
    sits alongside the model's own stated "why" (the parsed changelog).
    Line-based: this is a short bulleted Markdown doc, not prose, so
    line-level add/remove is already a meaningful unit of change."""
    old_lines = [l for l in old_text.splitlines() if l.strip()]
    new_lines = [l for l in new_text.splitlines() if l.strip()]
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    removed, added = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(old_lines[i1:i2])
        if tag in ("insert", "replace"):
            added.extend(new_lines[j1:j2])
    return {"removed_lines": removed, "added_lines": added, "n_removed": len(removed), "n_added": len(added)}


def parse_edit_output(raw: str) -> tuple[str, str, bool]:
    """(revised_md, changelog_text, changelog_parsed) — splits on
    _CHANGELOG_DELIM. changelog_parsed=False means the model didn't
    follow the format and the WHOLE output was treated as the document
    (not a silent failure — logged explicitly so a round with no real
    changelog is distinguishable from a round where parsing broke)."""
    if _CHANGELOG_DELIM in raw:
        doc, _, changelog = raw.partition(_CHANGELOG_DELIM)
        return doc.strip(), changelog.strip(), True
    return raw.strip(), "", False


# The exact bare-category-word failure mode v1/v2 (src/common/heldout_skill.py's
# own docstring history) found — a bullet whose ENTIRE content is just a bare
# category label, nothing else. Checked automatically every round instead of
# requiring a manual read of the full skill.md to catch a regression.
_BARE_CATEGORY_RE = re.compile(
    r"^[-*]\s*\*{0,2}(chewing pests?|sucking pests?|pest|disease|deficiency|viral infection)\*{0,2}:?\s*$",
    re.IGNORECASE,
)
_COMMON_SECTION_RE = re.compile(r"common (diseases|pests|conditions).{0,40}(to know|reference)", re.IGNORECASE)

# Direct detector for a channel-thought leak (round 1/4/5's real bug) —
# does NOT rely on parse_edit_output's own changelog_parsed flag, which
# turned out to be an unreliable signal: a model whose thinking trace
# never closed can still happen to mention the literal "---CHANGELOG---"
# string INSIDE its own unclosed planning prose (e.g. restating its own
# output-format instructions to itself), which parse_edit_output's naive
# string-partition then treats as a real delimiter — changelog_parsed=True
# even though the "document" it split off is still raw garbage thinking
# text. This checks the actual channel markers directly instead.
_CHANNEL_MARKER_RE = re.compile(r"<\|channel>|<channel\|>")


def check_rule_violations(skill_md: str) -> list[str]:
    """Automated QC against the documented historical failure modes (see
    heldout_prompts.yaml's heldout_skill_edit_prompt rules 1-2, plus the
    channel-thought-leak bug found in a real run) — returns a list of
    violation strings, empty if clean. Not a full content-quality check,
    just a cheap regex tripwire for the specific, already-known-bad
    patterns, run every round so a regression shows up in --round_log
    immediately rather than requiring a manual re-read."""
    violations = []
    if _CHANNEL_MARKER_RE.search(skill_md):
        violations.append("CHANNEL-THOUGHT LEAK: a raw <|channel>/<channel|> marker is present in the "
                           "document — this round's skill.md is likely the model's own unclosed "
                           "thinking trace, not a real revised document (see --edit_enable_thinking)")
    for i, line in enumerate(skill_md.splitlines(), start=1):
        if _BARE_CATEGORY_RE.match(line.strip()):
            violations.append(f"line {i}: bare-category-word bullet: {line.strip()!r}")
    if _COMMON_SECTION_RE.search(skill_md):
        violations.append("a 'Common Diseases/Pests/Conditions...' style reference section reappeared")
    return violations


def score_round(model_id: str, skill_memory_path: Path, split_dir: Path, split: str,
                 base_args: argparse.Namespace) -> tuple[list[dict], float | None, dict]:
    """Runs heldout-zeroshot's own --split {lessons,eval} generation+judge
    logic (via heldout_zeroshot.run_model, reused directly — not
    reimplemented) against skill_memory_path, writing its checkpoint
    under split_dir. Returns (rows, overall_accuracy, task_scores). split
    is "lessons" (TRAIN — drives the next edit) or "eval" (VAL — selects
    the best round only, caller must never feed this into an edit)."""
    round_args = argparse.Namespace(**vars(base_args))
    round_args.skill_memory_path = str(skill_memory_path)
    round_args.wrapper_only = False
    round_args.split = split
    round_args.output_dir = str(split_dir)
    round_args.rerun = False
    rows, _ckpt_path = run_zeroshot(model_id, round_args)
    return rows, compute_overall_accuracy(rows), compute_task_scores(rows)


def edit_round(model_id: str, llm, tokenizer, current_skill_md: str, task_scores_block: str,
                miss_examples_block: str, worked_examples_block: str, prior_round_feedback_block: str,
                edit_template: str, chat_template: str | None, args: argparse.Namespace) -> tuple[str, str, bool, str, str]:
    """ONE edit call. Returns (revised_md, changelog_text, changelog_parsed,
    thinking_trace, raw_cleaned_output) — the last two are for debugging a
    round that looks wrong (truncation, a channel-thought-leak, a model
    that ignored the format) without needing to re-run anything; callers
    should persist raw_cleaned_output to disk per round, not just the
    parsed halves, since parse_edit_output's own correctness is exactly
    the kind of thing you'd want to audit from the untouched source.
    worked_examples_block: CORRECT TRAIN rows, shown alongside the misses
    so the edit call has a concrete reason not to disturb wording that's
    already working (heldout_skill_edit_prompt's own rule 6) — without
    this, every round only ever saw evidence to add/change, never
    evidence to leave something alone. prior_round_feedback_block: empty
    unless --feedback_prior_round — see build_prior_round_feedback_block."""
    edit_text = edit_template.format(
        current_skill_md=current_skill_md, task_scores_block=task_scores_block,
        miss_examples_block=miss_examples_block, worked_examples_block=worked_examples_block,
        prior_round_feedback_block=prior_round_feedback_block,
    )
    hist = [{"role": "user", "content": edit_text}]
    prompt = render_chat_template(tokenizer, hist, enable_thinking=args.edit_enable_thinking, chat_template=chat_template)
    gen_params = SamplingParams(
        temperature=0.0, max_tokens=args.edit_max_tokens, seed=args.seed,
        skip_special_tokens=not args.edit_enable_thinking,
    )
    out = llm.generate([prompt], sampling_params=gen_params)[0]
    raw = out.outputs[0].text
    thinking, cleaned = extract_channel_thought(raw) if args.edit_enable_thinking else ("", raw)
    revised_md, changelog, changelog_parsed = parse_edit_output(cleaned)
    return revised_md, changelog, changelog_parsed, thinking, cleaned


def run_loop(model_id: str, args: argparse.Namespace) -> None:
    hprompts = load_yaml(Path(args.heldout_prompts))
    edit_template = hprompts["heldout_skill_edit_prompt"]
    chat_template = Path(args.chat_template_path).read_text(encoding="utf-8") if args.chat_template_path else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    round_log_path = Path(args.round_log) if args.round_log else out_dir / "round_log.json"
    history: list[dict] = json.loads(round_log_path.read_text()) if round_log_path.exists() else []
    completed_rounds = {h["round"] for h in history}

    # Base args for score_round — every heldout-zeroshot default
    # (gen_max_tokens, judge settings, chat template, etc.) comes from
    # its OWN build_arg_parser(), not hand-duplicated here, so the two
    # scripts can never silently drift apart on defaults.
    base_args = build_zeroshot_arg_parser().parse_args([])
    base_args.model = model_id
    for flag in ("manifest", "images_root", "prompts", "chat_template_path", "gen_max_tokens",
                 "chunk_size", "quantize", "gpu_memory_utilization", "max_model_len",
                 "vision_token_budget", "max_num_seqs", "enforce_eager", "kv_cache_dtype",
                 "enable_thinking", "skip_judge", "judge_model", "judge_quantize",
                 "judge_confidence_threshold", "judge_chunk_size", "judge_enable_thinking",
                 "judge_max_tokens", "judge_chat_template_path"):
        if hasattr(args, flag):
            setattr(base_args, flag, getattr(args, flag))

    current_skill_path = Path(args.init_skill_path)
    current_skill_text = current_skill_path.read_text(encoding="utf-8").strip()
    prev_changelog: str | None = None  # see build_prior_round_feedback_block / --feedback_prior_round

    for round_i in range(args.max_rounds + 1):  # round 0 = score the initial file as-is, no edit
        round_dir = out_dir / f"round{round_i}"
        round_skill_path = round_dir / "skill_memory.md"
        train_dir = round_dir / "train"
        val_dir = round_dir / "val"

        if round_i in completed_rounds and round_skill_path.exists():
            print(f"  [round {round_i}] already completed — skipping scoring/edit (resume)")
            current_skill_text = round_skill_path.read_text(encoding="utf-8").strip()
            current_skill_path = round_skill_path
            prior = next(h for h in history if h["round"] == round_i)
            # Reload this round's TRAIN checkpoint rows so the NEXT round's
            # edit step still has what it needs (miss examples, task
            # scores) — without this, resuming mid-loop silently edits
            # with an empty miss-examples block on the first post-resume
            # round. VAL rows aren't needed downstream (only logged), so
            # not reloaded here.
            ckpt_path = train_dir / safe_model_name(model_id) / f"ckpt_{safe_model_name(model_id)}.csv"
            if ckpt_path.exists():
                with open(ckpt_path, newline="", encoding="utf-8") as f:
                    prev_train_rows = list(csv.DictReader(f))
            else:
                prev_train_rows = []
            prev_train_task_scores = prior["train_task_scores"]
            prev_changelog = prior["changelog"]
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  === round {round_i}/{args.max_rounds} ===")
        changelog_parsed = True
        thinking_trace = ""
        if round_i == 0:
            # Round 0's "skill file" IS the init file, copied in place so
            # every round (0 included) has a uniform round<i>/skill_memory.md
            # location to select the winner from later.
            shutil.copy(current_skill_path, round_skill_path)
            changelog = "(round 0 — initial file, not yet edited)"
            diff_summary = {"removed_lines": [], "added_lines": [], "n_removed": 0, "n_added": 0}
            print(f"  Scoring initial file as-is: {round_skill_path}")

        if round_i > 0:
            # Steered ONLY by TRAIN (lessons) — prev_train_rows/
            # prev_train_task_scores come from round (i-1)'s TRAIN pass
            # below, never its VAL pass. VAL never reaches this call.
            prev_scores_block = build_task_scores_block(prev_train_task_scores)
            prev_misses = sample_miss_examples(prev_train_rows, args.n_examples_per_bucket, args.seed + round_i)
            miss_examples_block = build_miss_examples_block(prev_misses)
            n_examples = sum(len(v) for v in prev_misses.values())
            prev_worked = sample_worked_examples(prev_train_rows, args.n_examples_per_bucket, args.seed + round_i)
            worked_examples_block = build_worked_examples_block(prev_worked)
            n_worked = sum(len(v) for v in prev_worked.values())
            print(f"  [edit] {n_examples} TRAIN miss example(s) + {n_worked} TRAIN worked example(s) "
                  f"from round {round_i - 1} shown to the editor")
            prior_round_feedback_block = build_prior_round_feedback_block(
                prev_changelog, prev_train_task_scores, args.feedback_prior_round)
            if args.feedback_prior_round:
                print(f"  [edit] prior round's changelog + resulting TRAIN scores also shown to the editor")

            llm, tokenizer = load_llm(
                model_id, quantization=args.quantize,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_seqs=args.max_num_seqs,
                enforce_eager=args.enforce_eager,
                vision_token_budget=args.vision_token_budget,
                kv_cache_dtype=args.kv_cache_dtype,
                caller_tag="heldout_skill_optimize_edit",
            )
            revised_md, changelog, changelog_parsed, thinking_trace, raw_cleaned = edit_round(
                model_id, llm, tokenizer, current_skill_text, prev_scores_block, miss_examples_block,
                worked_examples_block, prior_round_feedback_block, edit_template, chat_template, args,
            )
            destroy_llm(llm)
            del tokenizer

            # Raw (pre-parse) output saved to disk EVERY round, not just the
            # parsed halves — if a round looks wrong (truncated, the
            # changelog delimiter missing, a bare-category regression),
            # this is the untouched source to check parse_edit_output's own
            # correctness against, without re-running anything.
            (round_dir / "edit_raw_output.txt").write_text(raw_cleaned)
            if thinking_trace:
                (round_dir / "edit_thinking.txt").write_text(thinking_trace)
            if not changelog_parsed:
                print(f"  [edit] WARNING: '{_CHANGELOG_DELIM}' not found in the model's output — "
                      f"whole response treated as the document, no changelog captured this round.")

            diff_summary = diff_skill_text(current_skill_text, revised_md)
            round_skill_path.write_text(revised_md)
            current_skill_text = revised_md
            print(f"  [edit] +{diff_summary['n_added']} / -{diff_summary['n_removed']} line(s) vs. round {round_i - 1}")

        rule_violations = check_rule_violations(current_skill_text)
        if rule_violations:
            print(f"  [QC] {len(rule_violations)} rule violation(s) in this round's skill.md:")
            for v in rule_violations:
                print(f"    - {v}")

        train_rows, train_acc, train_task_scores = score_round(
            model_id, round_skill_path, train_dir, "lessons", base_args)
        train_judged = sum(1 for r in train_rows if r.get("judged_correct") not in (None, ""))
        train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "n/a"
        print(f"  [round {round_i}] TRAIN judge_accuracy={train_acc_str} "
              f"(n_judged={train_judged}/{len(train_rows)})")

        val_rows, val_acc, val_task_scores = score_round(
            model_id, round_skill_path, val_dir, "eval", base_args)
        val_judged = sum(1 for r in val_rows if r.get("judged_correct") not in (None, ""))
        val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
        print(f"  [round {round_i}] VAL   judge_accuracy={val_acc_str} "
              f"(n_judged={val_judged}/{len(val_rows)}) <- used for round selection")

        for label, rows, judged, task_scores in (("TRAIN", train_rows, train_judged, train_task_scores),
                                                   ("VAL", val_rows, val_judged, val_task_scores)):
            if judged < len(rows):
                print(f"  [QC] WARNING: only {judged}/{len(rows)} {label} rows judged this round — "
                      f"not directly comparable to a round with full coverage.")
            for b in _BUCKETS:
                s = task_scores[b]
                if s["n"]:
                    print(f"    {label} {b}: {s['accuracy']:.4f} (n={s['n']})")

        history = [h for h in history if h["round"] != round_i]
        history.append({
            "round": round_i,
            "skill_path": str(round_skill_path),
            "train_overall_judge_accuracy": train_acc,
            "train_n_rows_total": len(train_rows),
            "train_n_judged_total": train_judged,
            "train_task_scores": train_task_scores,
            "val_overall_judge_accuracy": val_acc,
            "val_n_rows_total": len(val_rows),
            "val_n_judged_total": val_judged,
            "val_task_scores": val_task_scores,
            "diff_vs_previous_round": diff_summary,
            "changelog": changelog,
            "changelog_parsed": changelog_parsed,
            "rule_violations": rule_violations,
        })
        history.sort(key=lambda h: h["round"])
        round_log_path.write_text(json.dumps(history, indent=2))
        completed_rounds.add(round_i)

        prev_train_rows, prev_train_task_scores = train_rows, train_task_scores
        prev_changelog = changelog
        current_skill_path = round_skill_path

    # Selection is ALWAYS by VAL — TRAIN score never decides the winner,
    # even though it's logged every round for trajectory-reading.
    scored = [h for h in history if h["val_overall_judge_accuracy"] is not None]
    if not scored:
        print("\n  No round produced a judged VAL score — nothing to select.")
        return
    best = max(scored, key=lambda h: h["val_overall_judge_accuracy"])
    best_out_path = out_dir / "skill_memory_best.md"
    shutil.copy(best["skill_path"], best_out_path)
    print(f"\n  Best round: {best['round']} "
          f"(VAL judge_accuracy={best['val_overall_judge_accuracy']:.4f}, "
          f"TRAIN judge_accuracy={best['train_overall_judge_accuracy']})")
    print(f"  Copied to: {best_out_path}")
    print(f"  Full trajectory: {round_log_path}")
    if best["round"] == 0:
        print("  [finding] round 0 (the UNEDITED starting file) won — no later round improved on it. "
              "The loop made things worse or flat every time it tried to edit.")
    if best.get("rule_violations"):
        print(f"  [QC] WARNING: the SELECTED best round has {len(best['rule_violations'])} rule "
              f"violation(s) — check them before trusting this file:")
        for v in best["rule_violations"]:
            print(f"    - {v}")


def main(args: argparse.Namespace) -> None:
    if not args.model:
        print("ERROR: --model is required.")
        sys.exit(1)
    if not Path(args.init_skill_path).exists():
        print(f"ERROR: --init_skill_path {args.init_skill_path!r} does not exist.")
        sys.exit(1)
    run_loop(args.model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--init_skill_path", required=True,
                         help="Starting skill.md — e.g. src/config/skill_memory_disease-pest.md. "
                              "Round 0 scores this file AS-IS before any editing.")
    parser.add_argument("--max_rounds", type=int, required=True,
                         help="Number of EDIT rounds after round 0 (so max_rounds=10 means 11 total "
                              "scoring passes: round 0 plus rounds 1-10).")
    parser.add_argument("--output_dir", default="outputs_final/heldout_skill_optimize/",
                         help="Each round writes to <output_dir>/round<i>/ (its own skill_memory.md + "
                              "a train/ and val/ heldout-zeroshot checkpoint, one per split). The "
                              "round with the best VAL score is copied to "
                              "<output_dir>/skill_memory_best.md at the end.")
    parser.add_argument("--round_log", default=None,
                         help="Default: <output_dir>/round_log.json. A list of per-round records "
                              "(TRAIN and VAL scores kept separate, mechanical diff, editor's own "
                              "changelog) — read this to see the FULL trajectory, not just which "
                              "round won.")
    parser.add_argument("--n_examples_per_bucket", type=int, default=10,
                         help="Wrong-row examples sampled per task bucket (so default 10 = 40 total, "
                              "4 buckets) and shown to each round's edit call. Bumped from an original "
                              "default of 5 after a real 10-round run never beat round 0 — the "
                              "diagnosis was that 20 examples/round is too narrow a sample to extract "
                              "genuinely generalizable guidance from, compared to round 0's own origin "
                              "(Step 3's map-reduce synthesis over ~827 full TRAIN traces); each "
                              "round's edit was a noisier, narrower signal than what it was trying to "
                              "improve on. Mind the context budget raising this further: each example "
                              "can carry up to ~500 chars of reasoning + ~500 of expert details, so 40 "
                              "examples is already ~10-15K tokens of input alongside the (growing) "
                              "skill_md itself — pushing much past this risks crowding out "
                              "--edit_max_tokens within --max_model_len.")
    parser.add_argument("--feedback_prior_round", action=argparse.BooleanOptionalAction, default=False,
                         help="OFF by default (original behavior — flip with --feedback_prior_round "
                              "to A/B test). When on, each edit call (round i>0) is ALSO shown the "
                              "immediately-prior round's own changelog (what it changed and why) "
                              "paired with the TRAIN scores that edit actually produced — so the "
                              "editor can tell whether its last change helped before making another "
                              "one. Currently the changelog is write-only (logged to --round_log for "
                              "human reading, never fed back) — this is the one flag that changes "
                              "that. Deliberately only the IMMEDIATELY-prior round, not the full "
                              "history (bounded prompt growth, avoids chasing noise across many "
                              "rounds of small-sample TRAIN scores).")
    parser.add_argument("--heldout_prompts", default="src/config/heldout_prompts.yaml")
    parser.add_argument("--manifest", default="dataset/heldout_selected_images.json")
    parser.add_argument("--images_root", default="dataset/images_heldout")
    parser.add_argument("--prompts", default="src/config/prompts.yaml")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True,
                         help="Controls ONLY the scoring/generation pass (heldout-zeroshot's own "
                              "default, unchanged) — NOT the edit call, which has its own separate "
                              "--edit_enable_thinking (default OFF). Keeping these two decoupled is "
                              "deliberate: the scoring step's thinking behavior is already validated "
                              "by Step 1/4; the edit step's is not, and defaulting it on caused a "
                              "real bug (see --edit_enable_thinking's own help).")
    parser.add_argument("--chat_template_path", default="src/config/tool_chat_template_gemma4.jinja")
    parser.add_argument("--gen_max_tokens", type=int, default=4096,
                         help="Forwarded to each round's heldout-zeroshot scoring pass.")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--edit_enable_thinking", action=argparse.BooleanOptionalAction, default=False,
                         help="OFF by default, DELIBERATELY — editing a document given scores + "
                              "examples is a text-rewrite task, not the visual diagnosis-under-"
                              "uncertainty task thinking was actually validated for. Defaulting this "
                              "on (copied from heldout-summarize's map/reduce calls without "
                              "re-justifying it for this different task) caused a real failure: the "
                              "model spent its whole --edit_max_tokens budget on an unfinished "
                              "thinking trace, the closing channel marker never appeared, "
                              "extract_channel_thought's regex failed to match, and the RAW "
                              "truncated '<|channel>thought...' text got written to skill_memory.md "
                              "as if it were the revised document. Pass --edit_enable_thinking to "
                              "turn it back on if you have a specific reason to.")
    parser.add_argument("--edit_max_tokens", type=int, default=4096,
                         help="The revised document is at least as long as the input skill file. "
                              "With --edit_enable_thinking off (the default), this budget goes "
                              "entirely to the document itself, not partly consumed by a thinking "
                              "trace — 4096 should comfortably cover a skill.md of this size with "
                              "headroom for growth across rounds.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Base seed for TRAIN miss-example sampling — offset by round number so "
                              "different rounds see different miss subsets even against the same "
                              "train split.")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--vision_token_budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--kv_cache_dtype", choices=["auto", "fp8"], default=None)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--judge_model", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    parser.add_argument("--judge_quantize", choices=["4bit", "8bit", "awq", "gptq"], default=None)
    parser.add_argument("--judge_confidence_threshold", type=int, default=1)
    parser.add_argument("--judge_chunk_size", type=int, default=200)
    parser.add_argument("--judge_enable_thinking", action="store_true")
    parser.add_argument("--judge_max_tokens", type=int, default=1024)
    parser.add_argument("--judge_chat_template_path", default=None)

    main(parser.parse_args())
