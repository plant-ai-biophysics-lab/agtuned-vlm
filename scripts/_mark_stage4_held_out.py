"""One-off: mark the full Stage 4 skill-memory pool (docs/stage4_detection_
context_memory.md) skip: true (held out from Stage 2/6's own active pool)
in src/config/detection_datasets.yaml. Final design: of the ~54-dataset
catalog, 10 go to TRAIN, 5 to VAL (both used during skill optimization —
TRAIN steers the edit, VAL selects the best round), 15 total marked
skip: true here. The remaining ~39 are TEST — read for a final
confirmatory score only, never used to generate lessons, so they stay in
Stage 2/6's own active pool untouched (no marking needed). Which of these
15 are TRAIN vs VAL is decided in scripts/download_heldout_detection.py's
own TRAIN_DATASETS/VAL_DATASETS lists, not here — this script only needs
to get all 15 out of Stage 2/6's active pool, role assignment happens at
download time. Safe to re-run — already-marked entries are left alone.
Run once, then delete. Not wired into main.py."""
import yaml
from pathlib import Path

path = Path("src/config/detection_datasets.yaml")
text = path.read_text()
lines = text.splitlines(keepends=True)

note = ("HELD OUT for Paper 2 Stage 4 skill-memory lessons/eval "
        "(docs/stage4_detection_context_memory.md) -- do not flip skip back to false, do not "
        "include in Stage 2/6 sampling. See scripts/download_heldout_detection.py.")

# 10 TRAIN + 5 VAL = 15 total. GEMINI_cowpea_flower/pod_detection and
# grape_detection_californiaday were an earlier VAL pick (real domain-pair
# counterparts to the synthetic TRAIN datasets) — moved to TEST (left
# active, untouched) and replaced with 3 more generally diverse VAL picks
# per explicit direction, since VAL's job here is a representative
# validation set, not a strict domain-paired sim-to-real test.
targets = [
    # TRAIN (10)
    "synthetic_cowpea_flower_detection",
    "synthetic_cowpea_pod_detection",
    "grape_detection_syntheticday",
    "mango_growth_detection",
    "MH_Weed16_weed_detection",
    "ghai_strawberry_fruit_detection",
    "cashew_detection",
    "weed_crop_detection",
    "tomato_ripeness_detection",
    "onionfoliageset_detection",
    # VAL (5)
    "embrapa_wgisd_grape_detection",
    "grape_detection_californianight",
    "wheat_head_counting",
    "CocoaMFDB_detection",
    "coffee_detection",
]

i = 0
n_done = 0
while i < len(lines):
    line = lines[i]
    matched = None
    for name in targets:
        if line.strip() == "- agml_name: " + name:
            matched = name
    if matched:
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("- agml_name:"):
            if lines[j].strip() == "skip: false":
                lines[j] = lines[j].replace("skip: false", "skip: true")
            if lines[j].strip() == "notes: ''":
                lines[j] = lines[j].replace("notes: ''", "notes: '" + note + "'")
            j += 1
        n_done += 1
        print("marked held-out: " + matched)
        i = j
        continue
    i += 1

path.write_text("".join(lines))
print("patched " + str(n_done) + "/" + str(len(targets)) + " entries -> " + str(path))

parsed = yaml.safe_load(path.read_text())
by_name = {e["agml_name"]: e for e in parsed["datasets"]}
missing_or_wrong = [t for t in targets if not by_name.get(t, {}).get("skip")]
if missing_or_wrong:
    raise SystemExit("FAILED to mark held-out (check spelling/presence in the file): " + str(missing_or_wrong))
print("verified: all " + str(len(targets)) + " target(s) are skip: true")
n_active = sum(1 for e in parsed["datasets"] if not e["skip"])
n_holdout_marked = sum(1 for e in parsed["datasets"] if str(e.get("notes", "")).startswith("HELD OUT for Paper 2 Stage 4"))
print("active (Stage 2/6 pool):", n_active, "| Stage 4 held-out total:", n_holdout_marked)
