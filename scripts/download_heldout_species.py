"""
Paper 2, Stage 3 (docs/stage3_context_memory.md), Step 0c — download +
cache a held-out slice of iNatAg-mini (Project-AgML/iNatAg-mini on
HuggingFace) as a lesson-building source for a species-identification
skill memory, the same role Digital Green plays for disease/pest (Step 0)
and the 5 held-out AgML datasets play for quality (Step 0b). iNatAg-mini
is entirely external to AgML's own `species_identification` datasets
(never wired into datasets.yaml, same separation principle as Digital
Green) — none of the 25 active AgML species_identification datasets are
touched by this.

iNatAg-mini ships as one .zip PER SPECIES (998 real species currently
uploaded, confirmed via the real HF file tree — NOT the ~2959 species the
dataset card/agml package's own local metadata describes; agml's bundled
`_assets/iNatAg-mini_public_datasources.json` is used here only as a
crop/weed LABEL lookup, cross-checked against the real file tree so only
species that actually exist on HF right now are ever selected). Each zip
internally is `<species>/<crop|weed>/<image_id>.{jpg,jpeg,png}` plus an
`annotations.json` (confirmed via a real one-time manual inspection of
astrebla_lappacea.zip, not assumed) — so the label subfolder name IS the
row's crop/weed category, read directly rather than re-deriving it from
agml's own metadata a second time.

Real species-count math (40 total: 20 crop-labeled + 20 weed-labeled,
seed=42): downloads real HF zip files totaling ~3.48 GB. Each species is
then subsampled down to --images_per_species (default 50, seeded) before
being cached locally — the DOWNLOAD size is fixed by species count (an
entire zip has to be pulled regardless of how many of its images are
kept), but the LOCAL POOL size (and therefore downstream round-to-round
sampling variety) is controlled by this per-species cap: 40 * 50 = 2,000
images total (1,600 lessons / 400 eval after the usual 80/20 split).

Writes dataset/heldout_species_selected_images.json in the same row shape
download_heldout_digitalgreen.py/download_heldout_quality.py use
(image_path, crop, diagnosis as a 1-element list, details, country,
state, split), with one field reused for a real purpose rather than left
blank: "crop" holds the species' own "crop"/"weed" LABEL (from the zip's
own label subfolder), not an actual crop name — species_identification's
question has no {crop} slot to fill (see _species_question,
heldout_zeroshot.py), so this was free to repurpose; bucket_category_species
reads it directly to split Step 3's lessons into "plant"/"weed" sections.
"diagnosis" holds the species' own binomial name (underscore->space
cleaned) as a 1-element list — the real ground-truth label for a
species-identification question. "details"/"country"/"state" are always
empty — no per-row expert-note or location metadata exists here the way
Digital Green has.

Row keys and cached image paths follow the same flat convention
download_heldout_quality.py uses (images_root/<key>.jpg, no per-species
subdirectory) — keys are "<species>__<image_id>" (globally unique).

**Species-selection reproducibility**: unlike quality (whose held-out
source is a fixed, committed `skip: true` list in datasets.yaml — a
permanent, human-readable record needing no extra logging) or Digital
Green (the whole external dataset, no selection to log at all), WHICH 40
species get drawn here is a seeded random choice against the LIVE HF repo
listing — if that repo ever gains/loses species between runs, the exact
same --seed would no longer reproduce the exact same 40-species draw.
--selection_log (default dataset/heldout_species_selection.json) fixes
this: written ONCE on the first real run (species list + crop/weed label
+ the exact settings that produced it), then READ (not regenerated) on
every later run regardless of upstream repo drift or whether the first
run was interrupted partway through. This is the definitive, committed
answer to "which species were used" — check it into version control
alongside the manifest.

--eval_frac (default 0.2) splits each species' own (already-subsampled)
rows into lessons/eval independently, seeded by --seed.

Usage:
    uv run python scripts/download_heldout_species.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random
import zipfile
from collections import defaultdict

_REPO_ID = "Project-AgML/iNatAg-mini"
_AGML_METADATA_PATH = (
    Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages" / "agml" / "_assets" / "iNatAg-mini_public_datasources.json"
)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def _species_crop_weed_labels() -> dict[str, str]:
    """{species_name: "crop"|"weed"} from agml's own bundled metadata —
    a lookup only, never used to decide WHICH species exist (the real HF
    file tree, queried directly below, is the only source of truth for
    that)."""
    meta = json.loads(_AGML_METADATA_PATH.read_text())
    out = {}
    for key, v in meta.items():
        name = key.split("/", 1)[1]
        classes = v.get("classes", {})
        label = next(iter(classes.values()), None)
        if label in ("crop", "weed"):
            out[name] = label
    return out


def select_species(n_crop: int, n_weed: int, seed: int) -> list[tuple[str, str]]:
    """[(species_name, "crop"|"weed"), ...] — real species that ACTUALLY
    exist in the live HF repo (via list_repo_files), intersected with
    agml's crop/weed label lookup, then a seeded random draw of n_crop +
    n_weed from each category."""
    from huggingface_hub import list_repo_files

    real_files = list_repo_files(_REPO_ID, repo_type="dataset")
    real_species = {f[:-4] for f in real_files if f.endswith(".zip")}
    labels = _species_crop_weed_labels()

    crop_pool = sorted(s for s in real_species if labels.get(s) == "crop")
    weed_pool = sorted(s for s in real_species if labels.get(s) == "weed")

    rng = random.Random(seed)
    crop_pick = sorted(rng.sample(crop_pool, min(n_crop, len(crop_pool))))
    weed_pick = sorted(rng.sample(weed_pool, min(n_weed, len(weed_pool))))
    return [(s, "crop") for s in crop_pick] + [(s, "weed") for s in weed_pick]


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"hf_repo_id": _REPO_ID, "rows": {}}


def assign_splits(row_keys_by_species: dict[str, list[str]], eval_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    split_by_key: dict[str, str] = {}
    for name, keys in row_keys_by_species.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_eval = round(len(keys) * eval_frac)
        for i, key in enumerate(keys):
            split_by_key[key] = "eval" if i < n_eval else "lessons"
    return split_by_key


def main(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download
    from PIL import Image as PILImage

    images_root = Path(args.images_root)
    images_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.selection_log)

    manifest = load_manifest(out_path)
    rows = manifest["rows"]
    n_before = len(rows)

    if log_path.exists():
        log = json.loads(log_path.read_text())
        species_list = [(s["name"], s["category"]) for s in log["species"]]
        print(f"  Reusing species selection from {log_path} "
              f"({len(species_list)} species, logged {log['generated_at']}) — "
              f"--n_crop_species/--n_weed_species/--seed ignored for selection (delete this file "
              f"to redraw).")
    else:
        species_list = select_species(args.n_crop_species, args.n_weed_species, args.seed)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "n_crop_species": args.n_crop_species,
            "n_weed_species": args.n_weed_species,
            "species": [{"name": s, "category": c} for s, c in species_list],
        }, indent=2))
        print(f"  {len(species_list)} species selected and logged to {log_path} "
              f"({sum(1 for _, c in species_list if c == 'crop')} crop, "
              f"{sum(1 for _, c in species_list if c == 'weed')} weed)")

    rng = random.Random(args.seed)
    row_keys_by_species: dict[str, list[str]] = defaultdict(list)
    for species, category in species_list:
        display_name = species.replace("_", " ")
        # Row already fully cached for this species? Skip the zip
        # download/extraction entirely — resumable the same way
        # download_heldout_digitalgreen.py's image-cache check works.
        already_cached = [k for k in rows if k.startswith(f"{species}__")]
        if len(already_cached) >= args.images_per_species:
            row_keys_by_species[species] = already_cached[: args.images_per_species]
            continue

        print(f"  Downloading {species}.zip...")
        zip_path = hf_hub_download(repo_id=_REPO_ID, repo_type="dataset", filename=f"{species}.zip")
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m for m in zf.namelist()
                if m.startswith(f"{species}/{category}/") and Path(m).suffix in _IMAGE_EXTS
            ]
            picked = rng.sample(members, min(args.images_per_species, len(members)))
            for member in picked:
                image_id = Path(member).stem
                key = f"{species}__{image_id}"
                row_keys_by_species[species].append(key)
                img_path = images_root / f"{key}.jpg"
                if key in rows and img_path.exists():
                    continue
                with zf.open(member) as f:
                    PILImage.open(f).convert("RGB").save(img_path, format="JPEG")
                rows[key] = {
                    # "crop" holds the crop/weed LABEL, not a crop name —
                    # see this module's own docstring for why.
                    "image_path": str(img_path), "crop": category, "diagnosis": [display_name],
                    "details": "", "country": "", "state": "",
                    "agml_dataset": "iNatAg-mini_heldout",
                    "split": None,
                }

    split_by_key = assign_splits(row_keys_by_species, args.eval_frac, args.seed)
    for key, rec in rows.items():
        if key in split_by_key:
            rec["split"] = split_by_key[key]

    manifest["rows"] = rows
    out_path.write_text(json.dumps(manifest, indent=2))

    n_lessons = sum(1 for r in rows.values() if r.get("split") == "lessons")
    n_eval = sum(1 for r in rows.values() if r.get("split") == "eval")
    print(f"\n  {len(rows)} row(s) total ({len(rows) - n_before} new this run) across "
          f"{len(species_list)} species.")
    print(f"  lessons={n_lessons} eval={n_eval} (--eval_frac={args.eval_frac}, --seed={args.seed})")
    print(f"  Manifest: {out_path}")
    print(f"  Images:   {images_root}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_root", default="dataset/images_heldout_species")
    parser.add_argument("--output", default="dataset/heldout_species_selected_images.json")
    parser.add_argument("--selection_log", default="dataset/heldout_species_selection.json",
                         help="Written once on the first run (which species were chosen + why), then "
                              "READ (not regenerated) on every later run — the definitive, committed "
                              "record of species selection. Delete it to force a fresh draw.")
    parser.add_argument("--n_crop_species", type=int, default=20)
    parser.add_argument("--n_weed_species", type=int, default=20)
    parser.add_argument("--images_per_species", type=int, default=50,
                         help="Subsample cap per species (download size is unaffected — a whole zip "
                              "is downloaded regardless — this controls local pool size only).")
    parser.add_argument("--eval_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
