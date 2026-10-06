"""One-off: fill class_names for synthetic_cowpea_{flower,pod}_detection in
src/config/hf_detection_catalog.json (resolved manually — see the real
probed row's "categories": [0], no ClassLabel/.names metadata anywhere to
resolve it automatically; class names chosen to match the sibling
GEMINI_cowpea_flower_detection=['flower']/GEMINI_cowpea_pod_detection=['pods']
naming convention already in the catalog, per explicit user confirmation).
Run once, then delete. Not wired into main.py — not a reusable stage."""
import json
from pathlib import Path

path = Path("src/config/hf_detection_catalog.json")
cat = json.loads(path.read_text())

fills = {
    "synthetic_cowpea_flower_detection": ["flower"],
    "synthetic_cowpea_pod_detection": ["pods"],
}

for name, class_names in fills.items():
    if name not in cat:
        raise SystemExit("not found in catalog: " + name)
    current = cat[name]["class_names"]
    if current is not None:
        raise SystemExit("already has class_names, not overwriting: " + name + " -> " + str(current))
    cat[name]["class_names"] = class_names
    print(name + ": class_names -> " + str(class_names))

path.write_text(json.dumps(cat, indent=2, sort_keys=True))
print("wrote " + str(path))
