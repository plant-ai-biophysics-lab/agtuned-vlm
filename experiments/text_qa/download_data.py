"""
Run once on the login node before submitting batch jobs.
Saves AgroCoT to a local directory so compute nodes don't need internet.

Usage:
    uv run python download_data.py --output_dir /path/to/save
"""

import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data/agrocot")
args = parser.parse_args()

print("Downloading KisanVaani agriculture QA dataset...")
ds = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
ds.save_to_disk(args.output_dir)
print(f"  {len(ds)} samples saved to {args.output_dir}")
