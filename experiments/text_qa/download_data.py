"""
Run this once on the login node (which has internet) before submitting a batch job.
Downloads and caches all required datasets and models to ~/.cache/huggingface/.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

print("Downloading AgroCoT dataset...")
ds = load_dataset("wenyb/AgriCoT", split="train")
print(f"  {len(ds)} samples cached.")
