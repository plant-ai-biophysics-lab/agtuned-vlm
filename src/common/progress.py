"""
Shared progress-logging helpers — stage banners, sub-banners, duration
formatting, and a rate/ETA-tracking Progress class for long-running batched
inference loops.

Originally built for experiments/ag_finetune/data_prep/ (see that directory's
_progress.py, left in place and unchanged so the already-running data_prep
pipeline isn't touched). Promoted here as the canonical shared copy for reuse
across any future multi-stage or multi-batch script — import from here going
forward rather than re-deriving this pattern per script.

Usage:
  from src.common.progress import Progress, stage_banner, sub_banner, fmt_duration

  stage_banner(1, "Verifier loop")
  prog = Progress(total=len(items), label="Round 1 description", every=32)
  for batch in batches:
      ...
      prog.update(len(batch))
"""

from __future__ import annotations

import time


def stage_banner(stage_num: int, name: str, total_stages: int | None = None) -> None:
    bar = "=" * 74
    label = f"STAGE {stage_num}/{total_stages}: {name}" if total_stages else f"STAGE {stage_num}: {name}"
    print(f"\n{bar}\n {label}\n{bar}")


def sub_banner(text: str) -> None:
    print(f"\n--- {text} ---")


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    """Tracks elapsed/rate/ETA for a loop with a known total and prints a
    one-line status update. Call .update(n) after each chunk of work.
    """

    def __init__(self, total: int, label: str = "", every: int = 1) -> None:
        self.total = max(total, 1)
        self.label = label
        self.every = max(every, 1)
        self.done = 0
        self.t0 = time.time()

    def update(self, n: int = 1, extra: str = "") -> None:
        self.done += n
        is_last = self.done >= self.total
        if not is_last and self.done % self.every != 0:
            return
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0
        eta_s = (self.total - self.done) / rate if rate > 0 else 0
        pct = 100 * self.done / self.total
        msg = (f"  [{self.label}] {self.done:,}/{self.total:,} ({pct:.1f}%) "
               f"| elapsed {fmt_duration(elapsed)} | rate {rate:.1f}/s "
               f"| ETA {fmt_duration(eta_s)}")
        if extra:
            msg += f" | {extra}"
        print(msg)
