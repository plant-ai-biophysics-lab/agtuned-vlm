"""Shared progress-logging helpers for ag_finetune data_prep scripts."""

from __future__ import annotations

import time

TOTAL_STAGES = 6


def stage_banner(stage_num: int, name: str) -> None:
    bar = "=" * 74
    print(f"\n{bar}\n STAGE {stage_num}/{TOTAL_STAGES}: {name}\n{bar}")


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
