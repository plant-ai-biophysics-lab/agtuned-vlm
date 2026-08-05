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

import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


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


class ProgressBar:
    """Single-line, in-place progress bar — percent + ETA, redrawn on the
    same line (\\r) as work happens, committed with exactly one trailing
    newline when the chunk it's tracking finishes. Not tqdm: tqdm assumes
    it owns the terminal line exclusively, but these extraction loops
    sometimes interleave other prints mid-loop (load/unload messages,
    fallback [note] lines), which corrupts a tqdm bar's redraw — this one
    only ever touches its own line and explicitly hands the line back (one
    newline) at the end of each chunk/model/step, so the next bar always
    starts clean rather than overwriting or being overwritten.

    When stdout isn't a real terminal (e.g. a SLURM log file redirected to
    disk), \\r-driven redraws would just produce one line per item instead
    of one updating line — so in that case updates are throttled to
    whole-percent-point changes, each written as its own newline-terminated
    line, matching how the log would read if `Progress` above were used.

    Usage — drop-in for the `for x in tqdm(samples, desc=...)` sites this
    replaces:

        for sample in ProgressBar(samples, desc="Qwen3.5-9B embeddings"):
            ...

    Or manual control when the loop body doesn't map 1:1 to the iterable:

        bar = ProgressBar(total=n_batches, desc="training")
        for batch in loader:
            ...
            bar.update()
        bar.finish()
    """

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        total: int | None = None,
        desc: str = "",
        width: int = 30,
        stream=sys.stdout,
    ) -> None:
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.iterable = iterable
        self.total = total if total and total > 0 else None
        self.desc = desc
        self.width = width
        self.stream = stream
        self.is_tty = bool(getattr(stream, "isatty", lambda: False)())

        self.n = 0
        self._start = time.monotonic()
        self._last_len = 0
        self._last_pct_shown = -1
        self._done = False
        self._draw()

    def __iter__(self) -> Iterator[T]:
        assert self.iterable is not None, "ProgressBar built with total= only has no __iter__"
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.finish()

    def update(self, n: int = 1) -> None:
        if self._done:
            return
        self.n += n
        self._draw()

    def finish(self) -> None:
        if self._done:
            return
        already_at_final = self.total is not None and self.n >= self.total
        if self.total is not None:
            self.n = self.total
        self._done = True
        # Skip the redundant redraw if the last update() already landed
        # exactly on the final state (the common case) — non-tty mode would
        # otherwise print the same 100% line twice.
        if not already_at_final:
            self._draw(force=True)
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    def _draw(self, force: bool = False) -> None:
        elapsed = time.monotonic() - self._start
        rate = self.n / elapsed if elapsed > 0 else 0.0

        if self.total is not None:
            frac = min(self.n / self.total, 1.0)
            pct = frac * 100
            remaining = (self.total - self.n) / rate if rate > 0 else float("nan")
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            count = f"{self.n}/{self.total}"
            head = f"{self.desc + ' ' if self.desc else ''}[{bar}] {pct:5.1f}%"
            eta_str = fmt_duration(remaining) if remaining == remaining and remaining >= 0 else "--"
        else:
            # Unknown total (e.g. a plain generator) — no bar/percent/ETA
            # possible, just show throughput so the loop still looks alive.
            pct = None
            count = str(self.n)
            head = f"{self.desc + ' ' if self.desc else ''}{count} done"
            eta_str = "--"

        rate_str = f"{rate:.1f}it/s" if rate > 0 else "?it/s"
        line = f"{head}  {count}  ETA {eta_str}  ({rate_str})"

        if self.is_tty:
            pad = max(self._last_len - len(line), 0)
            self.stream.write("\r" + line + " " * pad)
            self.stream.flush()
            self._last_len = len(line)
            return

        # Non-tty: throttle to whole-percent-point changes (or the forced
        # final draw) so a redirected log file gets one line per percent
        # point, not one line per item.
        shown_pct = int(pct) if pct is not None else None
        if force or shown_pct is None or shown_pct != self._last_pct_shown:
            self.stream.write(line + "\n")
            self.stream.flush()
            if shown_pct is not None:
                self._last_pct_shown = shown_pct
