"""
Simple progress reporting utilities.

Purpose:
- Provide non-intrusive progress indication for long succinct MCMC runs.
- Avoid heavy dependencies; can be swapped with tqdm if user wants.

Initial:
    - A tiny textual progress bar / periodic logging.
Later:
    - Hooks for user-provided callbacks.
"""

import sys
from typing import Optional


class SimpleProgressBar:
    """
    Very small progress bar for terminal output.

    Usage:
        pb = SimpleProgressBar(total=T)
        for i in range(T):
            ... work ...
            pb.update(1)
        pb.close()
    """

    def __init__(self, total: int, width: int = 40, file=None):
        self.total = max(1, total)
        self.width = width
        self.file = file or sys.stderr
        self.current = 0
        self._closed = False
        self._last_frac_drawn: Optional[float] = None

    def update(self, n: int = 1) -> None:
        if self._closed:
            return
        self.current += n
        frac = min(1.0, self.current / self.total)
        # Only redraw if changed enough to matter
        if self._last_frac_drawn is not None and frac - self._last_frac_drawn < 0.01:
            return
        self._last_frac_drawn = frac

        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"\r[{bar}] {frac:6.1%}", end="", file=self.file)
        self.file.flush()

    def close(self) -> None:
        if not self._closed:
            self.update(0)
            print(file=self.file)
            self._closed = True
