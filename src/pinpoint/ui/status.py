"""Animated thinking spinner — map-pin / GPS themed."""

from __future__ import annotations

import itertools
import threading
import time
from typing import Optional

from rich.console import Console

# Map-pin frames — pulsing pin drop animation
_PIN_FRAMES = ["📍", "📌", "🗺 ", "📍", "📌", "🌐"]
_FALLBACK_FRAMES = ["◉", "◎", "○", "◎", "◉", "●"]

# Rotating thinking words — location/business themed
_THINKING_PHASES = {
    "Thinking": [
        "Scanning economic data",
        "Reading GDP signals",
        "Analyzing state metrics",
        "Weighing your priorities",
        "Mapping the landscape",
    ],
    "Running parse_business_profile": [
        "Parsing business profile",
        "Extracting requirements",
        "Structuring your input",
    ],
    "Running get_location_recommendations": [
        "Scoring 50 states",
        "Computing similarity",
        "Ranking locations",
        "Evaluating priorities",
        "Building recommendations",
    ],
}

_DEFAULT_WORDS = [
    "Processing",
    "Working",
    "Analyzing",
    "Computing",
]


class ThinkingStatus:
    """Displays an animated spinner while the agent is running."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._label = "Thinking"
        self._lock = threading.Lock()

    def start(self, label: str = "Thinking") -> None:
        with self._lock:
            if self._active:
                self._label = label
                return
            self._label = label
            self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def update(self, label: str) -> None:
        with self._lock:
            self._label = label

    def stop(self) -> None:
        with self._lock:
            self._active = False
        if self._thread:
            self._thread.join(timeout=1.0)
        # Clear the spinner line
        self._console.print("\r" + " " * 60 + "\r", end="")

    def _spin(self) -> None:
        frames = _FALLBACK_FRAMES
        word_cycle: dict[str, itertools.cycle] = {}
        start = time.time()
        frame_iter = itertools.cycle(enumerate(frames))

        while True:
            with self._lock:
                if not self._active:
                    break
                label = self._label

            i, frame = next(frame_iter)
            words = _THINKING_PHASES.get(label, _DEFAULT_WORDS)

            if label not in word_cycle:
                word_cycle[label] = itertools.cycle(words)

            # Rotate word every 2.5 seconds
            elapsed = time.time() - start
            word_idx = int(elapsed / 2.5) % len(words)
            word = words[word_idx]

            elapsed_str = _fmt_elapsed(elapsed)
            line = f"\r  [bold cyan]{frame}[/bold cyan]  [dim]{word}...[/dim]  [dim italic]({elapsed_str})[/dim italic]"
            self._console.print(line, end="", highlight=False)
            time.sleep(0.15)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"
