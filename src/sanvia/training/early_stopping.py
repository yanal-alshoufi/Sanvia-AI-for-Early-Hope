"""Early stopping on a score to maximise."""

from __future__ import annotations

import math
from typing import Any


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best: float = -math.inf
        self.counter = 0
        self.should_stop = False

    def reset(self) -> None:
        self.counter, self.should_stop = 0, False

    def step(self, score: float) -> bool:
        """Return True if ``score`` improved. NaN counts as no improvement."""
        if not math.isnan(score) and score > self.best + self.min_delta:
            self.best, self.counter = score, 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "counter": self.counter, "should_stop": self.should_stop}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best, self.counter, self.should_stop = state["best"], state["counter"], state["should_stop"]
