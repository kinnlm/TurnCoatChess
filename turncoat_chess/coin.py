"""Biased coin for Turncoat/Swap Chess.

flip() returns chess.WHITE or chess.BLACK according to p_white.
Deterministic when constructed with a seed.
"""
from __future__ import annotations

import random
import chess
from typing import Optional


class BiasedCoin:
    """A biased coin with optional deterministic seed.

    Parameters
    - p_white: float in [0, 1], probability of returning chess.WHITE
    - seed: optional int, if provided will seed an internal RNG
    """

    def __init__(self, p_white: float = 0.5, seed: Optional[int] = None) -> None:
        if not (0.0 <= p_white <= 1.0):
            raise ValueError("p_white must be in [0, 1]")
        self.p_white = p_white
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def flip(self) -> chess.Color:
        """Return chess.WHITE with probability p_white, else chess.BLACK."""
        return chess.WHITE if self._rng.random() < self.p_white else chess.BLACK
