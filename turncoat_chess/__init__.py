"""Turncoat/Swap Chess public API.

Exposes:
- TurncoatEngine: main engine wrapper around python-chess Board.
- BiasedCoin: biased coin with optional RNG seed.
"""
from .engine import TurncoatEngine
from .coin import BiasedCoin

__all__ = ["TurncoatEngine", "BiasedCoin"]
