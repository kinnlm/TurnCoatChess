"""Turncoat/Swap Chess engine on top of python-chess.

Variant specifics:
- Each turn, a biased coin decides which color moves (external to engine).
- Movement uses pseudo-legal moves only; moving into or staying in check is allowed.
- Checkmate/stalemate do not end the game; only capturing the king ends the game.
- v1 excludes castling and en passant. Promotions are allowed via UCI.
"""
from __future__ import annotations

from typing import List, Optional

import chess

from .rules import is_castling_or_en_passant, is_king_captured_before_push


class TurncoatEngine:
    """Minimal engine wrapper.

    Attributes
    - board: chess.Board representing the current state
    - winner: Optional[chess.Color] set to the color who captured the opponent's king
    - forced_color: Optional[chess.Color] helper for tests/UX prompts (does not affect rules)
    """

    def __init__(self, fen: Optional[str] = None) -> None:
        self.board = chess.Board(fen=fen) if fen else chess.Board()
        self.winner: Optional[chess.Color] = None
        self.forced_color: Optional[chess.Color] = None

    # --- Helpers for variant semantics ---
    def available_moves(self, color: chess.Color) -> List[chess.Move]:
        """Return a list of pseudo-legal moves for `color`, excluding castling/en passant.

        We filter python-chess pseudo-legal moves to those whose from-square matches `color`.
        """
        moves: List[chess.Move] = []
        for mv in self.board.pseudo_legal_moves:
            if self.board.color_at(mv.from_square) != color:
                continue
            if is_castling_or_en_passant(self.board, mv):
                continue
            moves.append(mv)
        return moves

    def is_pseudo_legal_for_color(self, move: chess.Move, color: chess.Color) -> bool:
        """Check that `move` is pseudo-legal for `color` under variant rules.

        This ignores check constraints and blocks castling/en passant.
        """
        if self.board.color_at(move.from_square) != color:
            return False
        if is_castling_or_en_passant(self.board, move):
            return False
        return move in self.board.pseudo_legal_moves

    def apply_move(self, move: chess.Move) -> None:
        """Apply a move to the board. If a king is captured, set `winner` to mover's color.

        We detect king capture by inspecting the target square before the push. We then
        push the move regardless of checks, consistent with pseudo-legal movement rules.
        """
        if self.winner is not None:
            return  # game already ended by king capture

        mover = self.board.color_at(move.from_square)
        if mover is None:
            raise ValueError("No piece on the move's from_square")

        # Enforce variant's move admissibility minimally
        if not self.is_pseudo_legal_for_color(move, mover):
            raise ValueError("Move is not pseudo-legal for this color under variant rules")

        king_captured = is_king_captured_before_push(self.board, move)
        self.board.push(move)
        if king_captured:
            self.winner = mover

    def force_color(self, color: chess.Color) -> None:
        """Testing/helper hook to remember a prompted color (does not affect rules)."""
        self.forced_color = color
