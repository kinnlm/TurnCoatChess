"""Rules helpers and constants for Turncoat/Swap Chess.

Key points for v1:
- Use pseudo-legal moves only.
- No castling or en passant.
- Capturing the king is the only win condition.
"""
from __future__ import annotations

import chess


def is_castling_or_en_passant(board: chess.Board, move: chess.Move) -> bool:
    """Return True if the move is a castling or en passant move (disallowed in v1)."""
    # python-chess helpers for clarity
    return board.is_castling(move) or board.is_en_passant(move)


def is_king_captured_before_push(board: chess.Board, move: chess.Move) -> bool:
    """Detect whether pushing `move` would capture a king.

    We inspect the target square before the push and check if it is occupied by a
    king of the opposite color. We deliberately ignore python-chess's game-over flags.

    En passant is excluded in v1, so simple target-square inspection is sufficient.
    """
    if is_castling_or_en_passant(board, move):
        return False
    captured = board.piece_at(move.to_square)
    if captured is None:
        return False
    return captured.piece_type == chess.KING
