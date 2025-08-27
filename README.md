Turncoat / Swap Chess (Engine)

A minimalist Python engine for the Turncoat (a.k.a. Swap) Chess variant, built on top of python-chess.

Rules summary
- Each turn, a biased coin decides which color moves. The same color can move multiple times in a row.
- Movement uses pseudo-legal moves (moving into or remaining in check is allowed).
- Checkmate and stalemate do not end the game.
- The only win condition is capturing the opponent's king.
- v1: no castling and no en passant (promotions allowed via UCI, e.g., a7a8q).
- If the coin picks a color with zero available pseudo-legal moves, that turn is skipped (no penalty).

Install
- Requires Python 3.9+
- This project uses python-chess and pytest. Install via:
  pip install -e .

Quickstart
- The engine exposes TurncoatEngine and BiasedCoin.
- Example:
  from turncoat_chess.engine import TurncoatEngine
  from turncoat_chess.coin import BiasedCoin
  import chess

  eng = TurncoatEngine()  # starting position
  coin = BiasedCoin(p_white=0.55, seed=20250827)
  # pick a color each "turn"
  color = coin.flip()  # chess.WHITE or chess.BLACK
  moves = eng.available_moves(color)
  if moves:
      eng.apply_move(moves[0])

Testing
- Run tests with: pytest -q

Notes
- The engine deliberately ignores python-chess built-in game termination detection.
- Only capturing the king ends the game.
