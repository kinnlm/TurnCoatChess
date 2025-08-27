# Project Guidelines for Juni — Turncoat/Swap Chess

## Project Structure
turncoat_chess/

init.py

engine.py # Wraps python-chess Board; move application; win detection

coin.py # BiasedCoin class (biased coin + RNG seed)

rules.py # Constants and helper functions (e.g., king capture check)

cli.py # Interactive CLI entrypoint

tests/

test_rules.py # Acceptance tests for variant behavior

test_coin.py # Tests for coin fairness/seed

pyproject.toml

README.md

## Rules of the Variant
- Each *turn* a biased coin decides which color moves. The same color can move multiple times in a row.
- Moves: standard chess piece movement, but **use pseudo-legal moves**, not legal moves.  
  → Moving into or staying in check is **allowed**.
- **Checkmate/stalemate do not end the game.**  
  → The only win condition is **capturing the king**.
- v1 excludes castling and en passant.
- Promotions allowed (UCI style, e.g., `a7a8q`).
- If the coin picks a color with no pseudo-legal moves, log a “skip” and continue.

## Development Instructions
- Use the [`python-chess`](https://pypi.org/project/chess/) library for board state, move parsing, and ASCII printing.
- All move generation must use `board.pseudo_legal_moves` and filter for coin-selected color.
- Ignore `board.is_game_over()` and similar built-ins—do not terminate except on king capture.
- Keep code PEP8-compliant and add inline docstrings.

## Testing Instructions
- Use `pytest`.
- Write tests for:
  - Moving into check is allowed.
  - Checkmate does not end the game.
  - Capturing the king ends the game.
  - Coin flips can repeat the same color.
  - Promotions work as expected.
- Run `pytest -q` after generating code.

## Build Instructions
- No special build step needed; code should run as a module.
- CLI should be runnable via:
  ```bash
  python -m turncoat_chess.cli --pwhite 0.55 --seed 20250827
