#!/usr/bin/env python3
"""
Swap Chess (Turncoat Chess) — v1 CLI prototype

Design goals:
- Standard chess pieces and movement rules (no castling or en passant in v1).
- Turn order is decided by a coin flip each turn; same color may move multiple times in a row.
- Moving into, through, or remaining in check is allowed (we ignore check legality).
- Win condition is CAPTURE THE KING (checkmate is not a win by itself).
- Optional biased coin; deterministically seedable RNG for repeatable games.
- Minimal, dependency‑free, text‑mode board with UCI‑style moves (e2e4, a7a8q for promotion).

Notes:
- This is a clean, readable baseline to iterate on. We can add castling/en passant, UI, bots, and tests later.
- Coordinates: files a..h map to 0..7, ranks 8..1 map to rows 0..7 (row 0 is the top/8th rank).
- White pawns start at row 6 moving up (row index -1); Black pawns start at row 1 moving down (+1).

Usage:
    python3 swap_chess.py                 # default fair coin, random seed
    python3 swap_chess.py --pwhite 0.55   # biased coin (55% White to move each flip)
    python3 swap_chess.py --seed 12345    # deterministic coin sequence

Controls:
- Enter moves like "e2e4". Promotions: "e7e8q" (q/r/b/n). "help" to see help. "board" to reprint.
- "resign" to concede for the currently selected side. "seed" shows RNG seed. "quit" exits.

License: MIT (do whatever, just keep attribution)
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable

FILES = "abcdefgh"
RANKS = "87654321"

WHITE = "W"
BLACK = "B"
COLORS = (WHITE, BLACK)

PIECE_NAMES = {
    "P": "Pawn",
    "N": "Knight",
    "B": "Bishop",
    "R": "Rook",
    "Q": "Queen",
    "K": "King",
}


# --- Helpers: coordinates ---

def in_bounds(r: int, f: int) -> bool:
    return 0 <= r < 8 and 0 <= f < 8


def sq_to_rf(sq: str) -> Tuple[int, int]:
    """Convert algebraic square (e.g., 'e2') to (row, file) indices.
    row 0 is rank 8, file 0 is 'a'."""
    if len(sq) != 2 or sq[0] not in FILES or sq[1] not in "12345678":
        raise ValueError(f"Bad square: {sq}")
    f = FILES.index(sq[0])
    # rank '8' -> row 0, rank '1' -> row 7
    r = RANKS.index(sq[1])
    return r, f


def rf_to_sq(r: int, f: int) -> str:
    return f"{FILES[f]}{RANKS[r]}"


# --- Data types ---

@dataclass
class Piece:
    color: str
    kind: str  # one of 'P', 'N', 'B', 'R', 'Q', 'K'

    def char(self) -> str:
        return self.kind if self.color == WHITE else self.kind.lower()


@dataclass
class Move:
    sr: int
    sf: int
    dr: int
    df: int
    promo: Optional[str] = None  # 'Q','R','B','N' (uppercase)

    @classmethod
    def parse(cls, text: str) -> "Move":
        t = text.strip().lower()
        if t in {"help", "board", "resign", "quit", "seed"}:
            # Sentinel commands handled by CLI
            raise ValueError("not-a-move sentinel")
        # UCI-like: e2e4 or e7e8q
        if len(t) not in (4, 5):
            raise ValueError("enter moves like e2e4 or e7e8q")
        sr, sf = sq_to_rf(t[0:2])
        dr, df = sq_to_rf(t[2:4])
        promo = None
        if len(t) == 5:
            p = t[4]
            if p not in "qrbn":
                raise ValueError("promotion piece must be one of q,r,b,n")
            promo = p.upper()
        return cls(sr, sf, dr, df, promo)

    def uci(self) -> str:
        s = rf_to_sq(self.sr, self.sf) + rf_to_sq(self.dr, self.df)
        if self.promo:
            s += self.promo.lower()
        return s


class Board:
    def __init__(self):
        self.grid: List[List[Optional[Piece]]] = [[None for _ in range(8)] for _ in range(8)]
        self._setup()

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.grid = [[None if p is None else Piece(p.color, p.kind) for p in row] for row in self.grid]
        return b

    def _setup(self):
        # Place pieces in starting chess position
        def place_back_rank(r: int, color: str):
            order = "RNBQKBNR"
            for f, k in enumerate(order):
                self.grid[r][f] = Piece(color, k)

        # Black at top rows 0-1 (ranks 8 and 7), White at bottom rows 7-6 (ranks 1 and 2)
        place_back_rank(0, BLACK)
        for f in range(8):
            self.grid[1][f] = Piece(BLACK, "P")
        for f in range(8):
            self.grid[6][f] = Piece(WHITE, "P")
        place_back_rank(7, WHITE)

    # --- Rendering ---
    def __str.

        me__(self) -> str:
    # pretty printer (internal)
    lines = []
    for r in range(8):
        row = []
        for f in range(8):
            p = self.grid[r][f]
            row.append(p.char() if p else ("·" if (r + f) % 2 else " "))
        lines.append(f"{RANKS[r]}  " + " ".join(row))
    lines.append("\n    a b c d e f g h")
    return "\n".join(lines)


def print(self):
    print(self.__str.me__())


# --- Queries ---
def piece_at(self, r: int, f: int) -> Optional[Piece]:
    return self.grid[r][f]


def find_king(self, color: str) -> Optional[Tuple[int, int]]:
    target = "K" if color == WHITE else "k"
    for r in range(8):
        for f in range(8):
            p = self.grid[r][f]
            if p and p.char() == target:
                return r, f
    return None


def color_pieces(self, color: str) -> Iterable[Tuple[int, int, Piece]]:
    for r in range(8):
        for f in range(8):
            p = self.grid[r][f]
            if p and p.color == color:
                yield r, f, p


# --- Move gen (ignores checks; no castling/en passant) ---
def legal_moves_for(self, color: str) -> List[Move]:
    moves: List[Move] = []
    for r, f, p in self.color_pieces(color):
        moves.extend(self._piece_moves(r, f, p))
    return moves


def _piece_moves(self, r: int, f: int, p: Piece) -> List[Move]:
    kind = p.kind
    if p.color == BLACK:
        # We'll use uppercase kinds internally; p.kind is uppercase already
        pass
    if kind == "P":
        return self._pawn_moves(r, f, p.color)
    elif kind == "N":
        return self._knight_moves(r, f, p.color)
    elif kind == "B":
        return self._slide_moves(r, f, p.color, directions=[(-1, -1), (-1, 1), (1, -1), (1, 1)])
    elif kind == "R":
        return self._slide_moves(r, f, p.color, directions=[(-1, 0), (1, 0), (0, -1), (0, 1)])
    elif kind == "Q":
        return self._slide_moves(r, f, p.color,
                                 directions=[(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)])
    elif kind == "K":
        return self._king_moves(r, f, p.color)
    else:
        return []


def _pawn_moves(self, r: int, f: int, color: str) -> List[Move]:
    m: List[Move] = []
    dir = -1 if color == WHITE else 1
    start_row = 6 if color == WHITE else 1
    promo_row = 0 if color == WHITE else 7

    # single step
    nr = r + dir
    if in_bounds(nr, f) and self.grid[nr][f] is None:
        if nr == promo_row:
            for P in ("Q", "R", "B", "N"):
                m.append(Move(r, f, nr, f, promo=P))
        else:
            m.append(Move(r, f, nr, f))
        # double step from start if clear
        nr2 = r + 2 * dir
        if r == start_row and in_bounds(nr2, f) and self.grid[nr2][f] is None:
            m.append(Move(r, f, nr2, f))

    # captures
    for df in (-1, 1):
        nr = r + dir
        nf = f + df
        if in_bounds(nr, nf):
            target = self.grid[nr][nf]
            if target and target.color != color:
                if nr == promo_row:
                    for P in ("Q", "R", "B", "N"):
                        m.append(Move(r, f, nr, nf, promo=P))
                else:
                    m.append(Move(r, f, nr, nf))
    # No en passant in v1
    return m


def _knight_moves(self, r: int, f: int, color: str) -> List[Move]:
    m: List[Move] = []
    for dr, df in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
        nr, nf = r + dr, f + df
        if in_bounds(nr, nf):
            tgt = self.grid[nr][nf]
            if tgt is None or tgt.color != color:
                m.append(Move(r, f, nr, nf))
    return m


def _slide_moves(self, r: int, f: int, color: str, directions: List[Tuple[int, int]]) -> List[Move]:
    m: List[Move] = []
    for dr, df in directions:
        nr, nf = r + dr, f + df
        while in_bounds(nr, nf):
            tgt = self.grid[nr][nf]
            if tgt is None:
                m.append(Move(r, f, nr, nf))
            else:
                if tgt.color != color:
                    m.append(Move(r, f, nr, nf))
                break
            nr += dr
            nf += df
    return m


def _king_moves(self, r: int, f: int, color: str) -> List[Move]:
    m: List[Move] = []
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            nr, nf = r + dr, f + df
            if in_bounds(nr, nf):
                tgt = self.grid[nr][nf]
                if tgt is None or tgt.color != color:
                    m.append(Move(r, f, nr, nf))
    # No castling in v1
    return m


# --- Apply moves ---
def apply(self, mv: Move) -> Tuple[Optional[Piece], Optional[Piece]]:
    """Apply a move to the board in-place.
    Returns (mover_before, captured) for logging. Promotion handled here.
    """
    mover = self.grid[mv.sr][mv.sf]
    if mover is None:
        raise ValueError("No piece at source square")
    captured = self.grid[mv.dr][mv.df]
    # Move piece
    self.grid[mv.dr][mv.df] = mover
    self.grid[mv.sr][mv.sf] = None
    # Promotion
    if mv.promo:
        self.grid[mv.dr][mv.df] = Piece(mover.color, mv.promo)
    return mover, captured


def is_move_legal(self, color: str, mv: Move) -> bool:
    # Generate and check membership to ensure path/ownership rules
    return any((x.sr, x.sf, x.dr, x.df, x.promo) == (mv.sr, mv.sf, mv.dr, mv.df, mv.promo) for x in
               self._piece_moves(mv.sr, mv.sf, self.grid[mv.sr][mv.sf]) if self.grid[mv.sr][mv.sf])


class BiasedCoin:
    def __init__(self, p_white: float = 0.5, rng: random.Random | None = None):
        if not (0.0 <= p_white <= 1.0):
            raise ValueError("p_white must be in [0,1]")
        self.pw = p_white
        self.rng = rng or random.Random()

    def flip(self) -> str:
        return WHITE if self.rng.random() < self.pw else BLACK


class Game:
    def __init__(self, p_white: float = 0.5, seed: Optional[int] = None):
        self.board = Board()
        self.rng = random.Random(seed)
        self.coin = BiasedCoin(p_white, self.rng)
        self.history: List[str] = []  # coin+move strings
        self.seed = seed
        self.winner: Optional[str] = None

    def available_moves(self, color: str) -> List[Move]:
        return self.board.legal_moves_for(color)

    def step(self) -> None:
        """Run one turn: flip coin to pick color, then request/execute one move for that color."""
        if self.winner:
            return
        color = self.coin.flip()
        print(f"\n— Coin flip: {'WHITE' if color == WHITE else 'BLACK'} to move —")
        moves = self.available_moves(color)
        if not moves:
            print(f"No legal moves for {color}. Turn is skipped.")
            self.history.append(f"{color}:skip")
            return
        self._prompt_and_move(color)

    def _prompt_and_move(self, color: str):
        while True:
            try:
                raw = input(f"{color} move (e.g., e2e4, 'help' for options): ").strip()
                if raw.lower() == "help":
                    print("Commands: e2e4 (move), e7e8q (promote), board, resign, seed, quit")
                    continue
                if raw.lower() == "board":
                    self.board.print()
                    continue
                if raw.lower() == "seed":
                    print(f"Current seed: {self.seed}")
                    continue
                if raw.lower() == "resign":
                    other = WHITE if color == BLACK else BLACK
                    self.winner = other
                    print(f"{color} resigns. {other} wins by resignation.")
                    return
                if raw.lower() == "quit":
                    print("Exiting game.")
                    sys.exit(0)

                mv = Move.parse(raw)
                if not self.board.piece_at(mv.sr, mv.sf):
                    print("No piece on source square.")
                    continue
                if self.board.piece_at(mv.sr, mv.sf).color != color:
                    print("That piece is not yours to move.")
                    continue
                if not self.board.is_move_legal(color, mv):
                    print("Illegal move for that piece (path or capture rules).")
                    continue

                mover, captured = self.board.apply(mv)
                self.history.append(f"{color}:{mv.uci()}")

                # Check king capture win condition
                if captured and captured.kind == "K":
                    self.winner = color
                    print(f"King captured on {rf_to_sq(mv.dr, mv.df)} — {color} wins!")
                    return

                # Optional: print if side is in check (informational only)
                # We'll compute 'attacked squares' to warn, but it's not a rule.
                opp = WHITE if color == BLACK else BLACK
                kpos = self.board.find_king(opp)
                if kpos and is_square_attacked(self.board, kpos[0], kpos[1], color):
                    print(f"Note: {opp}'s king is currently in check (capture is needed to win).")
                self.board.print()
                break
            except ValueError as e:
                print(f"Input error: {e}")


def is_square_attacked(board: Board, r: int, f: int, attacker_color: str) -> bool:
    """Lightweight check helper: does attacker_color attack square (r,f)?
    Uses the same move gen and tests if any move ends on (r,f)."""
    for sr, sf, p in board.color_pieces(attacker_color):
        for mv in board._piece_moves(sr, sf, p):
            if mv.dr == r and mv.df == f:
                return True
    return False


# --- CLI Entrypoint ---

def main():
    ap = argparse.ArgumentParser(description="Swap Chess (Turncoat Chess) — capture-the-king variant")
    ap.add_argument("--pwhite", type=float, default=0.5, help="Probability that WHITE moves on a coin flip (0..1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for deterministic coin flips")
    args = ap.parse_args()

    game = Game(p_white=args.pwhite, seed=args.seed)
    print("Swap Chess v1 — Coin decides who moves. You must capture the king to win.")
    print("Rules: moving into/through check is allowed. No castling/en passant in this prototype.")
    print(f"Coin bias: P(WHITE)= {args.pwhite:.3f}. Seed: {args.seed}")
    game.board.print()

    while not game.winner:
        game.step()

    print("\nGame over. History:")
    for i, h in enumerate(game.history, 1):
        print(f"{i:3d}. {h}")


if __name__ == "__main__":
    main()
