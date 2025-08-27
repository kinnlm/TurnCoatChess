import chess
from turncoat_chess.engine import TurncoatEngine


def test_moving_into_check_is_allowed():
    # Kings on e1 (white) and e2 (black)
    fen = "8/8/8/8/8/8/4k3/4K3 w - - 0 1"
    eng = TurncoatEngine(fen=fen)
    eng.force_color(chess.WHITE)

    # Move white king from e1 to d1 which is attacked by the black king on e2 in orthodox chess
    move = chess.Move.from_uci("e1d1")

    assert eng.is_pseudo_legal_for_color(move, chess.WHITE)
    eng.apply_move(move)
    # Game should not be over (no king was captured)
    assert eng.winner is None


def test_checkmate_does_not_end_game():
    # Scholar's mate position just delivered (black to move would be checkmated in orthodox chess)
    # FEN after 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? 4.Qxf7#: White to move would not be, but we setup a checkmated black.
    # Simpler: set a classic checkmate: Black king on h8, white queen on g7, white king on h6, black to move.
    fen = "7k/6Q1/7K/8/8/8/8/8 b - - 0 1"
    eng = TurncoatEngine(fen=fen)
    # Even though it's checkmate in orthodox chess, our engine should not consider the game over.
    assert eng.winner is None


def test_king_capture_ends_game():
    # Simple capture: white rook on a1, black king on a8, clear file
    fen = "k7/8/8/8/8/8/8/R6K w - - 0 1"
    eng = TurncoatEngine(fen=fen)
    # White plays a1a8 capturing the black king
    move = chess.Move.from_uci("a1a8")
    assert eng.is_pseudo_legal_for_color(move, chess.WHITE)
    eng.apply_move(move)
    assert eng.winner == chess.WHITE


def test_promotion_flow_works():
    # White pawn ready to promote on a7
    fen = "8/P7/8/8/8/8/4k3/4K3 w - - 0 1"
    eng = TurncoatEngine(fen=fen)
    move = chess.Move.from_uci("a7a8q")
    assert eng.is_pseudo_legal_for_color(move, chess.WHITE)
    eng.apply_move(move)
    # Ensure a white queen now sits on a8 and game not ended by promotion itself
    piece = eng.board.piece_at(chess.A8)
    assert piece is not None and piece.piece_type == chess.QUEEN and piece.color == chess.WHITE
    assert eng.winner is None
