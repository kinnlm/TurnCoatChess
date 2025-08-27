import chess
from turncoat_chess.coin import BiasedCoin


def test_coin_can_repeat_same_color_all_white():
    coin = BiasedCoin(p_white=1.0, seed=123)
    flips = [coin.flip() for _ in range(10)]
    assert all(c == chess.WHITE for c in flips)


def test_coin_bias_zero_all_black():
    coin = BiasedCoin(p_white=0.0, seed=456)
    flips = [coin.flip() for _ in range(10)]
    assert all(c == chess.BLACK for c in flips)
