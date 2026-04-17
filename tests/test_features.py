import random

import numpy as np

from ximu.features import compute_features, compute_max_consecutive_Q


def test_pure_polyq():
    seq = "Q" * 48
    feat = compute_features(seq)
    assert feat.shape == (26,)
    assert feat.dtype == np.float32
    assert feat[14] > 0.95
    assert feat[10] > 0.95


def test_high_complexity():
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2 + "ACDEFGH"
    feat = compute_features(seq)
    assert feat[14] < 0.2


def test_score_order_polyq_proxy():
    short_pure_q = "Q" * 30
    long_impure_q = "Q" * 80 + "SS" + "Q" * 80
    compute_features(short_pure_q)
    compute_features(long_impure_q)
    assert compute_max_consecutive_Q(long_impure_q) > compute_max_consecutive_Q(short_pure_q)


def test_feature_range():
    aas = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(100):
        seq = "".join(random.choices(aas, k=48))
        feat = compute_features(seq)
        assert feat.min() >= 0.0
        assert feat.max() <= 1.0


def test_window_position():
    seq = "Q" * 48
    feat = compute_features(seq, window_start=24, protein_length=96)
    assert feat[25] == 0.5
