import numpy as np
import pytest

pytest.importorskip('numba')
from classla.models.ner.numba_decode import decode
from classla.models.common.crf import viterbi_decode_batch


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_numba_matches_reference(dtype):
    rng = np.random.default_rng(42)
    for lengths in ([1], [1, 100, 2, 8], [31] * 16):
        for tied in (False, True):
            scores = rng.normal(size=(len(lengths), max(lengths), 9)).astype(dtype)
            transitions = rng.normal(size=(9, 9)).astype(dtype)
            if tied:
                scores.fill(0)
                transitions.fill(0)
            assert decode(scores, transitions, lengths) == viterbi_decode_batch(scores, transitions, lengths)


def test_empty_invalid_and_nonfinite():
    assert decode(np.zeros((0, 0, 2)), np.zeros((2, 2)), []) == []
    with pytest.raises(ValueError):
        decode(np.zeros((1, 2, 2)), np.zeros((2, 2)), [0])
    scores = np.array([[[np.nan, 0.], [0., 1.]]])
    transitions = np.zeros((2, 2))
    assert decode(scores, transitions, [2]) == viterbi_decode_batch(scores, transitions, [2])
