"""Tests for biaffine scoring layers."""

import pytest
import torch

from classla.models.common.biaffine import BiaffineScorer


@pytest.mark.parametrize("shape", [(32,), (4, 8)])
def test_biaffine_eval_matches_native_bilinear(shape):
    torch.manual_seed(1234)
    scorer = BiaffineScorer(11, 7, 19)
    input1 = torch.randn(*shape, 11)
    input2 = torch.randn(*shape, 7)

    scorer.train()
    expected = scorer(input1, input2)
    scorer.eval()
    actual = scorer(input1, input2)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
