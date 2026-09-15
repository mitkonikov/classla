"""Inference regressions; real-model checks opt in via CLASSLA_TEST_NER_MODEL."""

import os

import numpy as np
import pytest
import torch

from classla.models.ner.trainer import Trainer


def test_predict_skips_loss_and_gradients():
    class Model(torch.nn.Module):
        def forward(self, *args, **kwargs):
            assert kwargs['compute_loss'] is False
            assert not torch.is_grad_enabled()
            return None, torch.tensor([[[0., 1.]]]), torch.zeros(2, 2)

    class Tags:
        def unmap(self, ids):
            return ['O' for _ in ids]

    trainer = Trainer.__new__(Trainer)
    trainer.model = Model()
    trainer.use_cuda = False
    trainer.vocab = {'tag': Tags()}
    value = torch.ones(1, 1, dtype=torch.long)
    batch = (value,) * 6 + ([0], [0], [0], [1], [1], [1], None)
    assert trainer.predict(batch) == [['O']]


@pytest.mark.skipif(not os.environ.get('CLASSLA_TEST_NER_MODEL'), reason='Local NER model required')
def test_real_model_loss_and_predictions():
    import classla
    from classla.models.ner.data import DataLoader
    from classla.models.ner.trainer import unpack_batch

    pipeline = classla.Pipeline('sl', processors='tokenize,ner', use_gpu=False, verbose=False)
    document = pipeline.processors['tokenize'].process('Janez Novak živi v Ljubljani. Marija je doma.')
    processor = pipeline.processors['ner']
    trainer = processor.trainer
    loader = DataLoader(document, 16, processor.config, vocab=trainer.vocab,
                        evaluation=True, preprocess_tags=False)
    batch = loader[0]
    fast_loader = DataLoader(document, 16, processor.config, vocab=trainer.vocab,
                             evaluation=True, preprocess_tags=False, inference_only=True)
    fast_batch = fast_loader[0]
    assert fast_batch[4] is None  # Slovenian NER does not use charLM.
    assert fast_batch[5] is None  # No training targets during prediction.
    assert trainer.predict(batch) == trainer.predict(fast_batch)
    inputs, orig, word_orig, char_orig, sentlens, wordlens, charlens, offsets = unpack_batch(batch, False)
    word, mask, wordchars, charmask, chars, tags = inputs
    args = (word, mask, wordchars, charmask, tags, word_orig, sentlens, wordlens,
            chars, offsets, charlens, char_orig)
    trainer.model.eval()
    with torch.inference_mode():
        loss, expected, transitions = trainer.model(*args)
        skipped, actual, actual_transitions = trainer.model(*args, compute_loss=False)
    assert torch.isfinite(loss)
    assert skipped is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_transitions, transitions, rtol=0, atol=0)
    trainer.model.train()
    loss, _, _ = trainer.model(*args)
    loss.backward()
    assert trainer.model.tag_clf.weight.grad is not None


def test_charlm_inputs_are_preserved():
    from classla.models.common.doc import Document
    from classla.models.ner.data import DataLoader

    class Vocab:
        def map(self, values):
            return [4] * len(values)

        def unit2id(self, value):
            return 4

    document = Document([[{'id': 1, 'text': 'Janez'}, {'id': 2, 'text': 'Novak'}]])
    vocab = {key: Vocab() for key in ('word', 'char', 'tag')}
    loader = DataLoader(document, 16, {'charlm': True}, vocab=vocab,
                        evaluation=True, preprocess_tags=False, inference_only=True)
    batch = loader[0]
    assert batch[4].shape[0] == 2
    assert batch[5] is None
    assert len(batch[12]) == 2


def test_length_batches_restore_original_order():
    from classla.models.common.doc import Document
    from classla.models.common.utils import unsort
    from classla.models.ner.data import DataLoader

    class Vocab:
        def map(self, values):
            return [4] * len(values)

    sentences = [[{'id': i + 1, 'text': 'word'} for i in range(n)] for n in (2, 5, 1, 3)]
    document = Document(sentences)
    loader = DataLoader(document, 2, {'charlm': False},
                        vocab={key: Vocab() for key in ('word', 'char', 'tag')},
                        evaluation=True, preprocess_tags=False, inference_only=True, sort_during_eval=True)
    assert loader.data_orig_idx == [1, 3, 0, 2]
    assert unsort([5, 3, 2, 1], loader.data_orig_idx) == [2, 5, 1, 3]
    empty = DataLoader(Document([]), 2, {}, vocab={}, evaluation=True,
                       preprocess_tags=False, inference_only=True, sort_during_eval=True)
    assert len(empty) == 0
    assert empty.data_orig_idx == []


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('ties', [False, True])
def test_batched_viterbi_matches_scalar(dtype, ties):
    from classla.models.common.crf import viterbi_decode, viterbi_decode_batch

    rng = np.random.default_rng(42)
    for lengths in ([1], [1, 17, 3, 8], [31] * 16, [1, 200, 2, 1, 3]):
        scores = rng.normal(size=(len(lengths), max(lengths), 9)).astype(dtype)
        transitions = rng.normal(size=(9, 9)).astype(dtype)
        if ties:
            scores.fill(0)
            transitions.fill(0)
        expected = [viterbi_decode(scores[i, :length], transitions)[0]
                    for i, length in enumerate(lengths)]
        assert viterbi_decode_batch(scores, transitions, lengths) == expected


def test_batched_viterbi_empty_and_invalid_lengths():
    from classla.models.common.crf import viterbi_decode_batch

    assert viterbi_decode_batch(np.zeros((0, 0, 2)), np.zeros((2, 2)), []) == []
    with pytest.raises(ValueError, match='positive'):
        viterbi_decode_batch(np.zeros((1, 2, 2)), np.zeros((2, 2)), [0])
