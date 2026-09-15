"""Inference regressions; real-model checks opt in via CLASSLA_TEST_NER_MODEL."""

import os

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
