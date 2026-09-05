"""Unit tests for POS inference."""

import torch

from classla.models.pos.trainer import Trainer


class FakeModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        self.compute_loss = kwargs["compute_loss"]
        self.grad_enabled = torch.is_grad_enabled()
        return 0.0, [
            torch.tensor([[1]]),
            torch.tensor([[1]]),
            torch.tensor([[[1]]]),
        ]


class FakeVocab:
    def unmap(self, values):
        return [str(value) for value in values]


def test_predict_disables_gradients_and_loss_computation():
    trainer = Trainer.__new__(Trainer)
    trainer.use_cuda = False
    trainer.model = FakeModel()
    trainer.postprocessor = None
    trainer.vocab = {
        "upos": FakeVocab(),
        "xpos": FakeVocab(),
        "feats": FakeVocab(),
    }

    scalar = torch.tensor([[1]])
    batch = (
        scalar, scalar, scalar, scalar,
        scalar, scalar, scalar, scalar,
        [0], [0], [1], [1], [["beseda"]],
    )
    trainer.predict(batch)

    assert trainer.model.compute_loss is False
    assert trainer.model.grad_enabled is False
