import threading

import pytest

from classla.models.ner.inference import predict_batches


class Trainer:
    def __init__(self):
        self.gpu_threads = set()

    def predict_scores(self, batch):
        self.gpu_threads.add(threading.get_ident())
        return batch

    def decode_scores(self, scores):
        return [scores]

    def predict(self, batch):
        return self.decode_scores(self.predict_scores(batch))


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_order_and_gpu_owner(depth):
    trainer = Trainer()
    assert list(predict_batches(trainer, list(range(17)), depth)) == [[i] for i in range(17)]
    assert trainer.gpu_threads == {threading.get_ident()}


@pytest.mark.parametrize('stage', ['prepare', 'gpu', 'decode'])
def test_failure_cleans_up(stage):
    class Batches(list):
        def __getitem__(self, index):
            if stage == 'prepare':
                raise RuntimeError('failed')
            return super().__getitem__(index)

    class Broken(Trainer):
        def predict_scores(self, batch):
            if stage == 'gpu':
                raise RuntimeError('failed')
            return super().predict_scores(batch)

        def decode_scores(self, scores):
            if stage == 'decode':
                raise RuntimeError('failed')
            return super().decode_scores(scores)

    with pytest.raises(RuntimeError, match='failed'):
        list(predict_batches(Broken(), Batches(range(10)), 2))
    assert not any(t.name.startswith(('ner-tensors', 'ner-decoding')) for t in threading.enumerate())


def test_early_close_and_empty():
    assert list(predict_batches(Trainer(), [], 2)) == []
    stream = predict_batches(Trainer(), list(range(10)), 2)
    next(stream)
    stream.close()
    assert not any(t.name.startswith(('ner-tensors', 'ner-decoding')) for t in threading.enumerate())
