"""Bounded CPU tensor preparation and decoding around a single GPU caller."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def predict_batches(trainer, batches, depth=1):
    if depth < 1:
        raise ValueError('NER pipeline depth must be positive')
    if depth == 1:
        for batch in batches:
            yield trainer.predict(batch)
        return

    preparation = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ner-tensors')
    decoding = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ner-decoding')
    prepared, decoded = deque(), deque()
    indices = iter(range(len(batches)))
    try:
        for index in indices:
            prepared.append(preparation.submit(batches.__getitem__, index))
            if len(prepared) == depth:
                break
        while prepared:
            batch = prepared.popleft().result()
            index = next(indices, None)
            if index is not None:
                prepared.append(preparation.submit(batches.__getitem__, index))
            # CUDA work remains on this caller, never on the preparation/decoder threads.
            scores = trainer.predict_scores(batch)
            decoded.append(decoding.submit(trainer.decode_scores, scores))
            if len(decoded) == depth:
                yield decoded.popleft().result()
        while decoded:
            yield decoded.popleft().result()
    finally:
        for future in (*prepared, *decoded):
            future.cancel()
        decoding.shutdown(wait=True, cancel_futures=True)
        preparation.shutdown(wait=True, cancel_futures=True)
