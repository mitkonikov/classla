"""Optional exact Viterbi CPU kernel; import only when explicitly requested."""

import numpy as np

try:
    from numba import njit
except ImportError as error:
    raise ImportError('Numba NER decoding requires classla[ner-jit]') from error


@njit(cache=True, nogil=True)
def _decode(scores, transitions, lengths):
    batch, steps, tags = scores.shape
    output = np.empty((batch, steps), dtype=np.int64)
    for row in range(batch):
        previous = scores[row, 0].copy()
        current = previous.copy()
        back = np.empty((lengths[row], tags), dtype=np.int64)
        for step in range(1, lengths[row]):
            for target in range(tags):
                best = previous[0] + transitions[0, target]
                index = 0
                for source in range(1, tags):
                    candidate = previous[source] + transitions[source, target]
                    if candidate > best:
                        best = candidate
                        index = source
                current[target] = scores[row, step, target] + best
                back[step, target] = index
            previous, current = current, previous
        last = 0
        for tag in range(1, tags):
            if previous[tag] > previous[last]:
                last = tag
        output[row, lengths[row] - 1] = last
        for step in range(lengths[row] - 1, 0, -1):
            last = back[step, last]
            output[row, step - 1] = last
    return output


def decode(scores, transitions, lengths):
    from classla.models.common.crf import viterbi_decode_batch

    lengths = np.asarray(lengths, dtype=np.int64)
    if (scores.ndim != 3 or lengths.shape != (scores.shape[0],)
            or np.any(lengths < 1) or np.any(lengths > scores.shape[1])
            or transitions.shape != (scores.shape[2], scores.shape[2])):
        raise ValueError('Invalid NER score, transition or length dimensions')
    # Preserve the reference behavior for non-finite inputs and unusual dtypes.
    if (scores.dtype not in (np.float32, np.float64) or scores.dtype != transitions.dtype
            or not np.isfinite(scores).all() or not np.isfinite(transitions).all()):
        return viterbi_decode_batch(scores, transitions, lengths)
    output = _decode(scores, transitions, lengths)
    return [output[row, :length].tolist() for row, length in enumerate(lengths)]
