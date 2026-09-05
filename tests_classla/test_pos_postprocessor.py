"""Unit tests for POS postprocessing."""

import torch

from classla.models.pos.postprocessor import (
    InflectionalLexiconProcessor,
    SloveneInflectionalLexiconProcessor,
)


class FakeVocab:
    def __init__(self, units):
        self._units = list(units)
        self._ids = {unit: idx for idx, unit in enumerate(self._units)}

    def __iter__(self):
        return iter(self._units)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._ids[key]
        if isinstance(key, list):
            return ",".join(self._units[idx] for idx in key)
        return self._units[key]


def build_processor():
    vocab = {
        "xpos": FakeVocab(["<PAD>", "N", "Z"]),
        "upos": FakeVocab(["<PAD>", "NOUN", "PUNCT"]),
        "feats": FakeVocab(["<PAD>", "Case=Nom", "Number=Sing"]),
    }
    return InflectionalLexiconProcessor(None, vocab, None, pos_lemma_pretag=True)


def build_slovene_processor():
    vocab = {
        "xpos": FakeVocab(["<PAD>", "N", "A", "Z", "P", "S", "C", "Q"]),
        "upos": FakeVocab([
            "<PAD>", "NOUN", "ADJ", "PUNCT", "PRON",
            "DET", "ADP", "CCONJ", "SCONJ", "PART",
        ]),
        "feats": FakeVocab(["<PAD>", "Case=Nom", "Degree=Pos"]),
    }
    lexicon = [
        ("znana", "N", "NOUN", "Case=Nom", "znan"),
        ("dvoumna", "N", "NOUN", "Case=Nom", "dvoumnež"),
        ("dvoumna", "A", "ADJ", "Degree=Pos", "dvoumen"),
        ("dvoumna", "A", "NOUN", "Case=Nom", "ignored-duplicate"),
    ]
    return SloveneInflectionalLexiconProcessor(lexicon, vocab, None, pos_lemma_pretag=True)


def test_postprocessor_predictions():
    processor = build_processor()
    words = [["beseda", "."]]

    xpos_logits = torch.tensor([[[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]]])
    xpos = processor.process_xpos(xpos_logits, words)
    assert xpos == [["N", "N"]]

    upos_logits = torch.tensor([[[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]]])
    upos = processor.process_upos(upos_logits, words, xpos)
    assert upos == [["NOUN", "NOUN"]]

    feat_predictions = [
        torch.tensor([[[1], [2]]]),
        torch.tensor([[[2], [1]]]),
    ]
    feats = processor.process_feats(feat_predictions, words, xpos, upos)
    assert feats == [["Case=Nom,Number=Sing", "Number=Sing,Case=Nom"]]


def test_postprocessor_does_not_extract_individual_tensor_scalars():
    processor = build_processor()
    words = [["beseda", "."]]
    logits = torch.tensor([[[0.0, 4.0, 1.0], [0.0, 1.0, 4.0]]])
    feat_predictions = [torch.tensor([[[1], [2]]]), torch.tensor([[[2], [1]]])]

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        xpos = processor.process_xpos(logits, words)
        upos = processor.process_upos(logits, words, xpos)
        processor.process_feats(feat_predictions, words, xpos, upos)

    scalar_extractions = sum(
        event.count for event in profile.key_averages() if event.key == "aten::item"
    )
    assert scalar_extractions == 0


def test_slovene_lexicon_constraints_are_preserved():
    processor = build_slovene_processor()
    words = [["znana", "dvoumna", "neznana"]]

    xpos_logits = torch.tensor([[
        [0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    xpos = processor.process_xpos(xpos_logits, words)
    assert xpos == [["N", "A", "N"]]

    upos_logits = torch.tensor([[
        [0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]])
    upos = processor.process_upos(upos_logits, words, xpos)
    assert upos == [["NOUN", "ADJ", "NOUN"]]

    feat_predictions = [torch.tensor([[[2], [1], [2]]])]
    feats = processor.process_feats(feat_predictions, words, xpos, upos)
    assert feats == [["Case=Nom", "Degree=Pos", "Degree=Pos"]]


def test_slovene_lexicon_builds_reusable_lemma_dictionary():
    processor = build_slovene_processor()

    assert processor.lemma_dictionary == {
        ("znana", "N"): "znan",
        ("dvoumna", "N"): "dvoumnež",
        ("dvoumna", "A"): "dvoumen",
    }
