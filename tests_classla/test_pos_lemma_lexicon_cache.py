"""Unit tests for the reusable POS-to-lemma lexicon cache."""

import os

from classla.models.pos.trainer import Trainer


def test_inflectional_lexicon_cache_preserves_first_entry(tmp_path):
    model_path = tmp_path / "pos.pt"
    model_path.write_bytes(b"checkpoint signature")
    entries = [
        ("word", "N", "NOUN", "_", "first"),
        ("word", "N", "NOUN", "_", "second"),
        ("word", "V", "VERB", "_", "verb"),
    ]

    result = Trainer.load_inflectional_lexicon(model_path, entries=entries)

    assert result == {("word", "N"): "first", ("word", "V"): "verb"}
    assert (tmp_path / "pos.pt.lemma_lexicon.pkl").is_file()


def test_inflectional_lexicon_cache_is_reused_and_invalidated(tmp_path):
    model_path = tmp_path / "pos.pt"
    model_path.write_bytes(b"first signature")
    original = [("word", "N", "NOUN", "_", "cached")]
    assert Trainer.load_inflectional_lexicon(model_path, entries=original) == {
        ("word", "N"): "cached"
    }

    replacement = [("other", "V", "VERB", "_", "replacement")]
    assert Trainer.load_inflectional_lexicon(model_path, entries=replacement) == {
        ("word", "N"): "cached"
    }

    model_path.write_bytes(b"different checkpoint signature")
    stat = model_path.stat()
    os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert Trainer.load_inflectional_lexicon(model_path, entries=replacement) == {
        ("other", "V"): "replacement"
    }
