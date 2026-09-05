"""Unit tests for the bounded runtime lemma prediction cache."""

from classla.pipeline.lemma_processor import _LemmaPredictionCache


def test_prediction_cache_reuses_and_evicts_least_recently_used_values():
    cache = _LemmaPredictionCache(2)
    cache.put(("prva", "N"), "prvi")
    cache.put(("druga", "N"), "drugi")

    assert cache.get(("prva", "N")) == "prvi"
    cache.put(("tretja", "N"), "tretji")

    assert cache.get(("druga", "N")) is None
    assert cache.get(("prva", "N")) == "prvi"
    assert cache.get(("tretja", "N")) == "tretji"


def test_zero_sized_prediction_cache_is_disabled():
    cache = _LemmaPredictionCache(0)
    cache.put(("beseda", "N"), "beseda")

    assert cache.get(("beseda", "N")) is None
