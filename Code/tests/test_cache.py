import threading

import pytest

from cache import LRUCache


def test_evicts_least_recently_used():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # "a" is now most recent
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3


def test_tracks_hit_rate():
    cache = LRUCache(4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("missing")
    assert cache.stats() == {"size": 1, "maxsize": 4, "hits": 1, "misses": 1, "hit_rate": 0.5}


def test_rejects_non_positive_size():
    with pytest.raises(ValueError):
        LRUCache(0)


def test_concurrent_access_keeps_size_bounded():
    cache = LRUCache(50)

    def worker(offset):
        for i in range(2000):
            cache.put((offset, i % 80), i)
            cache.get((offset, (i * 7) % 80))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats()
    assert len(cache) == 50
    assert stats["hits"] + stats["misses"] == 8 * 2000
