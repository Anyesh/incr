import threading

import pytest
from incr_concurrent import Runtime


def test_basic_dag_matches_local_semantics():
    rt = Runtime()
    a = rt.create_input(10)
    b = rt.create_query(lambda r: r.get(a) * 2)
    assert rt.get(b) == 20
    rt.set(a, 15)
    assert rt.get(b) == 30


def test_runtime_is_shareable_across_threads():
    rt = Runtime()
    price = rt.create_input(100.0)
    taxed = rt.create_query(lambda r: r.get(price) * 1.08)
    assert rt.get(taxed) == pytest.approx(108.0)

    errors = []

    def reader():
        try:
            for _ in range(200):
                v = rt.get(taxed)
                assert 0 < v <= 1000 * 1.08 + 1
        except Exception as e:  # noqa: BLE001 - collected and re-raised below
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for v in range(1, 201):
        rt.set(price, float(v))
    for t in threads:
        t.join()

    assert not errors
    assert rt.get(taxed) == pytest.approx(200 * 1.08)


def test_collections_shareable_across_threads():
    rt = Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    count = evens.count()

    def writer(base):
        for i in range(50):
            col.insert(base + i)

    threads = [threading.Thread(target=writer, args=(b,)) for b in (0, 1000, 2000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert rt.get(count) == 75  # 25 evens per 50-element block


def test_exception_propagates_in_concurrent_module():
    rt = Runtime()
    a = rt.create_input(0)
    q = rt.create_query(lambda r: 10 // r.get(a))
    with pytest.raises(ZeroDivisionError):
        rt.get(q)
