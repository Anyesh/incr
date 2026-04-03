import incr
from incr import Runtime


def test_collection_count():
    rt = Runtime()
    col = rt.create_collection()
    count = col.count()
    assert rt.get(count) == 0
    col.insert(1)
    assert rt.get(count) == 1
    col.insert(2)
    assert rt.get(count) == 2
    col.delete(1)
    assert rt.get(count) == 1


def test_collection_filter():
    rt = Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    count = evens.count()

    col.insert(1)
    col.insert(2)
    col.insert(3)
    col.insert(4)
    assert rt.get(count) == 2


def test_collection_map():
    rt = Runtime()
    col = rt.create_collection()
    doubled = col.map(lambda x: x * 2)
    count = doubled.count()

    col.insert(1)
    col.insert(2)
    col.insert(3)
    assert rt.get(count) == 3


def test_collection_pipeline():
    rt = Runtime()
    col = rt.create_collection()
    result = col.filter(lambda x: x > 0).filter(lambda x: x < 10).map(lambda x: x * 2)
    count = result.count()

    col.insert(-5)
    col.insert(3)
    col.insert(15)
    col.insert(7)
    assert rt.get(count) == 2


def test_collection_delete_propagates():
    rt = Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    count = evens.count()

    col.insert(2)
    col.insert(4)
    assert rt.get(count) == 2

    col.delete(2)
    assert rt.get(count) == 1


def test_reduce_sum():
    rt = incr.Runtime()
    col = rt.create_collection()
    total = col.reduce(lambda elements: sum(elements))

    assert rt.get(total) == 0
    col.insert(10)
    assert rt.get(total) == 10
    col.insert(20)
    assert rt.get(total) == 30
    col.delete(10)
    assert rt.get(total) == 20


def test_reduce_after_filter():
    rt = incr.Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    total = evens.reduce(lambda elements: sum(elements))

    col.insert(1)
    col.insert(2)
    col.insert(4)
    assert rt.get(total) == 6

    col.delete(2)
    assert rt.get(total) == 4


def test_sort_by_key():
    rt = incr.Runtime()
    col = rt.create_collection()
    sorted_col = col.sort_by_key(lambda x: x)

    col.insert(30)
    col.insert(10)
    col.insert(20)

    rt.get(sorted_col.version_node)
    assert sorted_col.entries() == [10, 20, 30]


def test_sort_by_key_delete():
    rt = incr.Runtime()
    col = rt.create_collection()
    sorted_col = col.sort_by_key(lambda x: x)

    col.insert(10)
    col.insert(20)
    col.insert(30)
    rt.get(sorted_col.version_node)
    assert sorted_col.entries() == [10, 20, 30]

    col.delete(20)
    rt.get(sorted_col.version_node)
    assert sorted_col.entries() == [10, 30]


def test_pairwise():
    rt = incr.Runtime()
    col = rt.create_collection()
    sorted_col = col.sort_by_key(lambda x: x)
    pairs = sorted_col.pairwise()

    col.insert(10)
    col.insert(20)
    col.insert(30)

    pair_count = pairs.count()
    assert rt.get(pair_count) == 2


def test_sort_pairwise_map_reduce_pipeline():
    rt = incr.Runtime()
    visits = rt.create_collection()
    sorted_visits = visits.sort_by_key(lambda t: t)
    pairs = sorted_visits.pairwise()
    gaps = pairs.map(lambda pair: pair[1] - pair[0])
    total = gaps.reduce(lambda elements: sum(elements))

    visits.insert(10)
    visits.insert(30)
    visits.insert(50)
    assert rt.get(total) == 40

    visits.insert(20)
    assert rt.get(total) == 40  # same total

    visits.delete(30)
    assert rt.get(total) == 40  # still same

    visits.insert(100)
    assert rt.get(total) == 90
