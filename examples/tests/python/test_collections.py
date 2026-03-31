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
