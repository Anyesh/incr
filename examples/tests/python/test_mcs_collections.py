import pytest

from incr import Runtime


class Boom(Exception):
    pass


class Row:
    def __init__(self, key, raises=False):
        self.key = key
        self.raises = raises

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if self.raises or other.raises:
            raise Boom(self.key)
        return self.key == other.key


def readers(col):
    return {
        "values": col.reduce(lambda xs: sorted(xs)),
        "count": col.count(),
        "sum": col.aggregate(0, lambda x: x, lambda a, b: a + b),
        "groups": col.group_by(lambda x: x % 3),
    }


def read(rt, nodes):
    groups = nodes["groups"]
    rt.get(groups.version_node)
    sizes = {}
    for key in groups.keys():
        size = rt.get(groups.get_group(key).count())
        if size:
            sizes[key] = size
    return (
        rt.get(nodes["values"]),
        rt.get(nodes["count"]),
        rt.get(nodes["sum"]),
        sizes,
    )


def test_extend_matches_one_by_one_inserts():
    rt = Runtime()
    batched = rt.create_collection()
    single = rt.create_collection()
    batched_nodes = readers(batched)
    single_nodes = readers(single)
    read(rt, batched_nodes)
    read(rt, single_nodes)

    for rows in ([1, 2, 3, 3, 4], [], [5, 1, 6, 9, 9, 9]):
        batched.extend(rows)
        for row in rows:
            single.insert(row)
        assert read(rt, batched_nodes) == read(rt, single_nodes)

    single.delete(9)
    batched.delete(9)
    assert read(rt, batched_nodes) == read(rt, single_nodes)


def test_extend_takes_any_iterable_and_advances_the_version_as_inserts_do():
    rt = Runtime()
    batched = rt.create_collection()
    single = rt.create_collection()
    batched.extend(x for x in range(4))
    for x in range(4):
        single.insert(x)
    assert rt.get(batched.version_node) == rt.get(single.version_node) == 4
    assert rt.get(batched.count()) == 4


def test_extend_whose_row_eq_raises_leaves_the_collection_unchanged():
    rt = Runtime()
    col = rt.create_collection()
    col.extend([Row(1), Row(2)])
    count = col.count()
    assert rt.get(count) == 2
    version = rt.get(col.version_node)

    with pytest.raises(Boom):
        col.extend([Row(3), Row(4), Row(2, raises=True), Row(5)])

    assert rt.get(count) == 2
    assert rt.get(col.version_node) == version
    keys = rt.get(col.reduce(lambda rows: sorted(r.key for r in rows)))
    assert keys == [1, 2]

    col.extend([Row(3)])
    assert rt.get(count) == 3


def test_extend_whose_row_iteration_raises_leaves_the_collection_unchanged():
    rt = Runtime()
    col = rt.create_collection()
    col.insert(1)

    def rows():
        yield 2
        raise Boom("mid-iteration")

    with pytest.raises(Boom):
        col.extend(rows())
    assert rt.get(col.count()) == 1


def test_replace_swaps_one_value_for_another():
    rt = Runtime()
    col = rt.create_collection()
    col.extend([1, 2])
    count = col.count()
    total = col.aggregate(0, lambda x: x, lambda a, b: a + b)
    assert rt.get(total) == 3

    assert col.replace(2, 5) is True
    assert rt.get(count) == 2
    assert rt.get(total) == 6

    assert col.replace(42, 7) is False
    assert rt.get(count) == 3
    assert rt.get(total) == 13
