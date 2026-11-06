import pytest

import incr
from incr import Incomplete, LookupFailed, NotReady, Pending, Runtime, Unavailable


def test_not_ready_types_share_a_base_and_incomplete_is_not_one():
    for value in (Pending(), LookupFailed("timeout"), Unavailable("quarantined")):
        assert isinstance(value, NotReady)
    assert not isinstance(Incomplete(1, 1), NotReady)


def test_fields_default_and_read_back():
    assert Pending().key is None
    assert Pending(key="k").key == "k"
    failed = LookupFailed("timeout", key=3)
    assert (failed.reason, failed.key) == ("timeout", 3)
    assert LookupFailed("timeout").key is None
    quarantined = Unavailable("quarantined", row=(1, "a"))
    assert (quarantined.reason, quarantined.row) == ("quarantined", (1, "a"))
    assert Unavailable("quarantined").row is None
    incomplete = Incomplete(10, 2)
    assert (incomplete.value, incomplete.missing) == (10, 2)


def test_equality_is_by_type_and_fields_and_equal_values_hash_alike():
    assert Pending(key=1) == Pending(key=1)
    assert Pending(key=1) != Pending(key=2)
    assert Pending() != LookupFailed("x")
    assert LookupFailed("x", key=1) == LookupFailed("x", key=1)
    assert LookupFailed("x", key=1) != LookupFailed("y", key=1)
    assert Unavailable("q", row=1) == Unavailable("q", row=1)
    assert Unavailable("q", row=1) != Unavailable("q")
    assert Incomplete(1, 2) == Incomplete(1, 2)
    assert Incomplete(1, 2) != Incomplete(1, 3)
    assert Incomplete(1, 2) != 1
    assert hash(Pending(key=1)) == hash(Pending(key=1))
    assert hash(Unavailable("q", row=(1, 2))) == hash(Unavailable("q", row=(1, 2)))
    assert hash(Incomplete(1, 2)) == hash(Incomplete(1, 2))
    assert len({Pending(key=1), Pending(key=1), Unavailable("q")}) == 2


def test_values_are_frozen_and_have_a_repr():
    with pytest.raises(AttributeError):
        Pending(key=1).key = 2
    with pytest.raises(AttributeError):
        Incomplete(1, 1).missing = 2
    assert repr(Unavailable("q", row=1)) == "Unavailable('q', row=1)"
    assert repr(Pending(key="k")) == "Pending(key='k')"
    assert repr(LookupFailed("t", key=1)) == "LookupFailed('t', key=1)"
    assert repr(Incomplete(5, 2)) == "Incomplete(5, missing=2)"


def test_incomplete_needs_something_missing():
    with pytest.raises(ValueError):
        Incomplete(1, 0)


def test_the_signal_is_not_an_exception():
    assert issubclass(incr._NotReadySignal, BaseException)
    assert not issubclass(incr._NotReadySignal, Exception)


def test_a_quarantined_input_makes_its_dependents_unavailable():
    rt = Runtime()
    shift = rt.create_input(8)
    pay = rt.create_query(lambda ctx: ctx.get(shift) * 20)
    total = rt.create_query(lambda ctx: ctx.get(pay) + 100)
    assert rt.get(total) == 260

    rt.set(shift, Unavailable("quarantined"))
    assert rt.get(pay) == Unavailable("quarantined")
    assert rt.get(total) == Unavailable("quarantined")

    rt.set(shift, 9)
    assert rt.get(total) == 280


def test_except_exception_cannot_swallow_a_not_ready_read():
    rt = Runtime()
    leg = rt.create_input(Pending(key="leg"))

    def score(ctx):
        try:
            return ctx.get(leg)
        except Exception:
            return "swallowed"

    q = rt.create_query(score)
    assert rt.get(q) == Pending(key="leg")


def test_try_get_follows_the_same_rule():
    rt = Runtime()
    leg = rt.create_input(LookupFailed("timeout", key="leg"))
    q = rt.create_query(lambda ctx: ctx.try_get(leg))
    assert rt.get(q) == LookupFailed("timeout", key="leg")


def test_a_python_exception_still_fails_the_node():
    rt = Runtime()
    x = rt.create_input(0)

    def boom(ctx):
        return 1 / ctx.get(x)

    q = rt.create_query(boom)
    with pytest.raises(ZeroDivisionError):
        rt.get(q)


def test_reading_an_incomplete_value_gives_its_value_and_carries_missing():
    rt = Runtime()
    total = rt.create_input(Incomplete(100, 2))
    doubled = rt.create_query(lambda ctx: ctx.get(total) * 2)
    assert rt.get_with_status(doubled) == (200, Incomplete(200, 2))
    assert rt.get(doubled) == 200


def test_missing_counts_contributions_not_distinct_rows():
    rt = Runtime()
    a = rt.create_input(Incomplete(1, 1))
    b = rt.create_input(Incomplete(2, 1))
    both = rt.create_query(lambda ctx: ctx.get(a) + ctx.get(a) + ctx.get(b))
    assert rt.status(both) == Incomplete(4, 3)


def test_a_nested_recompute_keeps_its_own_missing():
    rt = Runtime()
    source = rt.create_input(Incomplete(10, 2))
    plain = rt.create_input(5)
    child = rt.create_query(lambda ctx: ctx.get(source) + 1)
    parent = rt.create_query(lambda ctx: ctx.get(plain) + ctx.get(child) * 2)
    assert rt.get_with_status(parent) == (27, Incomplete(27, 2))
    assert rt.status(child) == Incomplete(11, 2)

    rt.set(plain, 6)
    assert rt.status(parent) == Incomplete(28, 2)
    rt.set(source, 10)
    assert rt.get_with_status(parent) == (28, None)


def test_a_returned_incomplete_adds_the_frame_missing():
    rt = Runtime()
    source = rt.create_input(Incomplete(10, 2))
    q = rt.create_query(lambda ctx: Incomplete(ctx.get(source), 1))
    assert rt.status(q) == Incomplete(10, 3)


def test_get_status_opts_out_of_both_rules():
    rt = Runtime()
    pending = rt.create_input(Pending(key="k"))
    partial = rt.create_input(Incomplete(7, 1))
    plain = rt.create_input(3)
    q = rt.create_query(
        lambda ctx: (
            ctx.get_status(pending),
            ctx.get_status(partial),
            ctx.get_status(plain),
        )
    )
    assert rt.get_with_status(q) == (
        (
            (Pending(key="k"), Pending(key="k")),
            (7, Incomplete(7, 1)),
            (3, None),
        ),
        None,
    )


def test_top_level_reads_and_status_agree():
    rt = Runtime()
    plain = rt.create_input(1)
    pending = rt.create_input(Pending())
    partial = rt.create_input(Incomplete(2, 1))
    assert rt.get(plain) == 1
    assert rt.get(pending) == Pending()
    assert rt.get(partial) == 2
    assert rt.try_get(partial) == 2
    assert rt.get_with_status(plain) == (1, None)
    assert rt.get_with_status(pending) == (Pending(), Pending())
    assert rt.get_with_status(partial) == (2, Incomplete(2, 1))
    assert rt.status(plain) is None
    assert rt.status(pending) == Pending()
    assert rt.status(partial) == Incomplete(2, 1)


def test_status_of_watched_outputs_after_a_drain():
    rt = Runtime()
    shift = rt.create_input(8)
    total = rt.create_query(lambda ctx: ctx.get(shift) + 1)
    rt.get(total)
    rt.watch(total)
    rt.set(shift, Unavailable("quarantined"))
    (dirty,) = rt.drain_dirtied_watched()
    assert rt.status(dirty) == Unavailable("quarantined")
    assert rt.get_with_status(dirty) == (
        Unavailable("quarantined"),
        Unavailable("quarantined"),
    )


def shifts(rt):
    rows = rt.create_collection()
    rows.extend([("ann", 8), ("ann", 4), ("bob", 6)])
    return rows


def quarantine(rows, row):
    rows.replace(row, Unavailable("quarantined", row=row))


def test_count_excludes_and_counts_not_ready_elements():
    rt = Runtime()
    rows = shifts(rt)
    n = rows.count()
    assert rt.get_with_status(n) == (3, None)

    quarantine(rows, ("ann", 4))
    rows.insert(Pending(key="s9"))
    rows.insert(Incomplete(("bob", 1), 3))
    assert rt.get_with_status(n) == (2, Incomplete(2, 3))
    assert rt.get(n) == 2

    rows.delete(Pending(key="s9"))
    rows.delete(Incomplete(("bob", 1), 3))
    rows.replace(Unavailable("quarantined", row=("ann", 4)), ("ann", 4))
    assert rt.get_with_status(n) == (3, None)


def test_a_count_built_over_not_ready_elements_counts_them():
    rt = Runtime()
    rows = shifts(rt)
    quarantine(rows, ("bob", 6))
    assert rt.status(rows.count()) == Incomplete(2, 1)


def hours_total(rows):
    return rows.aggregate(0, lambda row: row[1], lambda a, b: a + b)


def test_aggregate_leaves_out_not_ready_elements_and_adds_incomplete_ones():
    rt = Runtime()
    rows = shifts(rt)
    total = hours_total(rows)
    assert rt.get_with_status(total) == (18, None)

    quarantine(rows, ("ann", 4))
    assert rt.get_with_status(total) == (14, Incomplete(14, 1))

    rows.insert(Incomplete(("cat", 5), 2))
    assert rt.status(total) == Incomplete(19, 3)

    rows.insert(Pending())
    assert rt.status(total) == Incomplete(19, 4)

    rows.delete(Pending())
    rows.delete(Incomplete(("cat", 5), 2))
    rows.replace(Unavailable("quarantined", row=("ann", 4)), ("ann", 4))
    assert rt.get_with_status(total) == (18, None)


def test_an_aggregate_built_over_not_ready_elements_counts_them():
    rt = Runtime()
    rows = shifts(rt)
    quarantine(rows, ("ann", 8))
    assert rt.status(hours_total(rows)) == Incomplete(10, 1)


def test_aggregate_callables_never_see_not_ready_values():
    rt = Runtime()
    rows = rt.create_collection()
    seen = []

    def lift(x):
        seen.append(x)
        return x

    def combine(a, b):
        seen.extend([a, b])
        return a + b

    total = rows.aggregate(0, lift, combine)
    rows.extend([1, 2, Unavailable("q", row=3), Incomplete(4, 1)])
    assert rt.status(total) == Incomplete(7, 2)
    assert all(isinstance(x, int) for x in seen)


def test_reduce_folds_ready_elements_and_counts_the_rest():
    rt = Runtime()
    rows = shifts(rt)
    folded = []

    def fold(elements):
        folded.append(list(elements))
        return sum(row[1] for row in elements)

    total = rows.reduce(fold)
    assert rt.get_with_status(total) == (18, None)

    quarantine(rows, ("bob", 6))
    rows.insert(Incomplete(("cat", 5), 2))
    assert rt.status(total) == Incomplete(17, 3)
    assert sorted(folded[-1]) == [("ann", 4), ("ann", 8), ("cat", 5)]


def test_a_two_level_chain_over_a_total_carries_its_missing():
    rt = Runtime()
    rows = shifts(rt)
    total = hours_total(rows)
    n = rows.count()
    rate = rt.create_input(20)
    pay = rt.create_query(lambda ctx: ctx.get(total) * ctx.get(rate))
    average = rt.create_query(lambda ctx: ctx.get(pay) / ctx.get(n))
    assert rt.get_with_status(average) == (120.0, None)

    quarantine(rows, ("ann", 4))
    assert rt.status(pay) == Incomplete(280, 1)
    assert rt.status(average) == Incomplete(140.0, 2)


def test_group_by_files_a_not_ready_element_under_its_row_key():
    rt = Runtime()
    rows = shifts(rt)
    by_worker = rows.group_by(lambda row: row[0])
    quarantine(rows, ("ann", 4))
    rt.get(by_worker.version_node)
    ann = hours_total(by_worker.get_group("ann"))
    bob = hours_total(by_worker.get_group("bob"))
    assert rt.status(ann) == Incomplete(8, 1)
    assert rt.get_with_status(bob) == (6, None)


def test_map_and_filter_apply_to_the_row_and_rewrap_it():
    rt = Runtime()
    rows = shifts(rt)
    quarantine(rows, ("ann", 4))
    rows.insert(Pending(key="s9"))
    rows.insert(Incomplete(("cat", 5), 2))

    hours = rows.map(lambda row: row[1])
    assert sorted(rt.get(hours.reduce(tuple))) == [5, 6, 8]
    assert rt.status(hours.reduce(lambda xs: sum(xs))) == Incomplete(19, 4)

    long = rows.filter(lambda row: row[1] >= 5)
    assert rt.status(long.count()) == Incomplete(2, 2)
    short = rows.filter(lambda row: row[1] < 5)
    assert rt.status(short.count()) == Incomplete(0, 2)


def test_a_keyed_operator_refuses_a_not_ready_element_without_a_row():
    rt = Runtime()
    rows = shifts(rt)
    rows.insert(Pending(key="s9"))
    with pytest.raises(TypeError, match="no row"):
        rows.group_by(lambda row: row[0])
    with pytest.raises(TypeError, match="no row"):
        rows.sort_by_key(lambda row: row[1])


def test_join_keys_on_the_row_and_makes_the_pair_not_ready():
    rt = Runtime()
    rows = shifts(rt)
    rates = rt.create_collection()
    rates.extend([("ann", 20), ("bob", 30)])
    quarantine(rows, ("bob", 6))
    pairs = rows.join(rates, lambda row: row[0], lambda rate: rate[0])
    pay = pairs.aggregate(0, lambda p: p[0][1] * p[1][1], lambda a, b: a + b)
    assert rt.status(pay) == Incomplete(240, 1)
    listed = rt.get(pairs.reduce(lambda xs: len(xs)))
    assert listed == 2


def test_sort_by_key_sorts_a_not_ready_element_by_its_row():
    rt = Runtime()
    rows = shifts(rt)
    quarantine(rows, ("ann", 4))
    ordered = rows.sort_by_key(lambda row: row[1])
    assert ordered.snapshot() == [
        Unavailable("quarantined", row=("ann", 4)),
        ("bob", 6),
        ("ann", 8),
    ]
