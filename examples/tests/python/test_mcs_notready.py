import gc
import weakref

import incr
import incr_concurrent
import pytest

PACKAGES = [
    pytest.param(incr, id="local"),
    pytest.param(incr_concurrent, id="concurrent"),
]


@pytest.fixture(params=PACKAGES)
def pkg(request):
    return request.param


@pytest.fixture
def local(pkg):
    if pkg is not incr:
        pytest.skip(
            "lookups and the not-ready rules of collection operators exist on incr only"
        )
    return pkg


def test_not_ready_types_share_a_base_and_incomplete_is_not_one(pkg):
    for value in (
        pkg.Pending(),
        pkg.LookupFailed("timeout"),
        pkg.Unavailable("quarantined"),
    ):
        assert isinstance(value, pkg.NotReady)
    assert not isinstance(pkg.Incomplete(1, 1), pkg.NotReady)


def test_fields_default_and_read_back(pkg):
    assert pkg.Pending().key is None
    assert pkg.Pending(key="k").key == "k"
    failed = pkg.LookupFailed("timeout", key=3)
    assert (failed.reason, failed.key) == ("timeout", 3)
    assert pkg.LookupFailed("timeout").key is None
    quarantined = pkg.Unavailable("quarantined", row=(1, "a"))
    assert (quarantined.reason, quarantined.row) == ("quarantined", (1, "a"))
    assert pkg.Unavailable("quarantined").row is None
    incomplete = pkg.Incomplete(10, 2)
    assert (incomplete.value, incomplete.missing) == (10, 2)


def test_equality_is_by_type_and_fields_and_equal_values_hash_alike(pkg):
    assert pkg.Pending(key=1) == pkg.Pending(key=1)
    assert pkg.Pending(key=1) != pkg.Pending(key=2)
    assert pkg.Pending() != pkg.LookupFailed("x")
    assert pkg.LookupFailed("x", key=1) == pkg.LookupFailed("x", key=1)
    assert pkg.LookupFailed("x", key=1) != pkg.LookupFailed("y", key=1)
    assert pkg.Unavailable("q", row=1) == pkg.Unavailable("q", row=1)
    assert pkg.Unavailable("q", row=1) != pkg.Unavailable("q")
    assert pkg.Incomplete(1, 2) == pkg.Incomplete(1, 2)
    assert pkg.Incomplete(1, 2) != pkg.Incomplete(1, 3)
    assert pkg.Incomplete(1, 2) != 1
    assert hash(pkg.Pending(key=1)) == hash(pkg.Pending(key=1))
    assert hash(pkg.Unavailable("q", row=(1, 2))) == hash(
        pkg.Unavailable("q", row=(1, 2))
    )
    assert hash(pkg.Incomplete(1, 2)) == hash(pkg.Incomplete(1, 2))
    assert len({pkg.Pending(key=1), pkg.Pending(key=1), pkg.Unavailable("q")}) == 2


def test_values_are_frozen_and_have_a_repr(pkg):
    with pytest.raises(AttributeError):
        pkg.Pending(key=1).key = 2
    with pytest.raises(AttributeError):
        pkg.Incomplete(1, 1).missing = 2
    assert repr(pkg.Unavailable("q", row=1)) == "Unavailable('q', row=1)"
    assert repr(pkg.Pending(key="k")) == "Pending(key='k')"
    assert repr(pkg.LookupFailed("t", key=1)) == "LookupFailed('t', key=1)"
    assert repr(pkg.Incomplete(5, 2)) == "Incomplete(5, missing=2)"


def test_incomplete_needs_something_missing(pkg):
    with pytest.raises(ValueError):
        pkg.Incomplete(1, 0)


def test_the_signal_is_not_an_exception(pkg):
    assert issubclass(pkg._NotReadySignal, BaseException)
    assert not issubclass(pkg._NotReadySignal, Exception)


def test_a_quarantined_input_makes_its_dependents_unavailable(pkg):
    rt = pkg.Runtime()
    shift = rt.create_input(8)
    pay = rt.create_query(lambda ctx: ctx.get(shift) * 20)
    total = rt.create_query(lambda ctx: ctx.get(pay) + 100)
    assert rt.get(total) == 260

    rt.set(shift, pkg.Unavailable("quarantined"))
    assert rt.get(pay) == pkg.Unavailable("quarantined")
    assert rt.get(total) == pkg.Unavailable("quarantined")

    rt.set(shift, 9)
    assert rt.get(total) == 280


def test_except_exception_cannot_swallow_a_not_ready_read(pkg):
    rt = pkg.Runtime()
    leg = rt.create_input(pkg.Pending(key="leg"))

    def score(ctx):
        try:
            return ctx.get(leg)
        except Exception:
            return "swallowed"

    q = rt.create_query(score)
    assert rt.get(q) == pkg.Pending(key="leg")


def test_try_get_follows_the_same_rule(pkg):
    rt = pkg.Runtime()
    leg = rt.create_input(pkg.LookupFailed("timeout", key="leg"))
    q = rt.create_query(lambda ctx: ctx.try_get(leg))
    assert rt.get(q) == pkg.LookupFailed("timeout", key="leg")


def test_a_python_exception_still_fails_the_node(pkg):
    rt = pkg.Runtime()
    x = rt.create_input(0)

    def boom(ctx):
        return 1 / ctx.get(x)

    q = rt.create_query(boom)
    with pytest.raises(ZeroDivisionError):
        rt.get(q)


def test_reading_an_incomplete_value_gives_its_value_and_carries_missing(pkg):
    rt = pkg.Runtime()
    total = rt.create_input(pkg.Incomplete(100, 2))
    doubled = rt.create_query(lambda ctx: ctx.get(total) * 2)
    assert rt.get_with_status(doubled) == (200, pkg.Incomplete(200, 2))
    assert rt.get(doubled) == 200


def test_missing_counts_contributions_not_distinct_rows(pkg):
    rt = pkg.Runtime()
    a = rt.create_input(pkg.Incomplete(1, 1))
    b = rt.create_input(pkg.Incomplete(2, 1))
    both = rt.create_query(lambda ctx: ctx.get(a) + ctx.get(a) + ctx.get(b))
    assert rt.status(both) == pkg.Incomplete(4, 3)


def test_a_nested_recompute_keeps_its_own_missing(pkg):
    rt = pkg.Runtime()
    source = rt.create_input(pkg.Incomplete(10, 2))
    plain = rt.create_input(5)
    child = rt.create_query(lambda ctx: ctx.get(source) + 1)
    parent = rt.create_query(lambda ctx: ctx.get(plain) + ctx.get(child) * 2)
    assert rt.get_with_status(parent) == (27, pkg.Incomplete(27, 2))
    assert rt.status(child) == pkg.Incomplete(11, 2)

    rt.set(plain, 6)
    assert rt.status(parent) == pkg.Incomplete(28, 2)
    rt.set(source, 10)
    assert rt.get_with_status(parent) == (28, None)


def test_a_returned_incomplete_adds_the_frame_missing(pkg):
    rt = pkg.Runtime()
    source = rt.create_input(pkg.Incomplete(10, 2))
    q = rt.create_query(lambda ctx: pkg.Incomplete(ctx.get(source), 1))
    assert rt.status(q) == pkg.Incomplete(10, 3)


def test_get_status_opts_out_of_both_rules(pkg):
    rt = pkg.Runtime()
    pending = rt.create_input(pkg.Pending(key="k"))
    partial = rt.create_input(pkg.Incomplete(7, 1))
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
            (pkg.Pending(key="k"), pkg.Pending(key="k")),
            (7, pkg.Incomplete(7, 1)),
            (3, None),
        ),
        None,
    )


def test_top_level_reads_and_status_agree(pkg):
    rt = pkg.Runtime()
    plain = rt.create_input(1)
    pending = rt.create_input(pkg.Pending())
    partial = rt.create_input(pkg.Incomplete(2, 1))
    assert rt.get(plain) == 1
    assert rt.get(pending) == pkg.Pending()
    assert rt.get(partial) == 2
    assert rt.try_get(partial) == 2
    assert rt.get_with_status(plain) == (1, None)
    assert rt.get_with_status(pending) == (pkg.Pending(), pkg.Pending())
    assert rt.get_with_status(partial) == (2, pkg.Incomplete(2, 1))
    assert rt.status(plain) is None
    assert rt.status(pending) == pkg.Pending()
    assert rt.status(partial) == pkg.Incomplete(2, 1)


def test_status_of_watched_outputs_after_a_drain(pkg):
    rt = pkg.Runtime()
    shift = rt.create_input(8)
    total = rt.create_query(lambda ctx: ctx.get(shift) + 1)
    rt.get(total)
    rt.watch(total)
    rt.set(shift, pkg.Unavailable("quarantined"))
    (dirty,) = rt.drain_dirtied_watched()
    assert rt.status(dirty) == pkg.Unavailable("quarantined")
    assert rt.get_with_status(dirty) == (
        pkg.Unavailable("quarantined"),
        pkg.Unavailable("quarantined"),
    )


def fill(rows, items):
    for item in items:
        rows.insert(item)


def swap(rows, old, new):
    rows.delete(old)
    rows.insert(new)


def shifts(rt):
    rows = rt.create_collection()
    fill(rows, [("ann", 8), ("ann", 4), ("bob", 6)])
    return rows


def quarantine(pkg, rows, row):
    swap(rows, row, pkg.Unavailable("quarantined", row=row))


def test_count_excludes_and_counts_not_ready_elements(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    n = rows.count()
    assert rt.get_with_status(n) == (3, None)

    quarantine(pkg, rows, ("ann", 4))
    rows.insert(pkg.Pending(key="s9"))
    rows.insert(pkg.Incomplete(("bob", 1), 3))
    assert rt.get_with_status(n) == (2, pkg.Incomplete(2, 3))
    assert rt.get(n) == 2

    rows.delete(pkg.Pending(key="s9"))
    rows.delete(pkg.Incomplete(("bob", 1), 3))
    swap(rows, pkg.Unavailable("quarantined", row=("ann", 4)), ("ann", 4))
    assert rt.get_with_status(n) == (3, None)


def test_a_count_built_over_not_ready_elements_counts_them(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    quarantine(pkg, rows, ("bob", 6))
    assert rt.status(rows.count()) == pkg.Incomplete(2, 1)


def hours_total(rows):
    return rows.aggregate(0, lambda row: row[1], lambda a, b: a + b)


def test_aggregate_leaves_out_not_ready_elements_and_adds_incomplete_ones(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    total = hours_total(rows)
    assert rt.get_with_status(total) == (18, None)

    quarantine(pkg, rows, ("ann", 4))
    assert rt.get_with_status(total) == (14, pkg.Incomplete(14, 1))

    rows.insert(pkg.Incomplete(("cat", 5), 2))
    assert rt.status(total) == pkg.Incomplete(19, 3)

    rows.insert(pkg.Pending())
    assert rt.status(total) == pkg.Incomplete(19, 4)

    rows.delete(pkg.Pending())
    rows.delete(pkg.Incomplete(("cat", 5), 2))
    swap(rows, pkg.Unavailable("quarantined", row=("ann", 4)), ("ann", 4))
    assert rt.get_with_status(total) == (18, None)


def test_an_aggregate_built_over_not_ready_elements_counts_them(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    quarantine(pkg, rows, ("ann", 8))
    assert rt.status(hours_total(rows)) == pkg.Incomplete(10, 1)


def test_aggregate_callables_never_see_not_ready_values(pkg, local):
    rt = pkg.Runtime()
    rows = rt.create_collection()
    seen = []

    def lift(x):
        seen.append(x)
        return x

    def combine(a, b):
        seen.extend([a, b])
        return a + b

    total = rows.aggregate(0, lift, combine)
    fill(rows, [1, 2, pkg.Unavailable("q", row=3), pkg.Incomplete(4, 1)])
    assert rt.status(total) == pkg.Incomplete(7, 2)
    assert all(isinstance(x, int) for x in seen)


def test_reduce_folds_ready_elements_and_counts_the_rest(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    folded = []

    def fold(elements):
        folded.append(list(elements))
        return sum(row[1] for row in elements)

    total = rows.reduce(fold)
    assert rt.get_with_status(total) == (18, None)

    quarantine(pkg, rows, ("bob", 6))
    rows.insert(pkg.Incomplete(("cat", 5), 2))
    assert rt.status(total) == pkg.Incomplete(17, 3)
    assert sorted(folded[-1]) == [("ann", 4), ("ann", 8), ("cat", 5)]


def test_a_two_level_chain_over_a_total_carries_its_missing(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    total = hours_total(rows)
    n = rows.count()
    rate = rt.create_input(20)
    pay = rt.create_query(lambda ctx: ctx.get(total) * ctx.get(rate))
    average = rt.create_query(lambda ctx: ctx.get(pay) / ctx.get(n))
    assert rt.get_with_status(average) == (120.0, None)

    quarantine(pkg, rows, ("ann", 4))
    assert rt.status(pay) == pkg.Incomplete(280, 1)
    assert rt.status(average) == pkg.Incomplete(140.0, 2)


def test_group_by_files_a_not_ready_element_under_its_row_key(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    by_worker = rows.group_by(lambda row: row[0])
    quarantine(pkg, rows, ("ann", 4))
    rt.get(by_worker.version_node)
    ann = hours_total(by_worker.get_group("ann"))
    bob = hours_total(by_worker.get_group("bob"))
    assert rt.status(ann) == pkg.Incomplete(8, 1)
    assert rt.get_with_status(bob) == (6, None)


def test_map_and_filter_apply_to_the_row_and_rewrap_it(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    quarantine(pkg, rows, ("ann", 4))
    rows.insert(pkg.Pending(key="s9"))
    rows.insert(pkg.Incomplete(("cat", 5), 2))

    hours = rows.map(lambda row: row[1])
    assert sorted(rt.get(hours.reduce(tuple))) == [5, 6, 8]
    assert rt.status(hours.reduce(lambda xs: sum(xs))) == pkg.Incomplete(19, 4)

    long = rows.filter(lambda row: row[1] >= 5)
    assert rt.status(long.count()) == pkg.Incomplete(2, 2)
    short = rows.filter(lambda row: row[1] < 5)
    assert rt.status(short.count()) == pkg.Incomplete(0, 2)


def test_a_keyed_operator_refuses_a_not_ready_element_without_a_row(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    rows.insert(pkg.Pending(key="s9"))
    with pytest.raises(TypeError, match="no row"):
        rows.group_by(lambda row: row[0])
    with pytest.raises(TypeError, match="no row"):
        rows.sort_by_key(lambda row: row[1])


def test_join_keys_on_the_row_and_makes_the_pair_not_ready(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    rates = rt.create_collection()
    fill(rates, [("ann", 20), ("bob", 30)])
    quarantine(pkg, rows, ("bob", 6))
    pairs = rows.join(rates, lambda row: row[0], lambda rate: rate[0])
    pay = pairs.aggregate(0, lambda p: p[0][1] * p[1][1], lambda a, b: a + b)
    assert rt.status(pay) == pkg.Incomplete(240, 1)
    listed = rt.get(pairs.reduce(lambda xs: len(xs)))
    assert listed == 2


def test_sort_by_key_sorts_a_not_ready_element_by_its_row(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    quarantine(pkg, rows, ("ann", 4))
    ordered = rows.sort_by_key(lambda row: row[1])
    assert ordered.snapshot() == [
        pkg.Unavailable("quarantined", row=("ann", 4)),
        ("bob", 6),
        ("ann", 8),
    ]


def test_a_lookup_starts_pending_and_is_drained_once(pkg, local):
    rt = pkg.Runtime()
    leg = rt.create_lookup(("ann", "bob"))
    assert rt.get(leg) == pkg.Pending(key=("ann", "bob"))
    assert rt.create_lookup(("ann", "bob")) == leg
    assert rt.drain_pending_lookups() == [("ann", "bob")]
    assert rt.drain_pending_lookups() == []


def test_a_query_reading_a_lookup_is_pending_until_it_resolves(pkg, local):
    rt = pkg.Runtime()
    rate = rt.create_input(2)
    score = rt.create_query(lambda ctx: ctx.lookup("leg") * ctx.get(rate))
    assert rt.get(score) == pkg.Pending(key="leg")
    assert rt.drain_pending_lookups() == ["leg"]

    rt.resolve_lookups({"leg": 15})
    assert rt.get_with_status(score) == (30, None)
    assert rt.drain_pending_lookups() == []


def test_a_lookup_resolved_before_the_drain_is_not_drained(pkg, local):
    rt = pkg.Runtime()
    rt.create_lookup("a")
    rt.create_lookup("b")
    rt.resolve_lookups({"a": 1})
    assert rt.drain_pending_lookups() == ["b"]


def test_resolving_an_unknown_key_creates_its_lookup_resolved(pkg, local):
    rt = pkg.Runtime()
    rt.resolve_lookups({"c": 3})
    assert rt.get(rt.create_lookup("c")) == 3
    assert rt.drain_pending_lookups() == []


def test_a_failed_lookup_is_a_value_and_retry_makes_it_pending_again(pkg, local):
    rt = pkg.Runtime()
    score = rt.create_query(lambda ctx: ctx.lookup("leg") + 1)
    rt.get(score)
    assert rt.drain_pending_lookups() == ["leg"]

    rt.fail_lookup("leg", "timeout")
    assert rt.get(score) == pkg.LookupFailed("timeout", key="leg")

    rt.retry_lookups(["leg"])
    assert rt.get(score) == pkg.Pending(key="leg")
    assert rt.drain_pending_lookups() == ["leg"]
    assert rt.drain_pending_lookups() == []

    rt.resolve_lookups([("leg", 4)])
    assert rt.get(score) == 5


def test_a_lookup_from_a_compute_survives_its_creator_dropping_it(pkg, local):
    rt = pkg.Runtime()
    use_leg = rt.create_input(True)
    score = rt.create_query(lambda ctx: ctx.lookup("leg") if ctx.get(use_leg) else 0)
    rt.get(score)
    rt.set(use_leg, False)
    assert rt.get(score) == 0
    rt.resolve_lookups({"leg": 7})
    assert rt.get(rt.create_lookup("leg")) == 7


def test_lookups_raise_on_a_closed_runtime(pkg, local):
    rt = pkg.Runtime()
    rt.create_lookup("a")
    rt.close()
    for call in (
        lambda: rt.create_lookup("a"),
        rt.drain_pending_lookups,
        lambda: rt.resolve_lookups({"a": 1}),
        lambda: rt.fail_lookup("a", "x"),
        lambda: rt.retry_lookups(["a"]),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_closing_releases_lookup_keys(pkg, local):
    class Key:
        pass

    rt = pkg.Runtime()
    key = Key()
    rt.create_lookup(key)
    ref = weakref.ref(key)
    del key
    gc.collect()
    assert ref() is not None
    rt.close()
    gc.collect()
    assert ref() is None


def test_incomplete_refuses_a_value_that_is_itself_not_ready(pkg):
    with pytest.raises(TypeError):
        pkg.Incomplete(pkg.Pending(), 1)
    with pytest.raises(TypeError):
        pkg.Incomplete(pkg.Incomplete(1, 1), 1)


def test_get_traced_unwraps_an_incomplete_value_as_get_does(pkg):
    rt = pkg.Runtime()
    total = rt.create_input(pkg.Incomplete(4, 1))
    doubled = rt.create_query(lambda ctx: ctx.get(total) * 2)
    value, _ = rt.get_traced(doubled)
    assert value == 8


def test_a_creator_cut_short_by_a_pending_lookup_keeps_its_children(pkg, local):
    rt = pkg.Runtime()
    use_leg = rt.create_input(False)

    def creator(ctx):
        rate = ctx.lookup("leg") if ctx.get(use_leg) else 1
        return ctx.child_input("hours", lambda: 0), rate

    node = rt.create_query(creator)
    child, _ = rt.get(node)
    rt.set(child, 8)

    rt.set(use_leg, True)
    assert rt.get(node) == pkg.Pending(key="leg")
    assert rt.get(child) == 8

    rt.resolve_lookups({"leg": 3})
    again, rate = rt.get(node)
    assert again == child and rate == 3
    assert rt.get(again) == 8


def gap_total(outputs):
    return outputs.aggregate(0, lambda w: w[-1][1] - w[0][1], lambda a, b: a + b)


@pytest.mark.parametrize(
    "outputs",
    [lambda ordered: ordered.pairwise(), lambda ordered: ordered.window(2)],
    ids=["pairwise", "window"],
)
def test_an_output_holding_a_not_ready_element_is_unavailable(outputs, pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    gaps = outputs(rows.sort_by_key(lambda row: row[1]))
    total = gap_total(gaps)
    assert rt.get_with_status(total) == (4, None)

    quarantine(pkg, rows, ("ann", 4))
    assert rt.status(total) == pkg.Incomplete(2, 1)
    held = pkg.Unavailable("quarantined", row=("ann", 4))
    unavailable = [
        o
        for o in gaps.sort_by_key(repr).snapshot()
        if isinstance(o, pkg.Unavailable) and o.row[0] == held
    ]
    assert len(unavailable) == 1
    assert unavailable[0].reason == "contains a not-ready element"

    quarantine(pkg, rows, ("bob", 6))
    assert rt.status(total) == pkg.Incomplete(0, 2)

    swap(rows, pkg.Unavailable("quarantined", row=("bob", 6)), ("bob", 6))
    swap(rows, held, ("ann", 4))
    assert rt.get_with_status(total) == (4, None)


def test_an_incomplete_element_makes_its_window_unavailable(pkg, local):
    rt = pkg.Runtime()
    rows = shifts(rt)
    gaps = rows.sort_by_key(lambda row: row[1]).window(2)
    total = gap_total(gaps)
    swap(rows, ("bob", 6), pkg.Incomplete(("bob", 6), 3))
    assert rt.status(total) == pkg.Incomplete(0, 2)


def test_the_signal_lives_under_the_package_module(pkg):
    assert pkg._NotReadySignal.__module__ == pkg.__name__
