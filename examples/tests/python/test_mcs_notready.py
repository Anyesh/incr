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
