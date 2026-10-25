import threading
import weakref

import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)

PLACEHOLDER = 0


def doom_ready(module):
    """A runtime whose replayable queries replay any read below their own
    frame: the first run of such a query is doomed at its first read of a
    node that is not Clean, and reads the placeholder there."""
    rt = module.Runtime()
    rt.set_placeholder(PLACEHOLDER)
    rt.set_max_frames(1)
    return rt


@BOTH
def test_a_swallowed_stash_never_labels_a_later_rust_panic(module):
    rt = doom_ready(module)
    runs = []

    def raises(r):
        runs.append("s")
        raise ZeroDivisionError("from a speculative entry")

    a = rt.create_query(lambda r: 1, replay=True)
    s = rt.create_query(raises)
    gone = rt.create_input(0)
    rt.delete_node(gone)

    # Doomed, `top` reads `s`, which the driver runs as a speculative entry
    # and whose exception it swallows; the retry reads a deleted node.
    def top(r):
        return r.get(s) if r.get(a) == PLACEHOLDER else r.get(gone)

    t = rt.create_query(top, replay=True)
    with pytest.raises(RuntimeError, match="stale"):
        rt.get(t)
    assert runs, "the speculative entry never ran"


@BOTH
def test_an_exception_in_a_doomed_frame_is_dropped(module):
    rt = doom_ready(module)
    a = rt.create_query(lambda r: 2, replay=True)
    t = rt.create_query(lambda r: 10 // r.get(a), replay=True)
    assert rt.get(t) == 5


class Payload:
    pass


@BOTH
def test_a_doomed_frames_exception_is_dropped_before_the_retry(module):
    rt = doom_ready(module)
    a = rt.create_query(lambda r: 2, replay=True)
    refs, alive = [], []

    def payload():
        p = Payload()
        refs.append(weakref.ref(p))
        return p

    def top(r):
        x = r.get(a)
        if x == PLACEHOLDER:
            raise ZeroDivisionError(payload())
        alive.append(refs[0]() is not None)
        return x

    t = rt.create_query(top, replay=True)
    assert rt.get(t) == 2
    assert alive == [False], "the doomed run's exception was kept"


@BOTH
def test_a_later_exception_after_a_doomed_discard_is_raised_as_itself(module):
    rt = doom_ready(module)
    a = rt.create_query(lambda r: 2, replay=True)

    def bad(r):
        raise ValueError("later")

    b = rt.create_query(bad)
    t = rt.create_query(lambda r: (10 // r.get(a)) + r.get(b), replay=True)
    with pytest.raises(ValueError, match="later"):
        rt.get(t)


@BOTH
def test_r3_through_a_closure_lands_the_fallback_with_empty_stderr(module, capfd):
    rt = module.Runtime()
    nodes = {}
    nodes["c"] = rt.create_query(
        lambda r: r.get(nodes["d"]) + 1, fallback=lambda ids: -1
    )
    nodes["d"] = rt.create_query(lambda r: r.get(nodes["c"]) + 1)
    assert rt.get(nodes["c"]) == -1
    assert capfd.readouterr().err == ""


def test_a_wound_inside_a_python_closure_retries():
    rt = incr_concurrent.Runtime()
    nodes = {}
    both_claimed = threading.Barrier(2, timeout=10)
    first = {"a": True, "b": True}
    seen = []

    def member(me, other):
        def body(r):
            if first[me]:
                first[me] = False
                both_claimed.wait()
            try:
                return r.get(nodes[other]) + 1
            except BaseException as e:
                seen.append(type(e).__name__)
                raise

        return body

    nodes["a"] = rt.create_query(member("a", "b"), fallback=lambda ids: -1)
    nodes["b"] = rt.create_query(member("b", "a"), fallback=lambda ids: -1)
    results = {}

    def read(name):
        results[name] = rt.get(nodes[name])

    threads = [threading.Thread(target=read, args=(n,), daemon=True) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "the wound did not resolve"
    assert results == {"a": -1, "b": -1}
    assert "_IncrEngineUnwind" in seen


def cleanup_query(rt, name, raises, runs):
    def body(r):
        runs.append(name)
        if raises:
            raise ValueError(f"{name} failed")
        return 7

    return rt.create_query(body)


def assert_discarded_then_lands_its_own_outcome(rt, node, name, raises, runs):
    """`node` ran once in an engine unwind's finally and was discarded with
    that run, so the next read runs it again and gets its own outcome."""
    assert runs == [name]
    if raises:
        with pytest.raises(ValueError, match=f"{name} failed"):
            rt.get(node)
    else:
        assert rt.get(node) == 7
    assert runs == [name, name]


def assert_given_up_unrun_then_lands_its_own_outcome(rt, node, name, raises, runs):
    """`node`'s read in a discard's finally raised the discard again before
    running it, so the next read runs it once and gets its own outcome."""
    assert runs == []
    if raises:
        with pytest.raises(ValueError, match=f"{name} failed"):
            rt.get(node)
    else:
        assert rt.get(node) == 7
    assert runs == [name]


@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
@pytest.mark.parametrize("cleanup_read", ["get", "try_get"])
def test_a_read_in_finally_during_a_wound_is_given_up_with_the_run(
    raises, cleanup_read, capfd
):
    rt = incr_concurrent.Runtime()
    nodes = {}
    both_claimed = threading.Barrier(2, timeout=10)
    first = {"a": True, "b": True}
    runs = {"a": [], "b": []}
    seen = {"a": [], "b": []}
    cleanup = {me: cleanup_query(rt, me, raises, runs[me]) for me in "ab"}

    # Each first run claims its node, then reads the other's: the waits-for
    # cycle wounds the younger thread, whose finally reads its cleanup node
    # while the wound is carried. A later run reads nothing.
    def member(me, other):
        def body(r):
            if not first[me]:
                return 1
            first[me] = False
            both_claimed.wait()
            unwound = False
            try:
                return r.get(nodes[other]) + 1
            except BaseException:
                unwound = True
                raise
            finally:
                if unwound:
                    try:
                        seen[me].append(getattr(r, cleanup_read)(cleanup[me]))
                    except BaseException as e:
                        seen[me].append(type(e).__name__)
                        raise

        return body

    nodes["a"] = rt.create_query(member("a", "b"))
    nodes["b"] = rt.create_query(member("b", "a"))
    results = {}

    def read(name):
        results[name] = rt.get(nodes[name])

    threads = [threading.Thread(target=read, args=(n,), daemon=True) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "the wound did not resolve"

    wounded = [me for me in "ab" if seen[me]]
    assert len(wounded) == 1, seen
    w = wounded[0]
    o = "b" if w == "a" else "a"
    # The older thread ran the wounded node's retry, which reads nothing.
    assert results == {w: 1, o: 2}
    assert seen[w] == ["_IncrEngineUnwind"]
    assert runs[o] == []
    assert_discarded_then_lands_its_own_outcome(rt, cleanup[w], w, raises, runs[w])
    assert capfd.readouterr().err == ""


@BOTH
@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
@pytest.mark.parametrize("cleanup_read", ["get", "try_get"])
@pytest.mark.parametrize("top_read", ["get", "try_get"])
def test_a_read_in_finally_during_a_discard_is_given_up_with_the_run(
    module, raises, cleanup_read, top_read, capfd
):
    rt = doom_ready(module)
    runs = []
    seen = []
    inner_runs = []
    c = cleanup_query(rt, "c", raises, runs)

    def fails(r):
        inner_runs.append("i")
        raise ZeroDivisionError("speculative")

    inner = rt.create_query(fails)

    # `spec` runs as a speculative entry, so `inner`'s exception goes on as
    # a discard unwind that the finally's read of `c` runs under.
    def spec(r):
        try:
            return r.get(inner)
        finally:
            try:
                seen.append(getattr(r, cleanup_read)(c))
            except BaseException as e:
                seen.append(type(e).__name__)
                raise

    s = rt.create_query(spec)
    a = rt.create_query(lambda r: 1, replay=True)
    t = rt.create_query(
        lambda r: r.get(s) if r.get(a) == PLACEHOLDER else 5, replay=True
    )
    assert getattr(rt, top_read)(t) == 5
    assert inner_runs == ["i"], "the speculative entry never ran"
    assert seen == ["_IncrEngineUnwind"]
    assert_given_up_unrun_then_lands_its_own_outcome(rt, c, "c", raises, runs)
    # Discarded too, never Failed: the real read raises its own exception.
    with pytest.raises(ZeroDivisionError, match="speculative"):
        rt.get(inner)
    assert capfd.readouterr().err == ""


@BOTH
def test_a_speculative_runs_exception_is_raised_by_the_real_read_then_dropped(module):
    rt = doom_ready(module)
    refs = []

    def raises(r):
        p = Payload()
        refs.append(weakref.ref(p))
        raise ZeroDivisionError(p)

    a = rt.create_query(lambda r: 1, replay=True)
    s = rt.create_query(raises)
    t = rt.create_query(
        lambda r: r.get(s) if r.get(a) == PLACEHOLDER else 0, replay=True
    )
    # The doomed run read `s` speculatively; the engine keeps that run's
    # outcome for `s`'s real read, which raises it without running `s`.
    assert rt.get(t) == 0
    with pytest.raises(ZeroDivisionError):
        rt.get(s)
    assert len(refs) == 1
    assert refs[0]() is None


@BOTH
def test_an_exception_the_engine_drops_unraised_is_freed_by_that_call(module):
    rt = doom_ready(module)
    refs = []
    x = rt.create_input(0)

    def raises(r):
        p = Payload()
        refs.append(weakref.ref(p))
        raise ZeroDivisionError(p)

    a = rt.create_query(lambda r: 1, replay=True)
    s = rt.create_query(raises)
    t = rt.create_query(
        lambda r: r.get(s) if r.get(a) == PLACEHOLDER else 0, replay=True
    )
    assert rt.get(t) == 0
    assert len(refs) == 1 and refs[0]() is not None, "the outcome is kept for s"
    # A set ends the revision, which drops the outcome unraised.
    rt.set(x, 1)
    assert refs[0]() is None


@BOTH
def test_another_runtimes_callback_inside_a_doomed_query_raises_as_itself(module):
    other = module.Runtime()
    col = other.create_collection()
    col.insert(1)

    def bad(x):
        raise ValueError("from the other runtime")

    rt = doom_ready(module)
    a = rt.create_query(lambda r: 2, replay=True)
    seen = []

    # `map` runs `bad` on the existing element in the other runtime's own
    # call, which pushes no frame over the doomed one.
    def top(r):
        if r.get(a) == PLACEHOLDER:
            try:
                col.map(bad)
            except Exception as e:
                seen.append(type(e).__name__)
        return 0

    t = rt.create_query(top, replay=True)
    assert rt.get(t) == 0
    assert seen == ["ValueError"]


@BOTH
def test_a_doomed_querys_read_of_another_runtime_discards_its_error(module):
    other = module.Runtime()
    col = other.create_collection()

    def bad(x):
        raise ValueError("from the other runtime")

    count = col.map(bad).count()
    col.insert(1)
    rt = doom_ready(module)
    a = rt.create_query(lambda r: 2, replay=True)
    seen = []

    # The doomed run reached the read only through its placeholder, so the
    # read is speculative: its error is discarded, not raised.
    def top(r):
        if r.get(a) == PLACEHOLDER:
            try:
                other.get(count)
            except Exception as e:
                seen.append(type(e).__name__)
        return 0

    t = rt.create_query(top, replay=True)
    assert rt.get(t) == 0
    assert seen == []
    with pytest.raises(ValueError, match="from the other runtime"):
        other.get(count)
