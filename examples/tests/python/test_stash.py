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


@BOTH
def test_a_swallowed_exception_is_dropped_when_the_call_returns(module):
    rt = doom_ready(module)
    refs = []

    def raises(r):
        p = Payload()
        refs.append(weakref.ref(p))
        raise ZeroDivisionError(p)

    a = rt.create_query(lambda r: 1, replay=True)
    s = rt.create_query(raises)
    t = rt.create_query(lambda r: r.get(s) if r.get(a) == PLACEHOLDER else 0, replay=True)
    assert rt.get(t) == 0
    assert refs and all(ref() is None for ref in refs)


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
