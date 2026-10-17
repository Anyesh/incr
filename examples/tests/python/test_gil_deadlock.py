import subprocess
import sys
import textwrap

import pytest

# Each reproduction runs in a child killed after 10 s, because a thread
# that holds the GIL inside native code cannot be stopped from inside the
# process. The main thread holds the GIL in a binding method that does not
# detach and waits on an engine lock, while the other thread holds that
# lock across a PyValue method that waits for the GIL.

ARENA_CHILD = textwrap.dedent(
    """
    import threading
    import time

    from incr_concurrent import Runtime

    rt = Runtime()
    a = rt.create_input(object())
    stop = threading.Event()


    def reader():
        while not stop.is_set():
            rt.get(a)


    t = threading.Thread(target=reader)
    t.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        rt.delete_node(rt.create_input(0))
    stop.set()
    t.join()
    """
)

COLLECTION_CHILD = textwrap.dedent(
    """
    import threading
    import time

    from incr_concurrent import Runtime

    rt = Runtime()
    col = rt.create_collection()
    count = col.filter(lambda x: x % 2 == 0).count()
    grouped = col.group_by(lambda x: x % 4)
    groups = grouped.version_node
    for i in range(8):
        col.insert(i)
    stop = threading.Event()


    def writer():
        i = 0
        while not stop.is_set():
            col.insert(i % 8)
            rt.get(count)
            rt.get(groups)
            col.delete(i % 8)
            i += 1


    t = threading.Thread(target=writer)
    t.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        col.snapshot_len()
        grouped.get_group(1)
        col.filter(bool)
    stop.set()
    t.join()
    """
)


def run_child(source):
    """The child's exit code, or None if it was killed after 10 s.

    A child that exits non-zero fails the test outright rather than raising
    AssertionError, so a broken reproduction never counts as the expected
    deadlock.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, timeout=10
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        pytest.fail(f"child exited with {proc.returncode}:\n{proc.stderr}")
    return proc.returncode


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug (f), arena form: a Clean get holds the arena read lock across "
    "PyValue clone, which waits for the GIL held by create_input waiting on the "
    "arena write lock",
)
def test_get_racing_create_input_does_not_deadlock_on_the_gil():
    assert run_child(ARENA_CHILD) is not None, "child deadlocked, killed after 10 s"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="bug (f), collection form: insert and operator reads hold collection "
    "locks across PyValue clone and eq, which wait for the GIL held by "
    "snapshot_len, get_group or filter waiting on those locks",
)
def test_collection_reads_racing_inserts_do_not_deadlock_on_the_gil():
    assert run_child(COLLECTION_CHILD) is not None, (
        "child deadlocked, killed after 10 s"
    )
