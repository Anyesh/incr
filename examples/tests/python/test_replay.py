import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)

DEPTH = 3000


def chain(rt, replay):
    nodes = [rt.create_input(0)]
    for _ in range(DEPTH):
        below = nodes[-1]
        nodes.append(
            rt.create_query(lambda r, below=below: r.get(below) + 1, replay=replay)
        )
    return nodes


@BOTH
def test_a_replayable_chain_deeper_than_python_recursion_computes(module):
    rt = module.Runtime()
    rt.set_placeholder(0)
    rt.set_max_frames(32)
    nodes = chain(rt, replay=True)
    assert rt.get(nodes[-1]) == DEPTH
    rt.set(nodes[0], 5)
    assert rt.get(nodes[-1]) == DEPTH + 5


@BOTH
def test_a_byte_budget_also_replays(module):
    rt = module.Runtime()
    rt.set_placeholder(0)
    rt.set_max_stack_bytes(256 * 1024)
    nodes = chain(rt, replay=True)
    assert rt.get(nodes[-1]) == DEPTH


@BOTH
def test_placeholder_is_set_once(module):
    rt = module.Runtime()
    rt.set_placeholder(0)
    with pytest.raises(RuntimeError, match="placeholder already set"):
        rt.set_placeholder(1)


@BOTH
def test_frame_budget_must_be_positive(module):
    rt = module.Runtime()
    with pytest.raises(RuntimeError):
        rt.set_max_frames(0)
