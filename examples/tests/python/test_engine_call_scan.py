import pathlib
import re

import pytest

CRATES = pathlib.Path(__file__).resolve().parents[3] / "crates"
SOURCES = [
    CRATES / "incr-python" / "src" / "lib.rs",
    CRATES / "incr-concurrent-python" / "src" / "lib.rs",
]

# Method calls on an engine runtime, collection or node handle, and calls
# that pass the runtime as an operator's first argument.
ENGINE_CALL = re.compile(
    r"\b\w+\.inner\.(\w+)\(|\brt\.(\w+)\(|\bself\.rt\.(\w+)\(|\.(\w+)\(\s*&?(?:self\.)?rt\w*\b"
)
# Handle accessors that take no lock, and PyO3's own methods on `Py<_>`.
ALLOWED = {"slot", "hash", "bind", "clone_ref", "borrow", "borrow_mut"}


def matching(text, open_at, opening, closing):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise AssertionError(f"unbalanced {opening} at {open_at}")


def blocks(text, pattern):
    for m in re.finditer(pattern, text):
        start = text.index("{", m.end())
        yield m.start(), start, matching(text, start, "{", "}")


def engine_calls(text, start, end, names=None):
    for m in ENGINE_CALL.finditer(text, start, end):
        name = next(g for g in m.groups() if g)
        if name not in ALLOWED:
            yield m.start(), name
    for name in names or ():
        for m in re.finditer(rf"\.{name}\(", text[start:end]):
            yield start + m.start(), name


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: p.parent.parent.name)
def test_every_engine_call_in_a_binding_method_goes_through_engine_call(source):
    text = source.read_text()
    methods = list(blocks(text, r"#\[pymethods\]\s*impl\b"))
    assert methods, "no #[pymethods] blocks found"

    def in_methods(pos):
        return any(s <= pos <= e for s, _, e in methods)

    # A helper outside #[pymethods] that calls the engine is itself an
    # engine call for the methods that call it.
    helpers = set()
    for s, body, e in blocks(text, r"\bfn (\w+)\b[^;{]*"):
        name = re.match(r"fn (\w+)", text[s:]).group(1)
        if name in {"engine_call", "pyerr_boundary"} or in_methods(s):
            continue
        if any(True for _ in engine_calls(text, body, e)):
            helpers.add(name)

    spans = []
    for m in re.finditer(r"\bengine_call\(", text):
        if in_methods(m.start()):
            spans.append((m.end() - 1, matching(text, m.end() - 1, "(", ")")))

    outside = [
        f"line {line_of(text, pos)}: {name}"
        for _, body, e in methods
        for pos, name in engine_calls(text, body, e, helpers)
        if not any(s <= pos <= e2 for s, e2 in spans)
    ]
    assert not outside, "engine calls outside engine_call:\n" + "\n".join(outside)
