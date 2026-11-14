import pathlib
import re

import pytest

CRATES = pathlib.Path(__file__).resolve().parents[3] / "crates"
LOCAL = CRATES / "incr-python" / "src"
CONCURRENT = CRATES / "incr-concurrent-python" / "src" / "lib.rs"
SHARED = CRATES / "incr-python-persist" / "src"

# Method calls on an engine runtime, collection or node handle, and calls
# that pass the runtime as an operator's first argument. Only the last
# kind can name one of the binding's own helpers. Store calls go through
# incr-python-persist, whose own scan below checks they enter the engine
# through the binding.
ENGINE_CALL = re.compile(
    r"\b\w+\.inner\.(\w+)\(|\brt\.(\w+)\(|\bself\.rt\.(\w+)\(|\.(\w+)\(\s*&?(?:self\.)?rt\w*\b"
    r"|\bcore\.(run|write|declare_keyed)\("
)
# Handle accessors that take no lock, and PyO3's own methods on `Py<_>`.
ALLOWED = {"slot", "hash", "bind", "clone_ref", "borrow", "borrow_mut"}
# A routing hold only bumps a counter and runs no user code, and neither
# does the sweep hold or taking the creators it kept (they only mark nodes
# to rerun). The last three are methods of a backend object that
# `.inner` holds in the persist backends, not of the engine.
ALLOWED_LOCAL = ALLOWED | {
    "hold_routing",
    "hold_routing_suppressed",
    "hold_child_sweeps",
    "take_held_sweep_creators",
    "read_only",
    "writable",
    "load",
}
# Core calls that write: inside one, a reached `group_by` router is queued.
MUTATING = {
    "set",
    "set_returning_displaced",
    "insert",
    "delete",
    "replace",
    "apply",
    "extend",
    "remove",
    "delete_node",
    "try_delete_node",
    "delete_with_dependents",
    "run",
    "write",
    "declare_keyed",
    "checkpoint",
    "upsert",
    "upsert_journaled",
    "remove_journaled",
    "reconcile",
    "reconcile_journaled",
    "apply_pairs_with",
}
ENTRIES = {
    "engine_read": "engine_read",
    "engine_in_query": "engine_read",
    "engine_write": "engine_write",
    "read_on": "engine_read",
    "write_on": "engine_write",
}


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


def calls(text, start, end, allowed):
    for m in ENGINE_CALL.finditer(text, start, end):
        name = next(g for g in m.groups() if g)
        if name not in allowed:
            yield m.start(), name, m.group(4) is not None


def engine_calls(text, start, end, names=None):
    for pos, name, _ in calls(text, start, end, ALLOWED):
        yield pos, name
    for name in names or ():
        for m in re.finditer(rf"\.{name}\(", text[start:end]):
            yield start + m.start(), name


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def method_blocks(text):
    return list(blocks(text, r"#\[pymethods\]\s*impl\b"))


def test_every_concurrent_engine_call_in_a_binding_method_goes_through_engine_call():
    text = CONCURRENT.read_text()
    methods = method_blocks(text)
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


class Helpers:
    """The functions outside `#[pymethods]` of the scanned sources, by name
    (a name several functions share keeps every body). A helper reaches the
    engine only through a runtime passed to it, so only calls that pass one
    are followed."""

    def __init__(self, texts):
        self.bodies = {}
        for text in texts:
            methods = method_blocks(text)
            for s, body, e in blocks(text, r"\bfn (\w+)\b[^;{]*"):
                name = re.match(r"fn (\w+)", text[s:]).group(1)
                if name in {*ENTRIES, "pyerr_boundary"}:
                    continue
                if not any(ms <= s <= me for ms, _, me in methods):
                    self.bodies.setdefault(name, []).append((text, body, e))

    def reached(self, name, may_be_helper, seen=None):
        """`name` and the call names its bodies reach, when it can be a
        helper."""
        names = {name}
        seen = set() if seen is None else seen
        if not may_be_helper or name in seen:
            return names
        seen.add(name)
        for text, body, e in self.bodies.get(name, ()):
            for _, called, helper in calls(text, body, e, ALLOWED_LOCAL):
                names |= self.reached(called, helper, seen)
        return names

    def calls_in(self, text, start, end):
        for pos, name, helper in calls(text, start, end, ALLOWED_LOCAL):
            yield pos, self.reached(name, helper)


def local_violations(text, helper_texts=()):
    """The breaches of incr-python's rule in the `#[pymethods]` of `text`:
    every engine call sits in an `engine_write` or `engine_read` span, an
    `engine_read` span reaches no core write, and an `engine_write` span
    reaches at least one."""
    methods = method_blocks(text)
    helpers = Helpers((text, *helper_texts))
    spans = []
    for entry in ENTRIES:
        for m in re.finditer(rf"\b{entry}\(", text):
            if any(s <= m.start() <= e for s, _, e in methods):
                open_at = m.end() - 1
                spans.append(
                    (ENTRIES[entry], open_at, matching(text, open_at, "(", ")"))
                )

    problems = []
    for _, body, e in methods:
        for pos, names in helpers.calls_in(text, body, e):
            if not any(s <= pos <= e2 for _, s, e2 in spans):
                problems.append(
                    f"line {line_of(text, pos)}: {sorted(names)} outside engine_read/engine_write"
                )
    for entry, s, e in spans:
        reached = set().union(*(names for _, names in helpers.calls_in(text, s, e)))
        writes = reached & MUTATING
        if entry == "engine_read" and writes:
            problems.append(
                f"line {line_of(text, s)}: engine_read writes {sorted(writes)}"
            )
        if entry == "engine_write" and not writes:
            problems.append(f"line {line_of(text, s)}: engine_write with no core write")
    return problems


# The persistence pyclasses are written once in incr-python-persist by
# `persist_classes!`, and Local registers them.
LOCAL_SOURCES = [
    p.relative_to(CRATES).as_posix()
    for p in [*sorted(LOCAL.rglob("*.rs")), SHARED / "persist_classes.rs"]
    if method_blocks(p.read_text())
]


def test_the_local_scan_covers_the_files_with_pymethods():
    assert {
        "incr-python/src/lib.rs",
        "incr-python-persist/src/persist_classes.rs",
    } <= set(LOCAL_SOURCES)


@pytest.mark.parametrize("source", LOCAL_SOURCES)
def test_every_local_engine_call_goes_through_the_entry_its_kind_needs(source):
    text = (CRATES / source).read_text()
    others = [
        p.read_text()
        for p in [*sorted(LOCAL.rglob("*.rs")), *sorted(SHARED.rglob("*.rs"))]
        if p != CRATES / source
    ]
    problems = local_violations(text, others)
    assert not problems, "\n".join(problems)


SYNTHETIC = """
fn engine_read<R>(rt: &Runtime, f: impl FnOnce() -> R) -> R { f() }
fn engine_write<R>(rt: &Runtime, f: impl FnOnce() -> R) -> R { f() }
#[pymethods]
impl PyThing {
    fn read(&self) { engine_read(rt, || rt.get(self.node)) }
    fn write(&self) {
        let _hold = rt.hold_routing_suppressed();
        engine_write(rt, || self.inner.insert(rt, v))
    }
    fn both(&self) {
        engine_write(rt, || lookups.bump(rt));
        engine_read(rt, || rt.get(n))
    }
    fn plain(&self) -> u32 { self.id }
    BODY
}
"""
HELPER = """
impl Lookups {
    fn bump(&self, rt: &Runtime) { self.store(rt, 1) }
    fn store(&self, rt: &Runtime, v: u64) { rt.set(self.node, v) }
}
"""


def test_the_local_scan_accepts_a_binding_that_follows_the_rule():
    assert local_violations(SYNTHETIC.replace("BODY", ""), [HELPER]) == []


def test_the_local_scan_flags_a_bare_core_call_in_an_overlay_style_pymethod():
    text = """
#[pymethods]
impl PyOverlay {
    fn __exit__(&self, py: Python<'_>) -> PyResult<bool> {
        let runtime = self.rt.bind(py).borrow();
        let rt = runtime.rt()?;
        rt.delete_with_dependents(node);
        Ok(false)
    }
}
"""
    problems = local_violations(text)
    assert len(problems) == 1, problems
    assert "outside engine_read/engine_write" in problems[0]


def test_the_local_scan_flags_a_store_write_outside_an_entry():
    text = """
#[pymethods]
impl PyStore {
    fn set(&self, py: Python<'_>) -> PyResult<()> {
        core.run(py, |store| store.checkpoint())
    }
}
"""
    problems = local_violations(text)
    assert len(problems) == 1, problems


@pytest.mark.parametrize(
    "body, problem",
    [
        ("fn bare(&self) { rt.get(self.node) }", "outside engine_read/engine_write"),
        (
            "fn sneaky(&self) { engine_read(rt, || rt.set(n, v)) }",
            "engine_read writes ['set']",
        ),
        (
            "fn hidden(&self) { engine_read(rt, || lookups.bump(rt)) }",
            "engine_read writes ['set']",
        ),
        (
            "fn idle(&self) { engine_write(rt, || rt.get(n)) }",
            "engine_write with no core write",
        ),
    ],
    ids=[
        "no-entry",
        "write-in-read",
        "write-in-read-through-helpers",
        "write-without-a-write",
    ],
)
def test_the_local_scan_rejects_a_violation(body, problem):
    problems = local_violations(SYNTHETIC.replace("BODY", body), [HELPER])
    assert len(problems) == 1, problems
    assert problem in problems[0]


# incr-python-persist reaches the engine through calls on the store, the
# runtime, the memo, a carried handle or the staged writes, and through
# calls that pass the store to a persisted handle.
SHARED_CALL = re.compile(
    r"\bstore\.(\w+)\(|\.(\w+)\(\s*&?store\b|\brt\.(\w+)\(|\bmemo\.(\w+)\("
    r"|\.into_inner\(\)\s*\.(\w+)\(|\bwrite\.(\w+)\("
)
# Accessors that take no engine entry, and the `OnceLock` holding the memo.
SHARED_ALLOWED = {"runtime", "get", "set", "into_inner", "take_encode_mismatches"}
# A binding entry, and the crate's own wrappers around one.
SHARED_ENTRY = re.compile(r"\bB::(?:write|read)\(|\b(?:self|core)\.(?:write|stage)\(")
SHARED_WRAPPERS = {"write", "stage"}
# Functions that run inside a query, where the engine entry is already open.
IN_QUERY = {"value_of"}


def without_strings(text):
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '"' + " " * (len(m.group()) - 2) + '"', text)


def shared_violations(text):
    """The engine calls of an incr-python-persist source outside a binding
    entry, and the crate's wrappers that do not enter through the binding."""
    text = without_strings(text)
    spans = [
        (m.end() - 1, matching(text, m.end() - 1, "(", ")"))
        for m in SHARED_ENTRY.finditer(text)
    ]
    problems = []
    for s, body, e in blocks(text, r"\bfn (\w+)\b[^;{]*"):
        name = re.match(r"fn (\w+)", text[s:]).group(1)
        if name in SHARED_WRAPPERS and not re.search(r"\bB::(?:write|read)\(", text[body:e]):
            problems.append(f"line {line_of(text, s)}: {name} does not enter through B")
        if name in IN_QUERY:
            continue
        for m in SHARED_CALL.finditer(text, body, e):
            called = next(g for g in m.groups() if g)
            if called in SHARED_ALLOWED:
                continue
            if not any(s2 <= m.start() <= e2 for s2, e2 in spans):
                problems.append(f"line {line_of(text, m.start())}: {called} outside B::write/B::read")
    return problems


SHARED_SOURCES = ["store.rs", "keyed.rs", "memo.rs", "batch.rs", "scope.rs"]


@pytest.mark.parametrize("source", SHARED_SOURCES)
def test_every_shared_store_call_enters_the_engine_through_the_binding(source):
    problems = shared_violations((SHARED / source).read_text())
    assert not problems, "\n".join(problems)


def test_the_shared_scan_covers_the_files_that_reach_the_store():
    reaching = {
        p.name
        for p in SHARED.glob("*.rs")
        if re.search(r"\bB::(?:write|read)\(|\bself\.(?:write|read)\(", p.read_text())
    }
    assert reaching <= set(SHARED_SOURCES)


@pytest.mark.parametrize(
    "body, problem",
    [
        (
            "pub fn refresh(&self) -> u64 { self.store.refresh() }",
            "refresh outside B::write/B::read",
        ),
        (
            "pub fn put(&self, c: Collection<B>) { c.insert(store, v) }",
            "insert outside B::write/B::read",
        ),
        (
            "fn write(&self, f: F) -> R { f(&self.store) }",
            "write does not enter through B",
        ),
    ],
    ids=["store-method", "store-argument", "wrapper-without-entry"],
)
def test_the_shared_scan_rejects_a_violation(body, problem):
    problems = shared_violations(f"impl<B: PyBinding> StoreCore<B> {{\n    {body}\n}}\n")
    assert any(problem in p for p in problems), problems


def test_the_shared_scan_accepts_a_store_call_inside_an_entry():
    text = """
impl<B: PyBinding> StoreCore<B> {
    pub fn refresh(&self, py: Python<'_>) -> PyResult<u64> {
        self.write(py, |store, _| store.refresh().map(|at| at.0))
    }
    fn write<R>(&self, py: Python<'_>, f: F) -> PyResult<R> {
        B::write(py, &self.store, move |store, d| f(store, d))
    }
}
"""
    assert shared_violations(text) == []
