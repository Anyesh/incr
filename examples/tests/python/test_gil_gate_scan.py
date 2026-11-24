"""R24: a checkpoint worker is off the GIL except inside the one gate in
`incr-python-protocol/src/gil.rs`. Outside that file and R25's
`gil_yield.rs`, the code a worker can run takes and gives up the GIL only
through the gate, and the backend wrappers and reference backends touch
Python containers only through the gate's sequence functions. Every gate
site has a unique name; `declared_sites` is what the probe's coverage check
reads."""

import pathlib
import re

import pytest

CRATES = pathlib.Path(__file__).resolve().parents[3] / "crates"
PROTOCOL = CRATES / "incr-python-protocol" / "src"
GATE = PROTOCOL / "gil.rs"
YIELD = PROTOCOL / "gil_yield.rs"
PERSIST = CRATES / "incr-python-persist" / "src"
CONTAINER_FREE = [PERSIST / "backend.rs", PERSIST / "memo_backend.rs", PERSIST / "memory.rs"]

GIL_CALL = re.compile(
    r"Python::attach\b|\battach_unchecked\b|\btry_attach\b|\.detach\(|"
    r"\bPyEval_SaveThread\b|\bPyEval_RestoreThread\b|\bPyGILState_Ensure\b"
)
SITE = re.compile(
    r'\bSite::(caller|one|bounded|unbounded)\(\s*"([^"]+)"\s*(?:,\s*(\d[\d_]*)\s*)?\)'
)
ALLOWED_TYPES = {"PyBytes", "PyString", "PyAny"}


def scanned_files():
    files = sorted(PERSIST.rglob("*.rs"))
    files.append(PROTOCOL / "persist.rs")
    files.extend(sorted((CRATES / "incr-python" / "src" / "persist").rglob("*.rs")))
    return files


def without_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def gil_calls(text):
    code = without_comments(text)
    return [f"line {line_of(code, m.start())}: {m.group(0)}" for m in GIL_CALL.finditer(code)]


def container_problems(text):
    code = without_comments(text)
    problems = []
    for m in re.finditer(r"pyo3::types::\{([^}]*)\}|pyo3::types::(\w+)", code):
        names = m.group(1) or m.group(2)
        for name in re.split(r"[,\s]+", names):
            if name and name not in ALLOWED_TYPES:
                problems.append(f"line {line_of(code, m.start())}: imports pyo3::types::{name}")
    for m in re.finditer(r"\btry_iter\(", code):
        problems.append(f"line {line_of(code, m.start())}: try_iter(")
    for m in re.finditer(r"extract::<", code):
        end = m.end()
        depth = 1
        while depth and end < len(code):
            depth += {"<": 1, ">": -1}.get(code[end], 0)
            end += 1
        if re.search(r"\bVec\b", code[m.end() : end]):
            problems.append(f"line {line_of(code, m.start())}: Vec inside extract::<")
    return problems


def pymethod_signature_problems(text):
    code = without_comments(text)
    problems = []
    for block in re.finditer(r"#\[pymethods\]", code):
        start = code.index("{", block.end())
        depth, end = 0, start
        for end in range(start, len(code)):
            depth += {"{": 1, "}": -1}.get(code[end], 0)
            if depth == 0:
                break
        for fn in re.finditer(r"\bfn\s+\w+\s*(?:<[^>]*>)?\s*\(", code[start:end]):
            sig_start = start + fn.end()
            sig_end = code.index("{", sig_start)
            signature = code[sig_start:sig_end]
            for name in ("Vec", "HashSet", "HashMap"):
                if re.search(rf"\b{name}\b", signature):
                    problems.append(f"line {line_of(code, sig_start)}: signature names {name}")
    return problems


def declared_sites():
    """Every gate site in the scanned code and the gate, as
    `{name: (kind, bound)}`, with `kind` one of `caller`, `one`, `bounded`
    and `unbounded`."""
    sites = {}
    duplicates = []
    for path in [GATE, *scanned_files()]:
        for m in SITE.finditer(without_comments(path.read_text())):
            kind, name, bound = m.group(1), m.group(2), m.group(3)
            if name in sites:
                duplicates.append(name)
            sites[name] = (kind, int(bound.replace("_", "")) if bound else None)
    return sites, duplicates


@pytest.mark.parametrize("path", scanned_files(), ids=lambda p: str(p.relative_to(CRATES)))
def test_the_scanned_code_takes_the_gil_only_through_the_gate(path):
    assert gil_calls(path.read_text()) == []


@pytest.mark.parametrize("path", CONTAINER_FREE, ids=lambda p: p.name)
def test_backend_wrappers_reach_python_containers_only_through_the_gate(path):
    assert container_problems(path.read_text()) == []


def test_the_reference_backends_take_no_rust_collections_from_python():
    assert pymethod_signature_problems((PERSIST / "memory.rs").read_text()) == []


def test_the_gate_declares_unique_sites_and_workers_reach_some():
    sites, duplicates = declared_sites()
    assert duplicates == []
    kinds = {kind for kind, _ in sites.values()}
    assert {"caller", "one", "bounded", "unbounded"} <= kinds
    assert all(bound for kind, bound in sites.values() if kind == "bounded")


def test_the_gate_is_the_only_other_file_that_releases():
    assert gil_calls(GATE.read_text()) != []
    assert gil_calls(YIELD.read_text()) != []


@pytest.mark.parametrize(
    "body,expected",
    [
        ("fn f() { Python::attach(|py| ()) }", 1),
        ("fn f(py: Python<'_>) { py.detach(|| ()) }", 1),
        ("// Python::attach in a comment\nfn f() {}", 0),
        ("fn f() { unsafe { ffi::PyGILState_Ensure(); } }", 1),
    ],
)
def test_the_gil_scan_finds_a_bare_attach(body, expected):
    assert len(gil_calls(body)) == expected


@pytest.mark.parametrize(
    "body,problem",
    [
        ("use pyo3::types::{PyBytes, PyList};", "PyList"),
        ("use pyo3::types::PyTuple;", "PyTuple"),
        ("fn f(o: &Bound<'_, PyAny>) { o.try_iter(); }", "try_iter("),
        ("fn f(o: &Bound<'_, PyAny>) { o.extract::<(u64, Vec<u8>)>(); }", "Vec inside"),
    ],
)
def test_the_container_scan_rejects_a_violation(body, problem):
    assert any(problem in p for p in container_problems(body))


def test_the_container_scan_accepts_scalars_and_objects():
    body = (
        "use pyo3::types::{PyAny, PyBytes, PyString};\n"
        "fn f(o: &Bound<'_, PyAny>) { o.extract::<(u64, u64)>(); }"
    )
    assert container_problems(body) == []


def test_the_signature_scan_rejects_a_vec_argument():
    body = "#[pymethods]\nimpl X {\n    fn append(&self, records: Vec<Vec<u8>>) -> u64 { 0 }\n}"
    assert pymethod_signature_problems(body) != []
