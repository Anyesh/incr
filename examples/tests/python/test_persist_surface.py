"""Guards the Python surface of `incr.persist` and of the `incr` classes the
persistence layer backs: every signature and docstring must match the
checked-in record. Regenerate it with INCR_UPDATE_SURFACE=1 after an
intended change."""

import inspect
import json
import os
import pathlib

import incr
import incr.persist

EXPECTED = pathlib.Path(__file__).with_name("persist_surface.json")

TOUCHED_INCR_CLASSES = (
    "Runtime",
    "RuntimeRef",
    "Collection",
    "KeyedCollection",
    "Overlay",
    "NotReady",
    "Pending",
    "LookupFailed",
    "Unavailable",
    "Incomplete",
)

SKIPPED_MEMBERS = {"__module__", "__dict__", "__weakref__", "__doc__"}


def _signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _member(raw, bound):
    kind = type(raw).__name__
    entry = {"kind": kind, "doc": getattr(raw, "__doc__", None)}
    if callable(bound) and not isinstance(raw, (property,)):
        entry["signature"] = _signature(bound)
    return entry


def _class(cls):
    record = {
        "doc": cls.__doc__,
        "signature": _signature(cls),
        "bases": [base.__name__ for base in cls.__bases__],
        "members": {},
    }
    for name, raw in sorted(vars(cls).items()):
        if name in SKIPPED_MEMBERS:
            continue
        record["members"][name] = _member(raw, getattr(cls, name))
    return record


def _function(obj):
    return {"doc": obj.__doc__, "signature": _signature(obj)}


def _surface():
    surface = {}
    for name in sorted(dir(incr.persist)):
        if name.startswith("__"):
            continue
        obj = getattr(incr.persist, name)
        key = f"incr.persist.{name}"
        if inspect.isclass(obj):
            surface[key] = _class(obj)
        elif callable(obj):
            surface[key] = _function(obj)
        elif inspect.ismodule(obj):
            for member in sorted(dir(obj)):
                if not member.startswith("__"):
                    surface[f"{key}.{member}"] = _function(getattr(obj, member))
    for name in TOUCHED_INCR_CLASSES:
        surface[f"incr.{name}"] = _class(getattr(incr, name))
    return surface


def test_persist_surface_matches_the_record():
    actual = _surface()
    if os.environ.get("INCR_UPDATE_SURFACE") == "1":
        EXPECTED.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n")
    expected = json.loads(EXPECTED.read_text())
    assert sorted(actual) == sorted(expected)
    for key in expected:
        assert actual[key] == expected[key], key
