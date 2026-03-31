from incr import Runtime


def test_create_and_get_input():
    rt = Runtime()
    x = rt.create_input(42)
    assert rt.get(x) == 42


def test_set_and_get():
    rt = Runtime()
    x = rt.create_input(10)
    assert rt.get(x) == 10
    rt.set(x, 20)
    assert rt.get(x) == 20


def test_simple_query():
    rt = Runtime()
    x = rt.create_input(10)
    y = rt.create_input(5)
    area = rt.create_query(lambda rt: rt.get(x) * rt.get(y))
    assert rt.get(area) == 50


def test_query_updates_on_input_change():
    rt = Runtime()
    x = rt.create_input(10)
    y = rt.create_input(5)
    area = rt.create_query(lambda rt: rt.get(x) * rt.get(y))
    assert rt.get(area) == 50
    rt.set(x, 12)
    assert rt.get(area) == 60


def test_chained_queries():
    rt = Runtime()
    a = rt.create_input(5)
    b = rt.create_query(lambda rt: rt.get(a) + 1)
    c = rt.create_query(lambda rt: rt.get(b) * 2)
    assert rt.get(c) == 12
    rt.set(a, 10)
    assert rt.get(c) == 22


def test_diamond_dependency():
    rt = Runtime()
    a = rt.create_input(1)
    b = rt.create_query(lambda rt: rt.get(a) + 10)
    c = rt.create_query(lambda rt: rt.get(a) + 100)
    d = rt.create_query(lambda rt: rt.get(b) + rt.get(c))
    assert rt.get(d) == 112
    rt.set(a, 2)
    assert rt.get(d) == 114


def test_string_values():
    rt = Runtime()
    first = rt.create_input("Hello")
    last = rt.create_input("World")
    full = rt.create_query(lambda rt: f"{rt.get(first)} {rt.get(last)}")
    assert rt.get(full) == "Hello World"
    rt.set(first, "Goodbye")
    assert rt.get(full) == "Goodbye World"


def test_early_cutoff():
    rt = Runtime()
    a = rt.create_input(50)
    b = rt.create_query(lambda rt: min(rt.get(a), 100))
    call_count = [0]

    def make_label(rt_ref):
        call_count[0] += 1
        return f"value: {rt_ref.get(b)}"

    c = rt.create_query(make_label)

    assert rt.get(c) == "value: 50"
    assert call_count[0] == 1

    rt.set(a, 200)
    assert rt.get(c) == "value: 100"
    assert call_count[0] == 2

    rt.set(a, 300)  # b still 100 — early cutoff
    assert rt.get(c) == "value: 100"
    assert call_count[0] == 2  # NOT recomputed
