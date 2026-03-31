from incr import Runtime


def test_collection_feeds_query():
    rt = Runtime()
    scores = rt.create_collection()
    high = scores.filter(lambda x: x >= 90)
    count = high.count()
    summary = rt.create_query(lambda rt: f"{rt.get(count)} high scores")

    scores.insert(85)
    scores.insert(92)
    scores.insert(95)
    assert rt.get(summary) == "2 high scores"

    scores.insert(91)
    assert rt.get(summary) == "3 high scores"

    scores.delete(92)
    assert rt.get(summary) == "2 high scores"


def test_early_cutoff_through_collection():
    rt = Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    count = evens.count()

    call_count = [0]
    def make_label(rt_ref):
        call_count[0] += 1
        return f"{rt_ref.get(count)} evens"

    label = rt.create_query(make_label)

    col.insert(2)
    assert rt.get(label) == "1 evens"
    assert call_count[0] == 1

    col.insert(3)  # odd — filter output unchanged
    assert rt.get(label) == "1 evens"
    # label should NOT recompute because count didn't change
    assert call_count[0] == 1
