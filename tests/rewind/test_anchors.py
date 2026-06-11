import pytest
from rewind.core.anchors import AnchorStore


def test_set_get_has_len():
    s = AnchorStore()
    assert len(s) == 0 and not s.has(0)
    s.set(0, "x0")
    s.set(1, "x1")
    assert s.has(0) and s.get(1) == "x1" and len(s) == 2


def test_missing_anchor_raises_keyerror():
    s = AnchorStore()
    with pytest.raises(KeyError):
        s.get(3)


def test_as_dict_and_items_roundtrip():
    s = AnchorStore()
    s.set(2, "b"); s.set(0, "a")
    assert s.as_dict() == {0: "a", 2: "b"}
    assert dict(s.items()) == {0: "a", 2: "b"}
