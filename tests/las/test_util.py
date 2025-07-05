from __future__ import annotations

from daft.las.utils import not_blank


def test_not_blank():
    assert not_blank(None) is False
    assert not_blank("") is False
    assert not_blank("  ") is False
    assert not_blank("a") is True
    assert not_blank("a ") is True
