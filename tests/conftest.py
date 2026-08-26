"""Shared pytest fixtures."""

import pytest

from app.core.persistence import get_store


@pytest.fixture(autouse=True)
def clean_state():
    """Isolate in-memory state between tests."""
    store = get_store()

    def _clear():
        for ns in list(store._ns.keys()):
            store.bucket(ns).clear()

    _clear()
    yield
    _clear()
