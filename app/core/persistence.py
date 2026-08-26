"""Persistence abstraction with an in-memory implementation.

State is held in process memory (never temp files). The store exposes
namespaced dictionaries guarded by a lock so a Redis/file backend can be
substituted later without touching endpoint logic.
"""

import threading


class StateStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._ns: dict = {}

    def bucket(self, namespace: str) -> dict:
        with self._lock:
            return self._ns.setdefault(namespace, {})

    def get(self, namespace: str, key):
        return self.bucket(namespace).get(key)

    def set(self, namespace: str, key, value):
        with self._lock:
            self.bucket(namespace)[key] = value

    def update(self, namespace: str, key, fn):
        """Atomically read-modify-write; fn(current)->new (current may be None)."""
        with self._lock:
            b = self.bucket(namespace)
            new = fn(b.get(key))
            b[key] = new
            return new


_store = StateStore()


def get_store() -> StateStore:
    return _store
