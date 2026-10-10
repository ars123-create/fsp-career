"""Bounded per-process limiter for the single-process MVP (not a cluster limiter)."""
from collections import OrderedDict
from threading import Lock
import time
from fastapi import HTTPException, Request

_entries: OrderedDict = OrderedDict()
_lock = Lock()


def limit(request: Request, group: str, maximum=20, seconds=60):
    key = (request.client.host if request.client else "unknown", group)
    stamp = time.monotonic()
    with _lock:
        values = [v for v in _entries.get(key, []) if v > stamp - seconds]
        if len(values) >= maximum:
            raise HTTPException(429, "Слишком много запросов. Подождите минуту.", headers={"Retry-After": str(seconds)})
        _entries[key] = values + [stamp]
        _entries.move_to_end(key)
        while len(_entries) > 4096:
            _entries.popitem(last=False)
