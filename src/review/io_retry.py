"""Retry helpers for flaky USB / LaCie EIO (errno 5)."""
from __future__ import annotations

import errno
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

_RETRY_ERRNOS = {errno.EIO, errno.EAGAIN, errno.EBUSY}
# ENXIO=6 appears on some macOS USB disconnects
if hasattr(errno, "ENXIO"):
    _RETRY_ERRNOS.add(errno.ENXIO)


def is_transient_io(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _RETRY_ERRNOS:
        return True
    msg = str(exc).lower()
    return "input/output error" in msg or "errno 5" in msg


def call_with_io_retry(
    fn: Callable[[], T],
    *,
    tries: int = 5,
    base_sleep: float = 0.15,
    label: str = "io",
) -> T:
    last: Optional[BaseException] = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — retry gate below
            last = exc
            if not is_transient_io(exc) or i + 1 >= tries:
                raise
            time.sleep(base_sleep * (2**i))
    assert last is not None
    raise last
