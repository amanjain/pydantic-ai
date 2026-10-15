"""Reserve logical request steps without changing reported or serialized usage.

A reservation holds its usage object alive, so its identity cannot be reused while the
side table contains an entry. Check, hold, count, and release run without an await;
this coordinates tasks sharing a usage object on the same event loop.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy

from .usage import RunUsage, UsageLimits

__all__ = ('InFlightRequest', 'pre_counted_request', 'reserved_requests')

_reserved: dict[int, int] = {}
_current: ContextVar[InFlightRequest | None] = ContextVar('pydantic_ai.in_flight_request', default=None)
_pre_counted: ContextVar[RunUsage | None] = ContextVar('pydantic_ai.pre_counted_request', default=None)


@contextmanager
def pre_counted_request(usage: RunUsage) -> Generator[None]:
    """Let the next run use a slot its caller already included in `usage.requests`.

    Harness judges claim a slot before starting a background task or durable operation,
    and offset their nested run's limit by one until that claim is released. Only that
    run consumes the claim; runs it delegates to must reserve their own slots.
    """
    token = _pre_counted.set(usage)
    try:
        yield
    finally:
        _pre_counted.reset(token)


def reserved_requests(usage: RunUsage) -> int:
    """Return the number of approved steps not yet counted against this usage."""
    return _reserved.get(id(usage), 0)


class InFlightRequest:
    """The slot held by a run's current step, until it is counted or abandoned."""

    def __init__(self) -> None:
        self._usage: RunUsage | None = None
        self._pre_counted = _pre_counted.get()
        # Consume the claim for this run; the enclosing context manager restores it.
        _pre_counted.set(None)

    @staticmethod
    def current_is_reserved(usage: RunUsage) -> bool:
        """Whether this task's current step holds a slot on this exact usage object.

        A judge's pre-counted claim and another run's hold do not reserve this step.
        Reconstructed usage in a durable worker likewise has no local parent hold.
        """
        current = _current.get()
        return current is not None and current._usage is usage and current._pre_counted is not usage

    @contextmanager
    def running(self) -> Generator[None]:
        """Identify the request owning hooks and tasks started in this scope."""
        token = _current.set(self)
        try:
            yield
        finally:
            _current.reset(token)

    def check_and_hold(self, limits: UsageLimits, usage: RunUsage) -> None:
        assert self._usage is None, 'A run can only hold one request slot'
        projected = copy(usage)
        # usage-attribution: a copy, projected forward only for admission
        projected.requests += reserved_requests(usage)
        limits.check_before_request(projected)
        self._usage = usage
        if self._pre_counted is not usage:
            _reserved[id(usage)] = reserved_requests(usage) + 1

    def release(self) -> None:
        usage = self._usage
        if usage is None:
            return
        self._usage = None
        if self._pre_counted is usage:
            return
        key = id(usage)
        if remaining := _reserved[key] - 1:
            _reserved[key] = remaining
        else:
            del _reserved[key]
