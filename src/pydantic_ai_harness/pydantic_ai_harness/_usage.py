"""Shared helpers for nested agent usage accounting."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai._request_reservations import InFlightRequest
from pydantic_ai.usage import RunUsage, UsageLimits


def reserved_usage_limits(limits: UsageLimits | None, *, usage: RunUsage | None = None) -> UsageLimits | None:
    """Reserve the pending parent request before a nested model call made from a hook.

    The hook may run after the parent request's limit check. Reducing a finite request limit
    prevents the nested call from spending the request that was already approved for the parent.
    """
    # Request hooks share core's reservation in process. A durable worker receives
    # reconstructed usage instead, so its limit must still leave the parent's slot free.
    # Callers outside the request window omit `usage` to reserve a future parent step.
    if (
        limits is None
        or limits.request_limit is None
        or (usage is not None and InFlightRequest.current_is_reserved(usage))
    ):
        return limits
    return replace(limits, request_limit=max(0, limits.request_limit - 1))
