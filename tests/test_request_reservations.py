"""Exercise admission through agents: usage accounting alone cannot expose concurrent holds."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai._request_reservations import InFlightRequest, pre_counted_request, reserved_requests
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from .conftest import try_import

with try_import() as harness_imported:
    from pydantic_ai_harness.compaction import SummarizingCompaction


READINESS_TIMEOUT = 10


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('request_limit', [0, 5, 8, None])
async def test_parallel_shared_request_limit(stream: bool, request_limit: int | None):
    usage = RunUsage()
    limits = UsageLimits(request_limit=request_limit)
    initial_usage = TypeAdapter(RunUsage).dump_python(usage)
    release = anyio.Event()
    all_admitted_or_refused = anyio.Event()
    started = 0
    refused = 0
    succeeded = 0

    def ready() -> None:
        if started + refused == 8:
            all_admitted_or_refused.set()

    async def respond(*_args: object) -> ModelResponse:
        nonlocal started
        started += 1
        ready()
        await release.wait()
        return ModelResponse(parts=[TextPart('done')])

    async def respond_stream(*_args: object) -> AsyncIterator[str]:
        await respond()
        yield 'done'

    agent = Agent(FunctionModel(respond, stream_function=respond_stream))

    async def run() -> None:
        nonlocal refused, succeeded
        try:
            if stream:
                async with agent.run_stream('go', usage=usage, usage_limits=limits) as result:
                    assert await result.get_output() == 'done'
            else:
                assert (await agent.run('go', usage=usage, usage_limits=limits)).output == 'done'
        except UsageLimitExceeded:
            refused += 1
            ready()
        else:
            succeeded += 1

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as group:
            for _ in range(8):
                group.start_soon(run)
            await all_admitted_or_refused.wait()
            # Reservations are coordination state, not reported or persisted usage.
            assert usage.requests == 0
            assert TypeAdapter(RunUsage).dump_python(usage) == initial_usage
            release.set()

    admitted = 8 if request_limit is None else min(request_limit, 8)
    assert (started, succeeded, refused, usage.requests) == (admitted, admitted, 8 - admitted, admitted)
    assert reserved_requests(usage) == 0


@pytest.mark.parametrize('stream', [False, True])
async def test_cancelled_request_releases_capacity(stream: bool):
    usage = RunUsage()
    entered = anyio.Event()
    release = anyio.Event()

    async def respond(*_args: object) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('done')])

    async def respond_stream(*_args: object) -> AsyncIterator[str]:
        await respond()
        yield 'done'

    agent = Agent(FunctionModel(respond, stream_function=respond_stream))
    limits = UsageLimits(request_limit=1)

    async def run() -> None:
        if stream:
            async with agent.run_stream('go', usage=usage, usage_limits=limits) as result:
                await result.get_output()
        else:
            await agent.run('go', usage=usage, usage_limits=limits)

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(run)
            await entered.wait()
            assert reserved_requests(usage) == 1
            group.cancel_scope.cancel()
    assert usage.requests == 0
    assert reserved_requests(usage) == 0
    release.set()
    await run()
    assert usage.requests == 1
    assert reserved_requests(usage) == 0


@pytest.mark.parametrize('request_limit', [1, 2])
async def test_nested_hook_counts_parent_reservation(request_limit: int):
    usage = RunUsage()
    child = Agent(TestModel(custom_output_text='child'))
    child_refused = False

    @dataclass
    class NestedRequest(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            nonlocal child_refused
            try:
                await child.run('nested', usage=ctx.usage, usage_limits=ctx.usage_limits)
            except UsageLimitExceeded:
                child_refused = True
            return request_context

    parent = Agent(TestModel(custom_output_text='parent'), deps_type=type(None), capabilities=[NestedRequest()])
    assert (
        await parent.run('go', usage=usage, usage_limits=UsageLimits(request_limit=request_limit))
    ).output == 'parent'
    assert child_refused is (request_limit == 1)
    assert usage.requests == request_limit
    assert reserved_requests(usage) == 0


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_request_releases_capacity(stream: bool):
    usage = RunUsage()
    fail = True

    async def respond(*_args: object) -> ModelResponse:
        if fail:
            raise ValueError('provider failed')
        return ModelResponse(parts=[TextPart('done')])

    async def respond_stream(*_args: object) -> AsyncIterator[str]:
        await respond()
        yield 'done'

    agent = Agent(FunctionModel(respond, stream_function=respond_stream))

    async def run() -> None:
        if stream:
            async with agent.run_stream('go', usage=usage, usage_limits=UsageLimits(request_limit=1)) as result:
                await result.get_output()
        else:
            await agent.run('go', usage=usage, usage_limits=UsageLimits(request_limit=1))

    with pytest.raises(ValueError, match='provider failed'):
        await run()
    assert usage.requests == 0
    assert reserved_requests(usage) == 0
    fail = False
    await run()
    assert usage.requests == 1
    assert reserved_requests(usage) == 0


def test_request_hold_checks_other_limits_and_can_be_reused():
    usage = RunUsage(input_tokens=2)
    hold = InFlightRequest()
    with pytest.raises(UsageLimitExceeded, match='input_tokens_limit'):
        hold.check_and_hold(UsageLimits(input_tokens_limit=1), usage)
    assert reserved_requests(usage) == 0
    hold.check_and_hold(UsageLimits(request_limit=None), usage)
    assert usage.requests == 0
    with pytest.raises(AssertionError, match='only hold one'):
        hold.check_and_hold(UsageLimits(), usage)
    hold.release()
    hold.release()
    assert reserved_requests(usage) == 0


@pytest.mark.parametrize('stream', [False, True])
async def test_retry_without_response_releases_capacity(stream: bool):
    usage = RunUsage()
    attempts = 0

    async def retry(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry('try again')
        return request_context

    async def respond(*_args: object) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    async def respond_stream(*_args: object) -> AsyncIterator[str]:
        yield 'done'

    agent = Agent(
        FunctionModel(respond, stream_function=respond_stream),
        deps_type=type(None),
        capabilities=[Hooks(before_model_request=retry)],
    )
    limits = UsageLimits(request_limit=1)
    if stream:
        async with agent.run_stream('go', usage=usage, usage_limits=limits) as result:
            assert await result.get_output() == 'done'
    else:
        assert (await agent.run('go', usage=usage, usage_limits=limits)).output == 'done'
    assert attempts == 2
    assert usage.requests == 1
    assert reserved_requests(usage) == 0


@pytest.mark.parametrize('stream', [False, True])
async def test_skip_counts_once_and_releases_capacity(stream: bool):
    usage = RunUsage()

    async def skip(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        raise SkipModelRequest(ModelResponse(parts=[TextPart('cached')]))

    agent = Agent(TestModel(), deps_type=type(None), capabilities=[Hooks(before_model_request=skip)])
    limits = UsageLimits(request_limit=1)
    if stream:
        async with agent.run_stream('go', usage=usage, usage_limits=limits) as result:
            assert await result.get_output() == 'cached'
    else:
        assert (await agent.run('go', usage=usage, usage_limits=limits)).output == 'cached'
    assert usage.requests == 1
    assert reserved_requests(usage) == 0
    with pytest.raises(UsageLimitExceeded):
        await agent.run('again', usage=usage, usage_limits=limits)
    assert reserved_requests(usage) == 0


async def test_token_preflight_does_not_count_its_own_hold():
    class CountingModel(TestModel):
        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            return RequestUsage(input_tokens=1)

    usage = RunUsage()
    agent = Agent(CountingModel(custom_output_text='done'))
    result = await agent.run(
        'go', usage=usage, usage_limits=UsageLimits(request_limit=1, count_tokens_before_request=True)
    )
    assert result.output == 'done'
    assert usage.requests == 1
    assert reserved_requests(usage) == 0


async def test_interrupted_stream_counts_partial_response_and_releases_capacity():
    usage = RunUsage()
    gate = anyio.Event()

    async def respond(*_args: object) -> AsyncIterator[str]:
        yield 'partial'
        await gate.wait()
        yield 'complete'

    agent = Agent(FunctionModel(stream_function=respond))
    with pytest.raises(ValueError, match='consumer failed'):
        async with agent.run_stream('go', usage=usage, usage_limits=UsageLimits(request_limit=1)) as result:
            async for text in result.stream_text(delta=True):
                assert text == 'partial'
                raise ValueError('consumer failed')
    assert usage.requests == 1
    assert reserved_requests(usage) == 0


async def test_pre_counted_claim_is_not_reserved_again():
    usage = RunUsage(requests=1)
    entered = anyio.Event()
    release = anyio.Event()

    async def respond(*_args: object) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('done')])

    judge = Agent(FunctionModel(respond))

    async def judge_run() -> None:
        with pre_counted_request(usage):
            await judge.run('judge', usage=usage, usage_limits=UsageLimits(request_limit=3))

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(judge_run)
            await entered.wait()
            # The claim already consumes the judge's slot; the parent can use its own.
            assert reserved_requests(usage) == 0
            await Agent(TestModel()).run('parent', usage=usage, usage_limits=UsageLimits(request_limit=2))
            release.set()
    # The caller releases its temporary claim after the nested run settles.
    assert usage.requests == 3
    assert reserved_requests(usage) == 0


async def test_pre_counted_claim_is_consumed_before_setup_hooks():
    claimed = RunUsage(requests=1)
    child_usage = RunUsage()
    entered = anyio.Event()
    release = anyio.Event()
    observed_holds: list[int] = []

    async def child_respond(*_args: object) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('child')])

    @dataclass
    class NestedSetup(AbstractCapability[None]):
        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            await Agent(FunctionModel(child_respond)).run(
                'setup', usage=child_usage, usage_limits=UsageLimits(request_limit=1)
            )
            return self

    async def judge_respond(*_args: object) -> ModelResponse:
        observed_holds.append(reserved_requests(claimed))
        return ModelResponse(parts=[TextPart('judge')])

    judge = Agent(FunctionModel(judge_respond), deps_type=type(None), capabilities=[NestedSetup()])

    async def judge_run() -> None:
        with pre_counted_request(claimed):
            await judge.run('judge', usage=claimed, usage_limits=UsageLimits(request_limit=2))

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(judge_run)
            await entered.wait()
            # Setup runs cannot inherit the judge's claim, even on another usage object.
            with pytest.raises(UsageLimitExceeded):
                await Agent(TestModel()).run('sibling', usage=child_usage, usage_limits=UsageLimits(request_limit=1))
            release.set()
    assert observed_holds == [0]
    assert child_usage.requests == 1
    assert reserved_requests(claimed) == reserved_requests(child_usage) == 0


@pytest.mark.skipif(not harness_imported(), reason='pydantic-ai-harness not installed')
async def test_pre_counted_compaction_does_not_borrow_a_siblings_hold():
    usage = RunUsage(requests=1)
    entered = anyio.Event()
    release = anyio.Event()

    async def respond(*_args: object) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('sibling')])

    sibling = Agent(FunctionModel(respond))
    judge = Agent(
        TestModel(custom_output_text='judge'),
        capabilities=[SummarizingCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False)],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('previous')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('previous')]),
    ]

    async def run_sibling() -> None:
        await sibling.run('sibling', usage=usage, usage_limits=UsageLimits(request_limit=2))

    with anyio.fail_after(READINESS_TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(run_sibling)
            await entered.wait()
            with pre_counted_request(usage):
                # The judge's launch claim and the sibling already occupy both slots.
                # Its compensated limit must still leave that claim for the judge itself.
                with pytest.raises(UsageLimitExceeded):
                    await judge.run(
                        'judge', message_history=history, usage=usage, usage_limits=UsageLimits(request_limit=3)
                    )
            # Match the judge caller's cleanup of its temporary launch claim.
            usage.requests -= 1
            release.set()
    assert usage.requests == 1
    assert reserved_requests(usage) == 0
