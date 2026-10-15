"""Exact-budget integration tests for nested model calls in request hooks."""

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, UsageLimitExceeded, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.compaction import SummarizingCompaction
from pydantic_ai_harness.system_reminders import LLMReminder, SystemReminders
from tests.harness._recording_durability import RecordingDurability


class TestRequestBudgets:
    @pytest.mark.parametrize('durable', [False, True])
    @pytest.mark.parametrize('request_limit', [1, 2])
    @pytest.mark.parametrize('capability', ['compaction', 'reminder'])
    async def test_nested_request_reserves_parent_once(self, durable: bool, request_limit: int, capability: str):
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart('done')])

        model = FunctionModel(respond)
        nested = (
            SummarizingCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False)
            if capability == 'compaction'
            else SystemReminders(dynamic_reminders=[LLMReminder(model=model)])
        )
        agent = Agent(
            model, name='request_budget', capabilities=[nested, RecordingDurability()] if durable else [nested]
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('first')]),
            ModelResponse(parts=[TextPart('previous')]),
            ModelRequest(parts=[UserPromptPart('second')]),
            ModelResponse(parts=[TextPart('previous')]),
        ]
        usage = RunUsage()
        if request_limit == 1 and capability == 'compaction':
            with pytest.raises(UsageLimitExceeded):
                await agent.run('go', message_history=history, usage=usage, usage_limits=UsageLimits(request_limit=1))
            assert usage.requests == calls == 0
        else:
            result = await agent.run(
                'go', message_history=history, usage=usage, usage_limits=UsageLimits(request_limit=request_limit)
            )
            assert result.output == 'done'
            assert usage.requests == calls == request_limit
