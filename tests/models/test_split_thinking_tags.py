"""Public streaming regressions for #3007 and the integration gaps in #6021.

Synthetic SDK chunks make delimiter splits and tool ordering deterministic;
provider recordings cannot guarantee these boundaries.
"""

from __future__ import annotations as _annotations

from typing import Literal

import httpx
import pytest

from pydantic_ai import (
    Agent,
    ModelRequest,
    NativeToolCallPart,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.profiles import ModelProfile

from .._inline_snapshot import snapshot
from ..conftest import IsStr, try_import

with try_import() as openai_available:
    from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIDelta

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI
    from .test_openai import chunk as openai_chunk, struc_chunk as openai_tool_chunk, text_chunk as openai_text_chunk

with try_import() as groq_available:
    from groq import APIError
    from groq.types.chat.chat_completion_chunk import (
        ChoiceDelta as GroqDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from groq.types.chat.chat_completion_message import ExecutedTool

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    from .test_groq import MockGroq, chunk as groq_chunk, text_chunk as groq_text_chunk

with try_import() as huggingface_available:
    from huggingface_hub import ChatCompletionStreamOutputDelta
    from huggingface_hub.inference._generated.types.chat_completion import (
        ChatCompletionStreamOutputDeltaToolCall,
        ChatCompletionStreamOutputFunction,
    )

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import MockHuggingFace, chunk as hf_chunk, text_chunk as hf_text_chunk


@pytest.fixture(params=['openai', 'groq', 'huggingface'])
def provider(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    available = {'openai': openai_available, 'groq': groq_available, 'huggingface': huggingface_available}
    if not available[name]():
        pytest.skip(f'{name} not installed')
    return name


def make_model(
    provider: str,
    chunks: list[str],
    *,
    tool_call: bool = False,
    native_reasoning: bool = False,
    ignore_whitespace: bool = False,
    thinking_tags: tuple[str, str] = ('<think>', '</think>'),
) -> Model:
    profile = ModelProfile(ignore_streamed_leading_whitespace=ignore_whitespace, thinking_tags=thinking_tags)
    if provider == 'openai':
        first = [openai_text_chunk(c) for c in chunks]
        if native_reasoning:
            first.append(openai_chunk([OpenAIDelta.model_construct(reasoning_content='Reason.')]))
        if tool_call:
            first.append(openai_tool_chunk('lookup', '{}'))
        streams = [first, [openai_text_chunk('Done.')]] if tool_call else [first]
        return OpenAIChatModel(
            'test', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(streams)), profile=profile
        )
    if provider == 'groq':
        groq_first = [groq_text_chunk(c) for c in chunks]
        if native_reasoning:
            groq_first.append(groq_chunk([GroqDelta(reasoning='Reason.')]))
        if tool_call:
            groq_first.append(
                groq_chunk(
                    [
                        GroqDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id='call',
                                    type='function',
                                    function=ChoiceDeltaToolCallFunction(name='lookup', arguments='{}'),
                                )
                            ]
                        )
                    ]
                )
            )
        groq_streams = [groq_first, [groq_text_chunk('Done.')]] if tool_call else [groq_first]
        return GroqModel(
            'test', provider=GroqProvider(groq_client=MockGroq.create_mock_stream(groq_streams)), profile=profile
        )
    hf_first = [hf_text_chunk(c) for c in chunks]
    if tool_call:
        hf_first.append(
            hf_chunk(
                [
                    ChatCompletionStreamOutputDelta(
                        role='assistant',
                        tool_calls=[
                            ChatCompletionStreamOutputDeltaToolCall(
                                index=0,
                                id='call',
                                type='function',
                                function=ChatCompletionStreamOutputFunction(name='lookup', arguments='{}'),
                            )
                        ],
                    )
                ]
            )
        )
    hf_streams = [hf_first, [hf_text_chunk('Done.')]] if tool_call else [hf_first]
    return HuggingFaceModel(
        'test',
        provider=HuggingFaceProvider(hf_client=MockHuggingFace.create_stream_mock(hf_streams), api_key='test'),
        profile=profile,
    )


@pytest.mark.parametrize(
    'chunks, expected',
    [
        (['literal<'], 'literal<'),
        ([' <', ' 3'], ' < 3'),
        ([' <'], ' <'),
        ([' <th', 'ong>'], ' <thong>'),
        ([' ', '<', ' 3'], '< 3'),
    ],
)
async def test_literal_suffix_is_not_lost(provider: str, chunks: list[str], expected: str, allow_model_requests: None):
    model = make_model(provider, chunks, ignore_whitespace=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass
        assert stream.get().parts == [TextPart(content=expected)]


@pytest.mark.parametrize('chunks', [[' <think>\n</think>\n\n'], [' <', 'think>\n</think>\n\n']])
async def test_thinking_prefix_does_not_skip_tool(provider: str, chunks: list[str], allow_model_requests: None):
    agent = Agent(make_model(provider, chunks, tool_call=True, ignore_whitespace=True))
    calls: list[str] = []

    @agent.tool_plain
    def lookup() -> str:
        calls.append('lookup')
        return 'found'

    async with agent.run_stream('Hello') as result:
        output = await result.get_output()
    assert calls == ['lookup']
    assert output == 'Done.'


@pytest.mark.parametrize('provider', ['openai', 'groq'], indirect=True)
async def test_pending_text_precedes_native_reasoning(provider: str, allow_model_requests: None):
    model = make_model(provider, ['<'], native_reasoning=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        async for _ in stream:
            pass
        assert stream.get().parts == [
            TextPart(content='<'),
            ThinkingPart(
                content='Reason.',
                id='reasoning_content' if provider == 'openai' else None,
                provider_name='openai' if provider == 'openai' else None,
            ),
        ]


@pytest.mark.parametrize('chunks', [['<'], ['literal<'], ['<think>Reason.</th']])
async def test_pending_text_precedes_tool(provider: str, chunks: list[str], allow_model_requests: None):
    model = make_model(provider, chunks, tool_call=True)
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        events = [event async for event in stream]
        parts = stream.get().parts
    expected = (
        ThinkingPart(
            content='Reason.</th',
            id='content' if provider == 'openai' else None,
            provider_name='openai' if provider == 'openai' else None,
        )
        if chunks[0].startswith('<think>')
        else TextPart(content=''.join(chunks))
    )
    assert parts == [expected, ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr())]
    assert [(event.index, event.part.part_kind) for event in events if isinstance(event, PartStartEvent)] == snapshot(
        [(0, expected.part_kind), (1, 'tool-call')]
    )
    assert [event.index for event in events if isinstance(event, PartEndEvent)] == snapshot([0, 1])


@pytest.mark.parametrize('thinking_tags', [('<think>', '</think>'), ('[reason]', '[/reason]')])
async def test_every_split_and_repeated_thinking_sections(
    provider: str, thinking_tags: tuple[str, str], allow_model_requests: None
):
    start, end = thinking_tags
    content = f'{start}one{end}A{start}two{end}B'
    expected = [
        ThinkingPart(
            content='one',
            id='content' if provider == 'openai' else None,
            provider_name='openai' if provider == 'openai' else None,
        ),
        TextPart(content='A'),
        ThinkingPart(
            content='two',
            id='content' if provider == 'openai' else None,
            provider_name='openai' if provider == 'openai' else None,
        ),
        TextPart(content='B'),
    ]
    # Exercise every two-chunk split as well as one-character chunks through each adapter.
    partitions = [[content[:split], content[split:]] for split in range(1, len(content))]
    partitions.append(list(content))
    for chunks in partitions:
        model = make_model(provider, chunks, thinking_tags=thinking_tags)
        async with model.request_stream(
            [ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
            assert stream.get().parts == expected, chunks


async def test_unfinished_tag_can_be_final_output(provider: str, allow_model_requests: None):
    agent = Agent(make_model(provider, ['<']))
    async with agent.run_stream('Hello') as result:
        assert await result.get_output() == '<'


@pytest.mark.parametrize('provider', ['groq'], indirect=True)
@pytest.mark.parametrize('tool_kind', ['native', 'recovered'])
async def test_pending_text_precedes_groq_tool(
    provider: str, tool_kind: Literal['native', 'recovered'], allow_model_requests: None
):
    if tool_kind == 'native':
        tool = groq_chunk([GroqDelta(executed_tools=[ExecutedTool(arguments='{}', index=0, type='search')])])
        expected = NativeToolCallPart(tool_name='web_search', args={}, tool_call_id=IsStr(), provider_name='groq')
    else:
        tool = APIError(
            'invalid arguments',
            request=httpx.Request('POST', 'https://api.groq.com'),
            body={
                'message': 'invalid arguments',
                'type': 'invalid_request_error',
                'code': 'tool_use_failed',
                'failed_generation': '{"name":"lookup","arguments":{}}',
            },
        )
        expected = ToolCallPart(tool_name='lookup', args={}, tool_call_id=IsStr())
    client = MockGroq.create_mock_stream([groq_text_chunk('<'), tool])
    model = GroqModel('test', provider=GroqProvider(groq_client=client))
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        events = [event async for event in stream]
        assert stream.get().parts == [TextPart(content='<'), expected]
        assert [event.index for event in events if isinstance(event, PartStartEvent)] == snapshot([0, 1])


@pytest.mark.parametrize('thinking_tags', [('', ''), ('', '</think>'), ('<think>', '')])
async def test_empty_thinking_tags_leave_text_unchanged(
    provider: str, thinking_tags: tuple[str, str], allow_model_requests: None
):
    model = make_model(provider, ['<think>Answer</think>'], thinking_tags=thinking_tags)
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        event_count = 0
        async for _ in stream:
            event_count += 1
            # Bound the reproduction: an empty delimiter must never emit unbounded empty thinking parts.
            if event_count == 8:
                break
        assert stream.get().parts == [TextPart(content='<think>Answer</think>')]


@pytest.mark.parametrize('provider', ['openai'], indirect=True)
async def test_pending_text_before_reasoning_keeps_resumed_text_separate(provider: str, allow_model_requests: None):
    chunks = [
        openai_text_chunk('<'),
        openai_chunk([OpenAIDelta.model_construct(reasoning_content='Reason.')]),
        openai_chunk([OpenAIDelta.model_construct(reasoning_content='More.')]),
        openai_text_chunk('answer'),
    ]
    model = OpenAIChatModel('test', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(chunks)))
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        events = [event async for event in stream]
        assert stream.get().parts == [
            TextPart(content='<'),
            ThinkingPart(content='Reason.More.', id='reasoning_content', provider_name='openai'),
            TextPart(content='answer'),
        ]
        assert [event.index for event in events if isinstance(event, PartEndEvent)] == snapshot([0, 1, 2])


@pytest.mark.parametrize('provider', ['openai'], indirect=True)
async def test_eof_flush_keeps_whitespace_after_reasoning(provider: str, allow_model_requests: None):
    chunks = [
        openai_text_chunk('before'),
        openai_chunk([OpenAIDelta.model_construct(reasoning_content='Reason.')]),
        openai_text_chunk(' '),
        openai_text_chunk('<'),
    ]
    model = OpenAIChatModel(
        'test',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(chunks)),
        profile=ModelProfile(ignore_streamed_leading_whitespace=True),
    )
    async with model.request_stream([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters()) as stream:
        events = [event async for event in stream]
        assert stream.get().parts == [
            TextPart(content='before'),
            ThinkingPart(content='Reason.', id='reasoning_content', provider_name='openai'),
            TextPart(content=' <'),
        ]
        assert [event.index for event in events if isinstance(event, PartEndEvent)] == snapshot([0, 1, 2])
