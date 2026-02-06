from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

import pytest

from pi_py.agent_core import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    AssistantMessageEvent,
    LlmContext,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from pi_py.pi_ai import OpenAIResponsesProvider, PiAIRequest
from pi_py.pi_ai.providers.openai import _event_to_mapping, _response_to_mapping


async def _noop_execute(
    tool_call_id: str,
    params: Mapping[str, Any],
    abort_event: asyncio.Event | None = None,
    on_update: Callable[[AgentToolResult[Any]], None] | None = None,
) -> AgentToolResult[Any]:
    del tool_call_id, params, abort_event, on_update
    return AgentToolResult(content=[TextContent(text="unused")], details={})


def _assistant_text(message: AssistantMessage) -> str:
    return " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


@pytest.mark.asyncio
async def test_openai_provider_emits_tool_call_response() -> None:
    seen_payloads: list[dict[str, Any]] = []

    async def request_fn(
        payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        seen_payloads.append(payload)
        return {
            "status": "completed",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Tokyo"}',
                }
            ],
        }

    provider = OpenAIResponsesProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(
            system_prompt="Use tools when needed.",
            messages=[UserMessage(content="Weather in Tokyo?")],
            tools=[
                AgentTool(
                    name="get_weather",
                    label="Get Weather",
                    description="Returns weather for a city",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    execute=_noop_execute,
                )
            ],
        ),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    assert message.stop_reason == "toolUse"
    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"city": "Tokyo"}
    assert message.usage.input == 7

    payload = seen_payloads[0]
    assert payload["model"] == "gpt-5"
    assert payload["instructions"] == "Use tools when needed."
    assert payload["input"][0]["role"] == "user"
    assert payload["tools"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_openai_provider_emits_text_response() -> None:
    async def request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello from OpenAI"}],
                }
            ],
        }

    provider = OpenAIResponsesProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    message = await stream.result()

    assert message.stop_reason == "stop"
    assert _assistant_text(message) == "Hello from OpenAI"


@pytest.mark.asyncio
async def test_openai_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PI_PY_TEST_OPENAI_KEY", raising=False)

    async def should_not_run(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        return await _raise_if_called()

    provider = OpenAIResponsesProvider(
        request_fn=should_not_run,
        api_key_env="PI_PY_TEST_OPENAI_KEY",
    )
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
    )

    stream = await provider.stream(request)
    events: list[AssistantMessageEvent] = []
    async for event in stream:
        events.append(event)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    error_message = events[0]["error"].error_message or ""
    assert "Missing OpenAI API key" in error_message


@pytest.mark.asyncio
async def test_openai_provider_sends_tool_result_as_function_output() -> None:
    seen_payloads: list[dict[str, Any]] = []

    async def request_fn(
        payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> Mapping[str, Any]:
        seen_payloads.append(payload)
        return {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Result acknowledged"}],
                }
            ],
        }

    provider = OpenAIResponsesProvider(request_fn=request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(
            messages=[
                UserMessage(content="Check weather in Berlin"),
                ToolResultMessage(
                    tool_call_id="call_berlin",
                    tool_name="get_weather",
                    content=[TextContent(text="Sunny in Berlin")],
                    is_error=False,
                ),
            ]
        ),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    async for _ in stream:
        pass
    _ = await stream.result()

    input_items = seen_payloads[0]["input"]
    function_output_items = [
        item for item in input_items if item.get("type") == "function_call_output"
    ]
    assert len(function_output_items) == 1
    assert function_output_items[0]["call_id"] == "call_berlin"
    assert "Sunny in Berlin" in function_output_items[0]["output"]


@pytest.mark.asyncio
async def test_openai_provider_streams_text_deltas() -> None:
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Hel",
            }
            yield {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "lo",
            }
            yield {
                "type": "response.output_text.done",
                "output_index": 0,
                "content_index": 0,
                "text": "Hello",
            }
            yield {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "output_index": 0,
                    "content": [
                        {"type": "output_text", "index": 0, "text": "Hello"},
                    ],
                },
            }
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Hello"},
                            ],
                        }
                    ],
                },
            }

        return _events()

    provider = OpenAIResponsesProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(messages=[UserMessage(content="Hello")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    event_types: list[str] = []
    text_deltas: list[str] = []
    async for event in stream:
        event_types.append(event["type"])
        if event["type"] == "text_delta":
            text_deltas.append(event["delta"])
    message = await stream.result()

    assert event_types == [
        "start",
        "text_start",
        "text_delta",
        "text_delta",
        "text_end",
        "done",
    ]
    assert text_deltas == ["Hel", "lo"]
    assert message.stop_reason == "stop"
    assert _assistant_text(message) == "Hello"


@pytest.mark.asyncio
async def test_openai_provider_streams_tool_call_updates() -> None:
    async def stream_request_fn(
        _payload: dict[str, Any],
        _api_key: str,
        _base_url: str | None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        async def _events() -> AsyncIterator[Mapping[str, Any]]:
            yield {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call_tokyo",
                    "name": "get_weather",
                },
            }
            yield {
                "type": "response.function_call_arguments.delta",
                "call_id": "call_tokyo",
                "name": "get_weather",
                "delta": '{"city":"Tok',
            }
            yield {
                "type": "response.function_call_arguments.delta",
                "call_id": "call_tokyo",
                "name": "get_weather",
                "delta": 'yo"}',
            }
            yield {
                "type": "response.function_call_arguments.done",
                "call_id": "call_tokyo",
                "name": "get_weather",
                "arguments": '{"city":"Tokyo"}',
            }
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_tokyo",
                            "name": "get_weather",
                            "arguments": '{"city":"Tokyo"}',
                        }
                    ],
                },
            }

        return _events()

    provider = OpenAIResponsesProvider(stream_request_fn=stream_request_fn)
    request = PiAIRequest(
        model=Model(id="gpt-5", provider="openai", api="openai"),
        context=LlmContext(messages=[UserMessage(content="Weather in Tokyo?")]),
        api_key="test-key",
    )

    stream = await provider.stream(request)
    event_types: list[str] = []
    async for event in stream:
        event_types.append(event["type"])
    message = await stream.result()

    tool_calls = [block for block in message.content if isinstance(block, ToolCall)]
    assert event_types == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]
    assert message.stop_reason == "toolUse"
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {"city": "Tokyo"}


def test_event_to_mapping_uses_warnings_false_for_model_dump() -> None:
    class FakeEvent:
        def model_dump(self, *, warnings: bool = True) -> Mapping[str, Any]:
            assert warnings is False
            return {"type": "response.completed"}

    mapped = _event_to_mapping(FakeEvent())
    assert mapped["type"] == "response.completed"


def test_response_to_mapping_uses_warnings_false_for_model_dump() -> None:
    class FakeResponse:
        def model_dump(self, *, warnings: bool = True) -> Mapping[str, Any]:
            assert warnings is False
            return {"status": "completed", "output": []}

    mapped = _response_to_mapping(FakeResponse())
    assert mapped["status"] == "completed"


async def _raise_if_called() -> Mapping[str, Any]:
    raise AssertionError("request_fn should not be called")
