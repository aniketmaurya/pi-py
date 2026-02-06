from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Mapping
from typing import TypeVar, cast

from ..agent_core.types import (
    AgentLoopConfig,
    AgentTool,
    AssistantMessage,
    AssistantStream,
    LlmContext,
    Model,
    StreamFn,
    ThinkingLevel,
    UserMessage,
)
from .registry import ProviderRegistry
from .types import PiAIRequest

T = TypeVar("T")


async def stream(
    *,
    model: Model,
    context: LlmContext,
    registry: ProviderRegistry,
    reasoning: ThinkingLevel | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
    thinking_budgets: Mapping[str, int] | None = None,
    max_retry_delay_ms: int | None = None,
    abort_event: asyncio.Event | None = None,
) -> AssistantStream:
    provider = registry.resolve(model)
    request = PiAIRequest(
        model=model,
        context=context,
        reasoning=reasoning,
        api_key=api_key,
        session_id=session_id,
        thinking_budgets=thinking_budgets,
        max_retry_delay_ms=max_retry_delay_ms,
    )
    return await _maybe_await(provider.stream(request, abort_event))


async def complete(
    *,
    model: Model,
    context: LlmContext,
    registry: ProviderRegistry,
    reasoning: ThinkingLevel | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
    thinking_budgets: Mapping[str, int] | None = None,
    max_retry_delay_ms: int | None = None,
    abort_event: asyncio.Event | None = None,
) -> AssistantMessage:
    assistant_stream = await stream(
        model=model,
        context=context,
        registry=registry,
        reasoning=reasoning,
        api_key=api_key,
        session_id=session_id,
        thinking_budgets=thinking_budgets,
        max_retry_delay_ms=max_retry_delay_ms,
        abort_event=abort_event,
    )
    async for _ in assistant_stream:
        pass
    return await assistant_stream.result()


async def stream_simple(
    prompt: str,
    *,
    model: Model,
    registry: ProviderRegistry,
    system_prompt: str | None = None,
    tools: list[AgentTool] | None = None,
    reasoning: ThinkingLevel | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
    thinking_budgets: Mapping[str, int] | None = None,
    max_retry_delay_ms: int | None = None,
    abort_event: asyncio.Event | None = None,
) -> AssistantStream:
    context = LlmContext(
        system_prompt=system_prompt,
        messages=[UserMessage(content=prompt)],
        tools=tools,
    )
    return await stream(
        model=model,
        context=context,
        registry=registry,
        reasoning=reasoning,
        api_key=api_key,
        session_id=session_id,
        thinking_budgets=thinking_budgets,
        max_retry_delay_ms=max_retry_delay_ms,
        abort_event=abort_event,
    )


async def complete_simple(
    prompt: str,
    *,
    model: Model,
    registry: ProviderRegistry,
    system_prompt: str | None = None,
    tools: list[AgentTool] | None = None,
    reasoning: ThinkingLevel | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
    thinking_budgets: Mapping[str, int] | None = None,
    max_retry_delay_ms: int | None = None,
    abort_event: asyncio.Event | None = None,
) -> AssistantMessage:
    assistant_stream = await stream_simple(
        prompt,
        model=model,
        registry=registry,
        system_prompt=system_prompt,
        tools=tools,
        reasoning=reasoning,
        api_key=api_key,
        session_id=session_id,
        thinking_budgets=thinking_budgets,
        max_retry_delay_ms=max_retry_delay_ms,
        abort_event=abort_event,
    )
    async for _ in assistant_stream:
        pass
    return await assistant_stream.result()


def create_agent_stream_fn(registry: ProviderRegistry) -> StreamFn:
    async def _stream_fn(
        model: Model,
        context: LlmContext,
        config: AgentLoopConfig,
        abort_event: asyncio.Event | None,
    ) -> AssistantStream:
        return await stream(
            model=model,
            context=context,
            registry=registry,
            reasoning=config.reasoning,
            api_key=config.api_key,
            session_id=config.session_id,
            thinking_budgets=config.thinking_budgets,
            max_retry_delay_ms=config.max_retry_delay_ms,
            abort_event=abort_event,
        )

    return _stream_fn


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value
