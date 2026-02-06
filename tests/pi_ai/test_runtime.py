from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from pi_py.agent_core import (
    Agent,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    Model,
    TextContent,
)
from pi_py.pi_ai import complete_simple, create_agent_stream_fn, create_default_registry


def _assistant_text(message: AssistantMessage) -> str:
    return " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


@pytest.mark.asyncio
async def test_complete_simple_returns_echo_response() -> None:
    registry = create_default_registry()
    model = Model(id="mock-model", provider="mock-provider", api="mock-api")

    message = await complete_simple(
        "hello from simple complete",
        model=model,
        registry=registry,
    )

    assert message.stop_reason == "stop"
    assert _assistant_text(message).startswith("Echo:")


@pytest.mark.asyncio
async def test_agent_e2e_uses_mock_provider_and_tool() -> None:
    registry = create_default_registry()
    model = Model(id="mock-model", provider="mock-provider", api="mock-api")
    seen_cities: list[str] = []

    async def get_weather(
        tool_call_id: str,
        params: Mapping[str, Any],
        abort_event: asyncio.Event | None = None,
        on_update: Callable[[AgentToolResult[Any]], None] | None = None,
    ) -> AgentToolResult[Any]:
        del tool_call_id, abort_event
        city = str(params.get("city", "Unknown"))
        seen_cities.append(city)
        result = AgentToolResult(
            content=[TextContent(text=f"sunny in {city}")],
            details={"city": city},
        )
        if on_update is not None:
            on_update(result)
        return result

    agent = Agent(
        stream_fn=create_agent_stream_fn(registry),
        session_id="test-session",
    )
    agent.set_model(model)
    agent.set_tools(
        [
            AgentTool(
                name="get_weather",
                label="Get Weather",
                description="Returns the weather for a city.",
                execute=get_weather,
            )
        ]
    )

    await agent.prompt("Can you check the weather in san francisco?")

    assert seen_cities == ["San Francisco"]
    assert isinstance(agent.state.messages[-1], AssistantMessage)
    assert _assistant_text(agent.state.messages[-1]).startswith("Weather update:")
    assert "sunny in San Francisco" in _assistant_text(agent.state.messages[-1])
