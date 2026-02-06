from __future__ import annotations

import asyncio
import os
from typing import cast

from pi_agent.agent_core import AssistantMessage, Model, TextContent, ToolCall
from pi_agent.agent_core.types import (
    ErrorEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
)
from pi_agent.pi_ai import create_default_registry, stream_simple


def extract_assistant_text(message: AssistantMessage) -> str:
    return " ".join(
        block.text for block in message.content if isinstance(block, TextContent)
    ).strip()


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it and run `uv run python examples/openai_streaming.py`."
        )

    registry = create_default_registry()
    model = Model(id="gpt-5-mini", provider="openai", api="openai")

    assistant_stream = await stream_simple(
        "Write a short haiku about shipping Python libraries.",
        model=model,
        registry=registry,
        system_prompt="Respond with plain text only.",
        reasoning="minimal",
    )

    print("[stream:start]")
    tool_call_started = False

    async for event in assistant_stream:
        event_type = event["type"]

        if event_type == "text_start":
            if not tool_call_started:
                print("[assistant]", end=" ", flush=True)
            continue

        if event_type == "text_delta":
            text_event = cast(TextDeltaEvent, event)
            print(text_event["delta"], end="", flush=True)
            continue

        if event_type == "text_end":
            print()
            continue

        if event_type == "toolcall_start":
            tool_call_started = True
            print("[toolcall:start]")
            continue

        if event_type == "toolcall_delta":
            tool_delta_event = cast(ToolCallDeltaEvent, event)
            print(f"[toolcall:delta] {tool_delta_event['delta']}")
            continue

        if event_type == "toolcall_end":
            tool_end_event = cast(ToolCallEndEvent, event)
            tool = tool_end_event["tool_call"]
            if isinstance(tool, ToolCall):
                print(f"[toolcall:end] {tool.name} args={tool.arguments}")
            continue

        if event_type == "error":
            error_event = cast(ErrorEvent, event)
            print(f"[stream:error] {error_event['error'].error_message}")
            continue

        if event_type == "done":
            print("[stream:done]")

    final_message = await assistant_stream.result()
    print(f"\nFinal text: {extract_assistant_text(final_message)}")


if __name__ == "__main__":
    asyncio.run(main())
