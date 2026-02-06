from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar, cast

from ...agent_core.event_stream import AssistantMessageEventStream
from ...agent_core.types import (
    AgentTool,
    AssistantContentBlock,
    AssistantMessage,
    AssistantStream,
    ImageContent,
    LlmContext,
    Model,
    StopReason,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from ..types import PiAIRequest

DoneReason = Literal["stop", "length", "toolUse"]
OpenAIRequestFn: TypeAlias = Callable[
    [dict[str, Any], str, str | None],
    Awaitable[Mapping[str, Any]],
]
OpenAIStreamRequestFn: TypeAlias = Callable[
    [dict[str, Any], str, str | None],
    AsyncIterator[Mapping[str, Any]] | Awaitable[AsyncIterator[Mapping[str, Any]]],
]
T = TypeVar("T")


@dataclass(slots=True)
class OpenAIResponsesProvider:
    request_fn: OpenAIRequestFn | None = None
    stream_request_fn: OpenAIStreamRequestFn | None = None
    api_key_env: str = "OPENAI_API_KEY"

    async def stream(
        self,
        request: PiAIRequest,
        abort_event: asyncio.Event | None = None,
    ) -> AssistantStream:
        stream = AssistantMessageEventStream()
        asyncio.create_task(self._emit(stream, request, abort_event))
        return stream

    async def _emit(
        self,
        stream: AssistantMessageEventStream,
        request: PiAIRequest,
        abort_event: asyncio.Event | None,
    ) -> None:
        await asyncio.sleep(0)

        if abort_event is not None and abort_event.is_set():
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": _assistant_error_message(
                        model=request.model,
                        stop_reason="aborted",
                        error_message="Request aborted",
                    ),
                }
            )
            return

        try:
            api_key = _resolve_api_key(request, self.api_key_env)
            payload = _build_openai_payload(request)
            base_url = request.model.base_url or None

            if self.stream_request_fn is not None:
                events = await _maybe_await(
                    self.stream_request_fn(payload, api_key, base_url)
                )
                await _consume_openai_event_stream(
                    stream=stream,
                    model=request.model,
                    events=events,
                )
                return

            if self.request_fn is not None:
                response = await self.request_fn(payload, api_key, base_url)
                assistant_message = _assistant_from_openai_response(
                    request.model,
                    response,
                )
                _push_terminal_message(stream, assistant_message)
                return

            events = _stream_openai_events(payload, api_key, base_url)
            await _consume_openai_event_stream(
                stream=stream,
                model=request.model,
                events=events,
            )
        except Exception as exc:  # noqa: BLE001
            stream.push(
                {
                    "type": "error",
                    "reason": "error",
                    "error": _assistant_error_message(
                        model=request.model,
                        stop_reason="error",
                        error_message=str(exc),
                    ),
                }
            )


@dataclass(slots=True)
class _OpenAIStreamingState:
    partial: AssistantMessage
    text_indices: dict[tuple[int, int], int] = field(default_factory=dict)
    closed_text_indices: set[int] = field(default_factory=set)
    tool_indices: dict[str, int] = field(default_factory=dict)
    closed_tool_call_ids: set[str] = field(default_factory=set)
    tool_arg_buffers: dict[str, str] = field(default_factory=dict)


def _push_terminal_message(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
) -> None:
    if message.stop_reason in {"error", "aborted"}:
        stream.push(
            {
                "type": "error",
                "reason": "aborted" if message.stop_reason == "aborted" else "error",
                "error": message,
            }
        )
        return

    stream.push(
        {
            "type": "done",
            "reason": _done_reason_for_message(message),
            "message": message,
        }
    )


async def _consume_openai_event_stream(
    *,
    stream: AssistantMessageEventStream,
    model: Model,
    events: AsyncIterator[Mapping[str, Any]],
) -> None:
    state = _OpenAIStreamingState(partial=_new_partial_message(model))
    stream.push({"type": "start", "partial": state.partial})

    completed = False
    async for raw_event in events:
        event = _event_to_mapping(raw_event)
        completed = _apply_openai_stream_event(
            stream=stream,
            model=model,
            state=state,
            event=event,
        )
        if completed:
            break

    if completed:
        return

    _close_open_text_blocks(state, stream)
    _close_open_tool_calls(state, stream)
    _push_terminal_message(stream, _clone_assistant_message(state.partial))


def _apply_openai_stream_event(
    *,
    stream: AssistantMessageEventStream,
    model: Model,
    state: _OpenAIStreamingState,
    event: Mapping[str, Any],
) -> bool:
    event_type = _as_str(event.get("type")) or ""

    if _event_type_matches(
        event_type,
        (
            "response.output_text.delta",
            "response.text.delta",
            "output_text.delta",
        ),
    ):
        delta = _as_str(event.get("delta")) or ""
        if not delta:
            return False

        content_index = _ensure_text_block(
            state=state,
            stream=stream,
            event=event,
        )
        text_block = cast(TextContent, state.partial.content[content_index])
        text_block.text += delta
        stream.push(
            {
                "type": "text_delta",
                "content_index": content_index,
                "delta": delta,
                "partial": state.partial,
            }
        )
        return False

    if _event_type_matches(
        event_type,
        (
            "response.output_text.done",
            "response.text.done",
            "output_text.done",
        ),
    ):
        content_index = _ensure_text_block(
            state=state,
            stream=stream,
            event=event,
        )
        text_block = cast(TextContent, state.partial.content[content_index])
        final_text = _as_str(event.get("text"))
        if final_text and not text_block.text:
            text_block.text = final_text
        _emit_text_end_if_needed(state, stream, content_index)
        return False

    if _event_type_matches(
        event_type,
        ("response.output_item.added", "output_item.added"),
    ):
        item = event.get("item") or event.get("output_item")
        if isinstance(item, Mapping):
            _apply_output_item(
                state=state,
                stream=stream,
                item=item,
                close_text=False,
            )
        return False

    if _event_type_matches(
        event_type,
        ("response.output_item.done", "output_item.done"),
    ):
        item = event.get("item") or event.get("output_item")
        if isinstance(item, Mapping):
            _apply_output_item(
                state=state,
                stream=stream,
                item=item,
                close_text=True,
            )
        return False

    if _event_type_matches(
        event_type,
        (
            "response.function_call_arguments.delta",
            "response.output_item.function_call_arguments.delta",
            "function_call_arguments.delta",
        ),
    ):
        call_id = _event_call_id(event)
        if not call_id:
            return False

        content_index = _ensure_tool_call(
            state=state,
            stream=stream,
            call_id=call_id,
            name=_event_tool_name(event),
        )
        delta = _as_str(event.get("delta")) or ""
        if delta:
            state.tool_arg_buffers[call_id] = (
                state.tool_arg_buffers.get(call_id, "") + delta
            )
            stream.push(
                {
                    "type": "toolcall_delta",
                    "content_index": content_index,
                    "delta": delta,
                    "partial": state.partial,
                }
            )
        return False

    if _event_type_matches(
        event_type,
        (
            "response.function_call_arguments.done",
            "response.output_item.function_call_arguments.done",
            "function_call_arguments.done",
        ),
    ):
        call_id = _event_call_id(event)
        if not call_id:
            return False

        content_index = _ensure_tool_call(
            state=state,
            stream=stream,
            call_id=call_id,
            name=_event_tool_name(event),
        )
        tool_call = cast(ToolCall, state.partial.content[content_index])
        args_str = _as_str(event.get("arguments")) or state.tool_arg_buffers.get(
            call_id,
            "",
        )
        tool_call.arguments = _extract_tool_call_arguments(args_str)
        _emit_tool_end_if_needed(state, stream, call_id)
        return False

    if _event_type_matches(event_type, ("response.completed", "completed")):
        response = event.get("response")
        _close_open_text_blocks(state, stream)
        _close_open_tool_calls(state, stream)
        if isinstance(response, Mapping):
            final_message = _assistant_from_openai_response(model, response)
        else:
            final_message = _clone_assistant_message(state.partial)
        _push_terminal_message(stream, final_message)
        return True

    if _event_type_matches(
        event_type,
        ("response.failed", "response.error", "response.cancelled", "error"),
    ):
        error_message = _error_message_from_event(event)
        if "cancel" in event_type:
            stream.push(
                {
                    "type": "error",
                    "reason": "aborted",
                    "error": _assistant_error_message(
                        model=model,
                        stop_reason="aborted",
                        error_message=error_message,
                    ),
                }
            )
        else:
            stream.push(
                {
                    "type": "error",
                    "reason": "error",
                    "error": _assistant_error_message(
                        model=model,
                        stop_reason="error",
                        error_message=error_message,
                    ),
                }
            )
        return True

    return False


def _apply_output_item(
    *,
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
    item: Mapping[str, Any],
    close_text: bool,
) -> None:
    item_type = _as_str(item.get("type"))
    if item_type == "function_call":
        call_id = _as_str(item.get("call_id")) or _as_str(item.get("id"))
        if not call_id:
            return
        content_index = _ensure_tool_call(
            state=state,
            stream=stream,
            call_id=call_id,
            name=_as_str(item.get("name")),
        )
        tool_call = cast(ToolCall, state.partial.content[content_index])
        args_raw = item.get("arguments")
        if isinstance(args_raw, Mapping):
            tool_call.arguments = {str(key): value for key, value in args_raw.items()}
        elif isinstance(args_raw, str):
            state.tool_arg_buffers[call_id] = args_raw
        return

    if item_type != "message":
        return

    for part in _as_mapping_list(item.get("content")):
        part_type = _as_str(part.get("type"))
        if part_type not in {"output_text", "text"}:
            continue

        text = _as_str(part.get("text"))
        if not text:
            continue

        event = {
            "output_index": _as_int(item.get("output_index")),
            "content_index": _as_int(part.get("index")),
        }
        content_index = _ensure_text_block(
            state=state,
            stream=stream,
            event=event,
        )
        text_block = cast(TextContent, state.partial.content[content_index])
        delta = _trailing_text_delta(current=text_block.text, observed=text)
        if delta:
            text_block.text += delta
            stream.push(
                {
                    "type": "text_delta",
                    "content_index": content_index,
                    "delta": delta,
                    "partial": state.partial,
                }
            )
        if close_text:
            _emit_text_end_if_needed(state, stream, content_index)


def _trailing_text_delta(*, current: str, observed: str) -> str:
    if not observed:
        return ""
    if not current:
        return observed
    if observed.startswith(current):
        return observed[len(current) :]
    if current.startswith(observed):
        return ""

    max_overlap = min(len(current), len(observed))
    for overlap in range(max_overlap, 0, -1):
        if current.endswith(observed[:overlap]):
            return observed[overlap:]
    return observed


def _ensure_text_block(
    *,
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
    event: Mapping[str, Any],
) -> int:
    key = (_as_int(event.get("output_index")), _as_int(event.get("content_index")))
    existing = state.text_indices.get(key)
    if existing is not None:
        return existing

    content_index = len(state.partial.content)
    state.partial.content.append(TextContent(text=""))
    state.text_indices[key] = content_index
    stream.push(
        {
            "type": "text_start",
            "content_index": content_index,
            "partial": state.partial,
        }
    )
    return content_index


def _emit_text_end_if_needed(
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
    content_index: int,
) -> None:
    if content_index in state.closed_text_indices:
        return

    content = state.partial.content[content_index]
    if not isinstance(content, TextContent):
        return

    stream.push(
        {
            "type": "text_end",
            "content_index": content_index,
            "content": content.text,
            "partial": state.partial,
        }
    )
    state.closed_text_indices.add(content_index)


def _close_open_text_blocks(
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
) -> None:
    for content_index in list(state.text_indices.values()):
        _emit_text_end_if_needed(state, stream, content_index)


def _ensure_tool_call(
    *,
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
    call_id: str,
    name: str | None,
) -> int:
    existing = state.tool_indices.get(call_id)
    if existing is not None:
        tool_call = state.partial.content[existing]
        if isinstance(tool_call, ToolCall) and name:
            tool_call.name = name
        return existing

    content_index = len(state.partial.content)
    tool_call = ToolCall(id=call_id, name=name or "tool", arguments={})
    state.partial.content.append(tool_call)
    state.tool_indices[call_id] = content_index
    state.tool_arg_buffers.setdefault(call_id, "")
    stream.push(
        {
            "type": "toolcall_start",
            "content_index": content_index,
            "partial": state.partial,
        }
    )
    return content_index


def _emit_tool_end_if_needed(
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
    call_id: str,
) -> None:
    if call_id in state.closed_tool_call_ids:
        return

    content_index = state.tool_indices.get(call_id)
    if content_index is None:
        return

    content = state.partial.content[content_index]
    if not isinstance(content, ToolCall):
        return

    stream.push(
        {
            "type": "toolcall_end",
            "content_index": content_index,
            "tool_call": content,
            "partial": state.partial,
        }
    )
    state.closed_tool_call_ids.add(call_id)


def _close_open_tool_calls(
    state: _OpenAIStreamingState,
    stream: AssistantMessageEventStream,
) -> None:
    for call_id, content_index in state.tool_indices.items():
        if call_id in state.closed_tool_call_ids:
            continue

        content = state.partial.content[content_index]
        if isinstance(content, ToolCall) and not content.arguments:
            args_str = state.tool_arg_buffers.get(call_id, "")
            if args_str:
                content.arguments = _extract_tool_call_arguments(args_str)

        _emit_tool_end_if_needed(state, stream, call_id)


def _event_type_matches(event_type: str, patterns: Sequence[str]) -> bool:
    return any(event_type == pattern or event_type.endswith(pattern) for pattern in patterns)


def _event_call_id(event: Mapping[str, Any]) -> str | None:
    call_id = _as_str(event.get("call_id"))
    if call_id:
        return call_id
    return _as_str(event.get("item_id")) or _as_str(event.get("id"))


def _event_tool_name(event: Mapping[str, Any]) -> str | None:
    return _as_str(event.get("name"))


def _error_message_from_event(event: Mapping[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, Mapping):
        message = _as_str(error.get("message"))
        if message:
            return message

    message = _as_str(event.get("message"))
    if message:
        return message
    return "OpenAI response failed."


def _new_partial_message(model: Model) -> AssistantMessage:
    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason="stop",
    )


def _clone_assistant_message(message: AssistantMessage) -> AssistantMessage:
    cloned_content: list[AssistantContentBlock] = []
    for block in message.content:
        if isinstance(block, TextContent):
            cloned_content.append(
                TextContent(
                    text=block.text,
                    text_signature=block.text_signature,
                )
            )
            continue

        if isinstance(block, ToolCall):
            cloned_content.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.arguments),
                    thought_signature=block.thought_signature,
                )
            )

    return AssistantMessage(
        content=cloned_content,
        api=message.api,
        provider=message.provider,
        model=message.model,
        usage=message.usage,
        stop_reason=message.stop_reason,
        error_message=message.error_message,
    )


def _resolve_api_key(request: PiAIRequest, api_key_env: str) -> str:
    if request.api_key:
        return request.api_key

    api_key = os.getenv(api_key_env)
    if api_key:
        return api_key

    raise RuntimeError(
        "Missing OpenAI API key. Set request.api_key or OPENAI_API_KEY."
    )


def _build_openai_payload(request: PiAIRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model.id,
        "input": _to_openai_input(request.context),
    }

    if request.context.system_prompt:
        payload["instructions"] = request.context.system_prompt

    tools = _to_openai_tools(request.context.tools)
    if tools:
        payload["tools"] = tools

    effort = _map_reasoning_effort(request.reasoning)
    if effort is not None:
        payload["reasoning"] = {"effort": effort}

    if request.session_id:
        payload["metadata"] = {"session_id": request.session_id}

    return payload


def _to_openai_input(context: LlmContext) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for message in context.messages:
        if isinstance(message, UserMessage):
            items.append(_to_openai_user_item(message))
            continue

        if isinstance(message, AssistantMessage):
            items.extend(_to_openai_assistant_items(message))
            continue

        if isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _tool_result_text(message),
                }
            )

    return items


def _to_openai_user_item(message: UserMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}

    content_items: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextContent):
            content_items.append({"type": "input_text", "text": block.text})
        elif isinstance(block, ImageContent):
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{block.mime_type};base64,{block.data}",
                }
            )

    if not content_items:
        content_items.append({"type": "input_text", "text": ""})

    return {"role": "user", "content": content_items}


def _to_openai_assistant_items(message: AssistantMessage) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    text_chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContent) and block.text
    ]
    if text_chunks:
        items.append({"role": "assistant", "content": "\n".join(text_chunks)})

    for block in message.content:
        if isinstance(block, ToolCall):
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(
                        block.arguments,
                        separators=(",", ":"),
                    ),
                }
            )

    return items


def _to_openai_tools(tools: Sequence[AgentTool] | None) -> list[dict[str, Any]]:
    if not tools:
        return []

    payload_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = _coerce_json_schema(tool.parameters)
        payload_tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            }
        )
    return payload_tools


def _coerce_json_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}

    schema_dict = dict(schema)
    if "type" not in schema_dict:
        schema_dict["type"] = "object"
    return schema_dict


def _assistant_from_openai_response(
    model: Model,
    response: Mapping[str, Any],
) -> AssistantMessage:
    content: list[AssistantContentBlock] = []
    for item in _as_mapping_list(response.get("output")):
        item_type = _as_str(item.get("type"))
        if item_type == "message":
            content.extend(_extract_text_content(item))
            continue

        if item_type == "function_call":
            tool_call = _extract_tool_call(item)
            if tool_call is not None:
                content.append(tool_call)

    error_message = _extract_error_message(response)
    stop_reason = _derive_stop_reason(content, response, error_message)
    if not content:
        content = [TextContent(text="")]

    usage = _extract_usage(response.get("usage"))
    return AssistantMessage(
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=usage,
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _extract_text_content(item: Mapping[str, Any]) -> list[TextContent]:
    blocks: list[TextContent] = []
    for part in _as_mapping_list(item.get("content")):
        part_type = _as_str(part.get("type"))
        if part_type not in {"output_text", "text"}:
            continue

        text = _as_str(part.get("text"))
        if text:
            blocks.append(TextContent(text=text))
    return blocks


def _extract_tool_call(item: Mapping[str, Any]) -> ToolCall | None:
    call_id = _as_str(item.get("call_id")) or _as_str(item.get("id"))
    name = _as_str(item.get("name"))
    if not call_id or not name:
        return None

    arguments = _extract_tool_call_arguments(item.get("arguments"))
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _extract_tool_call_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items()}

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        if isinstance(parsed, Mapping):
            return {str(key): value for key, value in parsed.items()}
    return {}


def _extract_error_message(response: Mapping[str, Any]) -> str | None:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = _as_str(error.get("message"))
        if message:
            return message

    status = _as_str(response.get("status"))
    if status == "failed":
        return "OpenAI response failed."
    if status == "cancelled":
        return "OpenAI response cancelled."
    return None


def _derive_stop_reason(
    content: Sequence[AssistantContentBlock],
    response: Mapping[str, Any],
    error_message: str | None,
) -> StopReason:
    if error_message:
        status = _as_str(response.get("status"))
        if status == "cancelled":
            return "aborted"
        return "error"

    if any(isinstance(block, ToolCall) for block in content):
        return "toolUse"

    status = _as_str(response.get("status"))
    if status == "incomplete":
        return "length"
    return "stop"


def _extract_usage(usage_data: Any) -> Usage:
    if not isinstance(usage_data, Mapping):
        return Usage(cost=UsageCost())

    input_tokens = _as_int(usage_data.get("input_tokens"))
    output_tokens = _as_int(usage_data.get("output_tokens"))
    total_tokens = _as_int(usage_data.get("total_tokens"))

    input_details = usage_data.get("input_tokens_details")
    cache_read = 0
    if isinstance(input_details, Mapping):
        cache_read = _as_int(input_details.get("cached_tokens"))

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=0,
        total_tokens=total_tokens,
        cost=UsageCost(),
    )


def _done_reason_for_message(message: AssistantMessage) -> DoneReason:
    if message.stop_reason == "toolUse":
        return "toolUse"
    if message.stop_reason == "length":
        return "length"
    return "stop"


def _assistant_error_message(
    *,
    model: Model,
    stop_reason: StopReason,
    error_message: str,
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(cost=UsageCost()),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def _map_reasoning_effort(reasoning: ThinkingLevel | None) -> str | None:
    if reasoning in {None, "off"}:
        return None
    if reasoning in {"minimal", "low"}:
        return "low"
    if reasoning == "medium":
        return "medium"
    return "high"


def _tool_result_text(message: ToolResultMessage) -> str:
    text_chunks = [
        block.text
        for block in message.content
        if isinstance(block, TextContent)
    ]
    return "\n".join(text_chunks).strip()


def _as_mapping_list(raw: Any) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        return []

    mappings: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            mappings.append(item)
    return mappings


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _event_to_mapping(event: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)

    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict(warnings=False)
        except TypeError:
            dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    fields = (
        "type",
        "delta",
        "text",
        "call_id",
        "item_id",
        "id",
        "name",
        "output_index",
        "content_index",
        "arguments",
        "item",
        "output_item",
        "response",
        "error",
        "message",
    )
    mapped: dict[str, Any] = {}
    for field_name in fields:
        if hasattr(event, field_name):
            mapped[field_name] = getattr(event, field_name)
    return mapped


def _response_to_mapping(response: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)

    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict(warnings=False)
        except TypeError:
            dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(warnings=False)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    raise RuntimeError(
        "Unsupported OpenAI response type. "
        f"Expected mapping-like object, got: {type(response).__name__}"
    )


def _load_async_openai_class() -> Any:
    try:
        openai_module = importlib.import_module("openai")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "OpenAI provider requires the `openai` package. "
            "Install it with: uv add openai"
        ) from exc

    async_openai = getattr(openai_module, "AsyncOpenAI", None)
    if async_openai is None:
        raise RuntimeError(
            "Installed `openai` package does not expose AsyncOpenAI."
        )
    return async_openai


async def _stream_openai_events(
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
) -> AsyncIterator[Mapping[str, Any]]:
    async_openai = _load_async_openai_class()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    async with async_openai(**client_kwargs) as client:
        stream_factory = getattr(client.responses, "stream", None)
        if not callable(stream_factory):
            raise RuntimeError(
                "Installed `openai` package does not support responses.stream."
            )

        async with stream_factory(**payload) as response_stream:
            async for event in response_stream:
                yield _event_to_mapping(event)

            final_response = await response_stream.get_final_response()

    yield {"type": "response.completed", "response": _response_to_mapping(final_response)}


async def _request_openai_response(
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
) -> dict[str, Any]:
    async_openai = _load_async_openai_class()

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    async with async_openai(**client_kwargs) as client:
        response = await client.responses.create(**payload)

    return _response_to_mapping(response)


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value
