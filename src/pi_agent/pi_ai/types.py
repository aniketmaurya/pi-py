from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..agent_core.types import AssistantStream, LlmContext, Model, ThinkingLevel


@dataclass(slots=True, frozen=True)
class PiAIRequest:
    model: Model
    context: LlmContext
    reasoning: ThinkingLevel | None = None
    api_key: str | None = None
    session_id: str | None = None
    thinking_budgets: Mapping[str, int] | None = None
    max_retry_delay_ms: int | None = None


class Provider(Protocol):
    async def stream(
        self,
        request: PiAIRequest,
        abort_event: asyncio.Event | None = None,
    ) -> AssistantStream: ...
