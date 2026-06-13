"""Agent memory backends.

Provides:
- ``Memory`` — abstract interface for conversation history.
- ``InMemoryMemory`` — simple list-based memory (default).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from product.providers.base import Message


class Memory(ABC):
    """Abstract interface for agent conversation memory."""

    @abstractmethod
    def add_message(self, message: Message) -> None: ...

    @abstractmethod
    def get_history(self) -> list[Message]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class InMemoryMemory(Memory):
    """Simple in-memory conversation memory."""

    def __init__(self, max_messages: int | None = None) -> None:
        self._messages: list[Message] = []
        self._max_messages = max_messages

    def add_message(self, message: Message) -> None:
        self._messages.append(message)
        if self._max_messages is not None:
            while len(self._messages) > self._max_messages:
                self._messages.pop(0)

    def get_history(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
