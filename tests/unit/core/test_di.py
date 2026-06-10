"""Tests for core.di module."""
from __future__ import annotations

from typing import Optional

import pytest

from product.core.di import Container, inject, provider


# ── Fixtures ───────────────────────────────────────────────────────────────

class _Engine:
    def __init__(self, name: str = "default") -> None:
        self.name = name


class _Database:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine


# ── Tests ──────────────────────────────────────────────────────────────────

class TestContainer:
    def test_register_and_resolve(self) -> None:
        c = Container()
        c.register(int, factory=lambda: 42)
        assert c.resolve(int) == 42

    def test_register_with_concrete(self) -> None:
        c = Container()
        c.register(_Engine, concrete=_Engine)
        engine = c.resolve(_Engine)
        assert isinstance(engine, _Engine)
        assert engine.name == "default"

    def test_resolve_raises_on_missing(self) -> None:
        c = Container()
        with pytest.raises(KeyError):
            c.resolve(float)

    def test_singleton_with_instance(self) -> None:
        c = Container()
        obj = _Engine("shared")
        c.singleton(_Engine, instance=obj)
        assert c.resolve(_Engine) is obj

    def test_singleton_with_factory(self) -> None:
        c = Container()
        c.singleton(_Engine, factory=lambda: _Engine("singleton"))
        e1 = c.resolve(_Engine)
        e2 = c.resolve(_Engine)
        assert e1 is e2
        assert e1.name == "singleton"

    def test_override(self) -> None:
        c = Container()
        c.register(int, factory=lambda: 1)
        c.override(int, 99)
        assert c.resolve(int) == 99

    def test_clear(self) -> None:
        c = Container()
        c.register(int, factory=lambda: 1)
        c.clear()
        assert not c.has(int)

    def test_has(self) -> None:
        c = Container()
        assert not c.has(int)
        c.register(int, factory=lambda: 0)
        assert c.has(int)

    def test_parent_container(self) -> None:
        parent = Container()
        parent.register(int, factory=lambda: 42)
        child = Container(parent=parent)
        assert child.resolve(int) == 42
        assert child.has(int)

    def test_child_override_parent(self) -> None:
        parent = Container()
        parent.register(int, factory=lambda: 1)
        child = Container(parent=parent)
        child.register(int, factory=lambda: 2)
        assert child.resolve(int) == 2
        assert parent.resolve(int) == 1


class TestInjectDecorator:
    def test_inject_resolves_parameter(self) -> None:
        c = Container()
        c.singleton(int, instance=99)

        @inject(c)
        def target(x: int) -> int:
            return x * 2

        assert target() == 198

    def test_inject_does_not_override_passed_arg(self) -> None:
        c = Container()
        c.singleton(int, instance=99)

        @inject(c)
        def target(x: int) -> int:
            return x

        assert target(x=10) == 10

    def test_inject_ignores_parameters_with_defaults(self) -> None:
        c = Container()

        @inject(c)
        def target(x: int = 5) -> int:
            return x

        assert target() == 5

    def test_inject_works_with_multiple_params(self) -> None:
        c = Container()
        c.singleton(int, instance=10)
        c.singleton(str, instance="hello")

        @inject(c)
        def target(a: int, b: str, c: Optional[str] = None) -> str:
            return f"{a} {b}"

        assert target() == "10 hello"


class TestProviderDecorator:
    def test_provider_registers_class(self) -> None:
        c = Container()

        @provider(c, _Engine)
        class MyEngine(_Engine):
            pass

        engine = c.resolve(_Engine)
        assert isinstance(engine, MyEngine)