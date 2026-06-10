"""Simple dependency injection container.

Provides:
- ``Container`` — a mutable registry for factories / singletons.
- ``inject`` — decorator that resolves parameters from the container.
- ``provider`` — decorator that registers a factory in the container.

Usage::

    container = Container()
    container.register(LLMProvider, factory=OpenAIProvider)
    container.singleton(Settings, Settings())

    @inject(container)
    async def handle(llm: LLMProvider, settings: Settings) -> None:
        ...
"""
from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    get_type_hints,
)

T = TypeVar("T")


class _Registration:
    """Internal registration record."""

    __slots__ = ("factory", "instance", "singleton")

    def __init__(self, factory: Callable[[], Any], singleton: bool = False) -> None:
        self.factory = factory
        self.instance: Any = None
        self.singleton = singleton


class Container:
    """Lightweight DI container.

    Example::

        c = Container()
        c.register(int, lambda: 42)
        c.singleton(str, "hello")
        assert c.resolve(int) == 42
    """

    def __init__(self, parent: Optional[Container] = None) -> None:
        self._registrations: Dict[Type, _Registration] = {}
        self._parent = parent

    def register(
        self,
        abstract: Type[T],
        *,
        factory: Optional[Callable[[], T]] = None,
        concrete: Optional[Type[T]] = None,
    ) -> None:
        """Register a type with a factory function or concrete class.

        Each call to :meth:`resolve` will invoke the factory.
        """
        if factory is None and concrete is not None:
            factory = lambda: concrete()  # noqa: E731
        if factory is None:
            raise ValueError("Provide either 'factory' or 'concrete'")
        self._registrations[abstract] = _Registration(factory, singleton=False)

    def singleton(
        self,
        abstract: Type[T],
        instance: Optional[T] = None,
        *,
        factory: Optional[Callable[[], T]] = None,
    ) -> None:
        """Register a singleton.

        If *instance* is given it is returned on every call.
        If *factory* is given it is called once and the result cached.
        """
        if instance is not None:
            reg = _Registration(lambda: instance, singleton=True)
            reg.instance = instance
            self._registrations[abstract] = reg
        elif factory is not None:
            self._registrations[abstract] = _Registration(factory, singleton=True)
        else:
            raise ValueError("Provide either 'instance' or 'factory'")

    def resolve(self, abstract: Type[T]) -> T:
        """Resolve an instance of *abstract*."""
        reg = self._registrations.get(abstract)
        if reg is None:
            if self._parent is not None:
                return self._parent.resolve(abstract)
            raise KeyError(f"No registration for {abstract.__name__}")

        if reg.singleton:
            if reg.instance is None:
                reg.instance = reg.factory()
            return reg.instance

        return reg.factory()

    def override(self, abstract: Type[T], instance: T) -> None:
        """Temporarily override a registration (useful in tests)."""
        self._registrations[abstract] = _Registration(lambda: instance, singleton=True)
        self._registrations[abstract].instance = instance

    def clear(self) -> None:
        """Remove all registrations."""
        self._registrations.clear()

    def has(self, abstract: Type[T]) -> bool:
        """Check if a type is registered (including parent container)."""
        if abstract in self._registrations:
            return True
        if self._parent is not None:
            return self._parent.has(abstract)
        return False


def inject(container: Container) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that resolves type-annotated parameters from the container.

    Only parameters whose type is registered in the container are injected.
    Parameters that already have a default value or are explicitly passed
    are left untouched.

    Usage::

        @inject(container)
        def handle(request: Request, llm: LLMProvider) -> Response:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        def wrapper(*args: Any, **kwargs: Any) -> T:
            bound = sig.bind_partial(*args, **kwargs)
            params = sig.parameters

            for name, param in params.items():
                if name in bound.arguments:
                    continue  # Already provided
                if param.default is not inspect.Parameter.empty:
                    continue  # Has a default, skip injection

                hint = hints.get(name)
                if hint is None:
                    continue
                # Resolve generic origin (e.g., List[str] -> list)
                origin = getattr(hint, "__origin__", hint)
                if container.has(origin):
                    bound.arguments[name] = container.resolve(origin)

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def provider(container: Container, abstract: Type[T]) -> Callable[[Type[T]], Type[T]]:
    """Class decorator that auto-registers the class as a provider.

    Usage::

        @provider(container, LLMProvider)
        class OpenAIProvider(LLMProvider):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        container.register(abstract, concrete=cls)
        return cls

    return decorator