import typing


class ModuleRegistry:
    def __init__(self):
        self._registry: dict[str, typing.Callable[[], type]] = {}

    def register(self, name: str):
        def wrapper(loader: typing.Callable[[], type]):
            assert name not in self._registry, f"'{name}' already registered!"
            self._registry[name] = loader
            return loader

        return wrapper

    def get(self, name: str) -> type:
        if name not in self._registry:
            raise KeyError(f"'{name}' not registered. Available: {list(self._registry.keys())}")
        return self._registry[name]()

    def list_available(self) -> list[str]:
        return list(self._registry.keys())
