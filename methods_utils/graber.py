from typing import Any, Hashable

_default = object()


class Grabber:
    _store: dict[Hashable, Any] = {}

    @classmethod
    def __getitem__(cls, key: Hashable) -> Any:
        return cls._store[key]

    @classmethod
    def __setitem__(cls, key: Hashable, value: Any) -> None:
        cls._store[key] = value

    @classmethod
    def __delitem__(cls, key: Hashable) -> None:
        del cls._store[key]

    @classmethod
    def __contains__(cls, key: Hashable) -> bool:
        return key in cls._store

    @classmethod
    def __len__(cls) -> int:
        return len(cls._store)

    @classmethod
    def __iter__(cls):
        return iter(cls._store)

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()

    @classmethod
    def pop(cls, key: Hashable, default: Any = _default) -> Any:
        if default is _default:
            return cls._store.pop(key)
        return cls._store.pop(key, default)

    @classmethod
    def get(cls, key: Hashable, default: Any = _default) -> Any:
        if default is _default:
            return cls._store.get(key)
        return cls._store.get(key, default)

    @classmethod
    def keys(cls) -> Any:
        return cls._store.keys()

    @classmethod
    def items(cls) -> Any:
        return cls._store.items()

    def __str__(self) -> str:
        return str(self._store)

    def __repr__(self) -> str:
        return f"Graber({repr(self._store)})"


grabber = Grabber()

# Deprecated
Graber = Grabber
graber = grabber
