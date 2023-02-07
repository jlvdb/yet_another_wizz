from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Callable


class Registry(Mapping):

    _register = {}

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._register})"

    def __getitem__(self, name: str) -> Callable:
        try:
            return self._register[name]
        except KeyError as e:
            raise KeyError(f"no item named '{name}' registered") from e

    def __iter__(self) -> Iterator[Callable]:
        return iter(self._register)

    def __len__(self) -> int:
        return len(self._register)

    def register(self, obj):
        self._register[obj.__name__] = obj
        return obj
