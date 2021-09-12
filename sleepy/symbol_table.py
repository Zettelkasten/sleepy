from __future__ import annotations

import itertools
from typing import TypeVar, Dict, Mapping, Iterator

K = TypeVar("K")
V = TypeVar("V")

class Stub(Mapping[K, V]):
  def __len__(self) -> int:
    return 0

  def __iter__(self) -> Iterator[V]:
    return iter(())

  def __getitem__(self, key: K) -> V:
    raise KeyError()

STUB = Stub()

class HierarchicalDict(Mapping[K, V]):
  def __init__(self, parent: Mapping[K, V] = None, init_with: Dict[K, V] = None):
    self.parent = parent if parent is not None else STUB
    self.underlying_dict: Dict[K, V] = {} if init_with is None else init_with

  def __getitem__(self, key: K) -> V:
    value = self.underlying_dict.get(key)
    return value if value is not None else self.parent[key]

  def __setitem__(self, key: K, value: V):
    self.underlying_dict[key] = value

  def __contains__(self, key: K) -> bool:
    return (key in self.underlying_dict) or (key in self.parent)

  def __iter__(self) -> Iterator[V]:
    return itertools.chain(self.underlying_dict.__iter__(), self.parent.__iter__())

  def __len__(self) -> int:
    return len(self.underlying_dict) + len(self.parent)

